import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


IPERF_INTERVAL_RE = re.compile(
    r"\[\s*\d+\]\s+"
    r"(?P<start>\d+(?:\.\d+)?)\s*-\s*(?P<end>\d+(?:\.\d+)?)\s+sec\s+"
    r".+?\s+(?P<bw>\d+(?:\.\d+)?)\s+(?P<unit>[KMG])bits/sec",
    re.IGNORECASE,
)


def unit_to_mbps(value: float, unit: str) -> float:
    unit = unit.upper()
    if unit == "G":
        return value * 1000.0
    if unit == "M":
        return value
    if unit == "K":
        return value / 1000.0
    return value / 1_000_000.0


def load_meta(run_dir: Path) -> dict:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8", errors="ignore"))


def load_scenario(run_dir: Path, meta: dict) -> pd.DataFrame:
    scenario_path = run_dir / "scenario.csv"
    if not scenario_path.exists():
        raise FileNotFoundError(f"Missing scenario.csv: {scenario_path}")
    df = pd.read_csv(scenario_path)
    required = {"ts", "true_bw", "true_loss_rate"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {scenario_path}: {sorted(missing)}")
    if "stage_duration_s" not in df.columns:
        df["stage_duration_s"] = df["ts"].shift(-1) - df["ts"]
        df["stage_duration_s"] = df["stage_duration_s"].fillna(df["stage_duration_s"].median())

    df = df.copy()
    df["stage_index"] = range(1, len(df) + 1)
    df["start_ts"] = df["ts"].astype(float)
    df["end_ts"] = df["start_ts"].shift(-1)
    df["end_ts"] = df["end_ts"].fillna(df["start_ts"] + df["stage_duration_s"].astype(float))

    stable_sleep = float(meta.get("initial_stable_sleep", 8.0))
    first_ts = float(df["start_ts"].iloc[0])
    df["start_rel_s"] = stable_sleep + (df["start_ts"] - first_ts)
    df["end_rel_s"] = stable_sleep + (df["end_ts"] - first_ts)
    return df


def load_iperf(log_path: Path) -> pd.DataFrame:
    rows = []
    if not log_path.exists():
        return pd.DataFrame(columns=["start_s", "end_s", "mid_s", "throughput_mbps"])
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = IPERF_INTERVAL_RE.search(line)
        if not match:
            continue
        start_s = float(match.group("start"))
        end_s = float(match.group("end"))
        if start_s == 0.0 and end_s > 2.0:
            continue
        rows.append(
            {
                "start_s": start_s,
                "end_s": end_s,
                "mid_s": (start_s + end_s) / 2.0,
                "throughput_mbps": unit_to_mbps(float(match.group("bw")), match.group("unit")),
            }
        )
    return pd.DataFrame(rows)


def load_link_state(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def mean_or_nan(series: pd.Series) -> float:
    if series is None or len(series) == 0:
        return float("nan")
    return float(series.mean())


def quantile_or_nan(series: pd.Series, q: float) -> float:
    if series is None or len(series) == 0:
        return float("nan")
    return float(series.quantile(q))


def value_counts_fraction(series: pd.Series, value: str) -> float:
    if series is None or len(series) == 0:
        return float("nan")
    return float((series == value).mean())


def stage_iperf_rows(iperf_df: pd.DataFrame, start_s: float, end_s: float) -> pd.DataFrame:
    if iperf_df.empty:
        return iperf_df
    return iperf_df[(iperf_df["mid_s"] >= start_s) & (iperf_df["mid_s"] < end_s)]


def stage_link_rows(link_df: pd.DataFrame, start_ts: float, end_ts: float) -> pd.DataFrame:
    if link_df.empty or "ts" not in link_df.columns:
        return link_df.iloc[0:0]
    return link_df[(link_df["ts"] >= start_ts) & (link_df["ts"] < end_ts)]


def classify_transition(prev_bw: float, cur_bw: float, prev_loss: float, cur_loss: float) -> str:
    bw_delta = cur_bw - prev_bw
    loss_delta = cur_loss - prev_loss
    parts = []
    if abs(bw_delta) >= 1e-6:
        parts.append("bw_up" if bw_delta > 0 else "bw_down")
    if abs(loss_delta) >= 1e-9:
        parts.append("loss_up" if loss_delta > 0 else "loss_down")
    return "+".join(parts) if parts else "stable"


def recovery_time_after_switch(
    iperf_df: pd.DataFrame,
    switch_s: float,
    target_bw: float,
    direction: str,
    horizon_s: float,
    ratio: float,
) -> float:
    if iperf_df.empty:
        return float("nan")
    window = iperf_df[(iperf_df["mid_s"] >= switch_s) & (iperf_df["mid_s"] < switch_s + horizon_s)].copy()
    if window.empty:
        return float("nan")

    if "bw_up" in direction:
        ok = window["throughput_mbps"] >= target_bw * ratio
    elif "bw_down" in direction:
        ok = window["throughput_mbps"] <= target_bw / max(ratio, 1e-6)
    else:
        ok = window["throughput_mbps"] >= target_bw * ratio
    hits = window.loc[ok]
    if hits.empty:
        return float("nan")
    return float(hits["mid_s"].iloc[0] - switch_s)


def deficit_area_after_switch(
    iperf_df: pd.DataFrame,
    switch_s: float,
    target_bw: float,
    horizon_s: float,
) -> float:
    if iperf_df.empty:
        return float("nan")
    window = iperf_df[(iperf_df["mid_s"] >= switch_s) & (iperf_df["mid_s"] < switch_s + horizon_s)]
    if window.empty:
        return float("nan")
    deficit = (target_bw - window["throughput_mbps"]).clip(lower=0.0)
    return float(deficit.sum())


def discover_experiment_dirs(scene_dir: Path) -> list[Path]:
    if (scene_dir / "scenario.csv").exists() and (scene_dir / "iperf_d.log").exists():
        return [scene_dir]
    exp_dirs = []
    for child in sorted(scene_dir.iterdir()):
        if child.is_dir() and (child / "scenario.csv").exists() and (child / "iperf_d.log").exists():
            exp_dirs.append(child)
    return exp_dirs


def discover_runs(root: Path):
    for method_dir in sorted(root.iterdir()):
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        for scene_dir in sorted(method_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            exp_dirs = discover_experiment_dirs(scene_dir)
            for idx, run_dir in enumerate(exp_dirs, start=1):
                meta = load_meta(run_dir)
                scene_id = str(meta.get("scene_id") or meta.get("scene_name") or scene_dir.name)
                exp_id = run_dir.name if run_dir != scene_dir else f"exp_{idx}"
                yield method, scene_id, exp_id, run_dir


def analyze_run(method: str, scene_id: str, exp_id: str, run_dir: Path, transition_horizon_s: float, recovery_ratio: float):
    meta = load_meta(run_dir)
    scenario = load_scenario(run_dir, meta)
    iperf = load_iperf(run_dir / "iperf_d.log")
    link_state = load_link_state(run_dir / "link_state.csv")

    overall = {
        "method": method,
        "scene_id": scene_id,
        "exp_id": exp_id,
        "run_dir": str(run_dir),
        "num_stages": int(len(scenario)),
        "iperf_mean_mbps": mean_or_nan(iperf["throughput_mbps"]) if not iperf.empty else float("nan"),
        "iperf_p95_mbps": quantile_or_nan(iperf["throughput_mbps"], 0.95) if not iperf.empty else float("nan"),
        "iperf_duration_s": float(iperf["end_s"].max()) if not iperf.empty else float("nan"),
        "mean_rtt_s": mean_or_nan(link_state["rtt"]) if "rtt" in link_state.columns else float("nan"),
        "p95_rtt_s": quantile_or_nan(link_state["rtt"], 0.95) if "rtt" in link_state.columns else float("nan"),
        "mean_queue_delay_s": mean_or_nan(link_state["queue_delay"]) if "queue_delay" in link_state.columns else float("nan"),
        "p95_queue_delay_s": quantile_or_nan(link_state["queue_delay"], 0.95) if "queue_delay" in link_state.columns else float("nan"),
        "mean_fec_ratio": mean_or_nan(link_state["decision_fec_ratio"]) if "decision_fec_ratio" in link_state.columns else float("nan"),
    }

    stage_rows = []
    for _, stage in scenario.iterrows():
        ip_rows = stage_iperf_rows(iperf, float(stage["start_rel_s"]), float(stage["end_rel_s"]))
        ls_rows = stage_link_rows(link_state, float(stage["start_ts"]), float(stage["end_ts"]))
        row = {
            "method": method,
            "scene_id": scene_id,
            "exp_id": exp_id,
            "stage_index": int(stage["stage_index"]),
            "true_bw_mbps": float(stage["true_bw"]),
            "true_loss_rate": float(stage["true_loss_rate"]),
            "stage_duration_s": float(stage["end_rel_s"] - stage["start_rel_s"]),
            "start_rel_s": float(stage["start_rel_s"]),
            "end_rel_s": float(stage["end_rel_s"]),
            "iperf_mean_mbps": mean_or_nan(ip_rows["throughput_mbps"]) if not ip_rows.empty else float("nan"),
            "iperf_p95_mbps": quantile_or_nan(ip_rows["throughput_mbps"], 0.95) if not ip_rows.empty else float("nan"),
            "target_utilization": (
                mean_or_nan(ip_rows["throughput_mbps"]) / float(stage["true_bw"]) if not ip_rows.empty and float(stage["true_bw"]) > 0 else float("nan")
            ),
            "mean_rtt_s": mean_or_nan(ls_rows["rtt"]) if "rtt" in ls_rows.columns else float("nan"),
            "p95_rtt_s": quantile_or_nan(ls_rows["rtt"], 0.95) if "rtt" in ls_rows.columns else float("nan"),
            "mean_queue_delay_s": mean_or_nan(ls_rows["queue_delay"]) if "queue_delay" in ls_rows.columns else float("nan"),
            "p95_queue_delay_s": quantile_or_nan(ls_rows["queue_delay"], 0.95) if "queue_delay" in ls_rows.columns else float("nan"),
            "mean_fec_ratio": mean_or_nan(ls_rows["decision_fec_ratio"]) if "decision_fec_ratio" in ls_rows.columns else float("nan"),
            "brake_fraction": value_counts_fraction(ls_rows["decision_state"], "BRAKE") if "decision_state" in ls_rows.columns else float("nan"),
            "hold_fraction": value_counts_fraction(ls_rows["decision_state"], "HOLD") if "decision_state" in ls_rows.columns else float("nan"),
            "accel_fraction": value_counts_fraction(ls_rows["decision_state"], "ACCEL") if "decision_state" in ls_rows.columns else float("nan"),
        }
        if "sense_bw_mbps" in ls_rows.columns:
            row["sense_bw_mae_mbps"] = float((ls_rows["sense_bw_mbps"] - float(stage["true_bw"])).abs().mean()) if not ls_rows.empty else float("nan")
        if "sense_loss_rate" in ls_rows.columns:
            row["sense_loss_mae"] = float((ls_rows["sense_loss_rate"] - float(stage["true_loss_rate"])).abs().mean()) if not ls_rows.empty else float("nan")
        stage_rows.append(row)

    transition_rows = []
    for idx in range(1, len(scenario)):
        prev = scenario.iloc[idx - 1]
        cur = scenario.iloc[idx]
        direction = classify_transition(
            float(prev["true_bw"]),
            float(cur["true_bw"]),
            float(prev["true_loss_rate"]),
            float(cur["true_loss_rate"]),
        )
        switch_s = float(cur["start_rel_s"])
        pre_rows = stage_iperf_rows(iperf, max(0.0, switch_s - 3.0), switch_s)
        post_rows = stage_iperf_rows(iperf, switch_s, min(float(cur["end_rel_s"]), switch_s + transition_horizon_s))
        transition_rows.append(
            {
                "method": method,
                "scene_id": scene_id,
                "exp_id": exp_id,
                "transition_index": idx,
                "from_bw_mbps": float(prev["true_bw"]),
                "to_bw_mbps": float(cur["true_bw"]),
                "from_loss_rate": float(prev["true_loss_rate"]),
                "to_loss_rate": float(cur["true_loss_rate"]),
                "transition_type": direction,
                "switch_rel_s": switch_s,
                "pre_3s_mean_mbps": mean_or_nan(pre_rows["throughput_mbps"]) if not pre_rows.empty else float("nan"),
                "post_horizon_mean_mbps": mean_or_nan(post_rows["throughput_mbps"]) if not post_rows.empty else float("nan"),
                "recovery_time_s": recovery_time_after_switch(
                    iperf,
                    switch_s,
                    float(cur["true_bw"]),
                    direction,
                    min(transition_horizon_s, float(cur["end_rel_s"] - switch_s)),
                    recovery_ratio,
                ),
                "deficit_area_mbps_s": deficit_area_after_switch(
                    iperf,
                    switch_s,
                    float(cur["true_bw"]),
                    min(transition_horizon_s, float(cur["end_rel_s"] - switch_s)),
                ),
            }
        )

    return overall, pd.DataFrame(stage_rows), pd.DataFrame(transition_rows), scenario, iperf, link_state


def prettify_method_label(method: str) -> str:
    mapping = {
        "original": "Original PEPesc",
        "hybrid_adaptive_v2": "Hybrid Adaptive",
        "hybrid_adaptive": "Hybrid Adaptive",
        "rule_adaptive": "Rule Adaptive",
    }
    return mapping.get(method, method)


def build_mean_iperf_trace(run_items: list[tuple[Path, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for _run_dir, iperf_df in run_items:
        if iperf_df.empty:
            continue
        tmp = iperf_df[["mid_s", "throughput_mbps"]].copy()
        tmp["mid_s"] = tmp["mid_s"].round(3)
        rows.append(tmp)
    if not rows:
        return pd.DataFrame(columns=["mid_s", "throughput_mbps"])
    all_df = pd.concat(rows, ignore_index=True)
    return all_df.groupby("mid_s", as_index=False)["throughput_mbps"].mean()


def build_mean_control_trace(run_items: list[tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for scenario, link_state in run_items:
        if link_state.empty or "ts" not in link_state.columns:
            continue
        rel_t = link_state["ts"] - float(scenario["start_ts"].iloc[0]) + float(scenario["start_rel_s"].iloc[0])
        tmp = pd.DataFrame({"rel_s": rel_t.round(2)})
        if "decision_fec_ratio" in link_state.columns:
            tmp["decision_fec_ratio"] = link_state["decision_fec_ratio"]
        if "decision_pacing_mbps" in link_state.columns:
            tmp["decision_pacing_mbps"] = link_state["decision_pacing_mbps"]
        rows.append(tmp)
    if not rows:
        return pd.DataFrame()
    all_df = pd.concat(rows, ignore_index=True)
    return all_df.groupby("rel_s", as_index=False).mean(numeric_only=True)


def plot_scene_timeseries(scene_id: str, runs_by_method: dict, outpath: Path) -> None:
    if not runs_by_method:
        return
    plt.figure(figsize=(10, 5))
    plotted_target = False
    for method, items in runs_by_method.items():
        mean_trace = build_mean_iperf_trace([(run_dir, iperf_df) for run_dir, _scenario, iperf_df, _link_state in items])
        if not mean_trace.empty:
            plt.plot(mean_trace["mid_s"], mean_trace["throughput_mbps"], label=f"{prettify_method_label(method)} mean", linewidth=1.8)
        if not plotted_target and items:
            scenario = items[0][1]
            for _, stage in scenario.iterrows():
                plt.hlines(
                    y=float(stage["true_bw"]),
                    xmin=float(stage["start_rel_s"]),
                    xmax=float(stage["end_rel_s"]),
                    colors="black",
                    linestyles="dashed",
                    linewidth=1.0,
                    alpha=0.55,
                )
                plt.axvline(float(stage["start_rel_s"]), color="gray", linewidth=0.6, alpha=0.25)
            plotted_target = True
    plt.xlabel("Time (s)")
    plt.ylabel("Throughput / target bandwidth (Mbps)")
    plt.title(f"Dynamic scene: mean throughput trace")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_scene_control(scene_id: str, runs_by_method: dict, outpath: Path) -> None:
    if not runs_by_method:
        return
    plt.figure(figsize=(11, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    method_colors = {
        "original": "#4C78A8",
        "hybrid_adaptive_v2": "#E45756",
        "hybrid_adaptive": "#E45756",
        "rule_adaptive": "#54A24B",
    }
    for method, items in runs_by_method.items():
        mean_trace = build_mean_control_trace([(scenario, link_state) for _run_dir, scenario, _iperf, link_state in items])
        if mean_trace.empty:
            continue
        color = method_colors.get(method, None)
        label_prefix = prettify_method_label(method)
        if "decision_fec_ratio" in mean_trace.columns:
            ax1.plot(mean_trace["rel_s"], mean_trace["decision_fec_ratio"], label=f"{label_prefix} FEC", linewidth=1.8, linestyle="solid", alpha=0.95, color=color)
        if "decision_pacing_mbps" in mean_trace.columns:
            ax2.plot(mean_trace["rel_s"], mean_trace["decision_pacing_mbps"], label=f"{label_prefix} pacing", linewidth=1.4, linestyle=(0, (4, 2)), alpha=0.8, color=color)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("FEC ratio")
    ax2.set_ylabel("Decision pacing (Mbps)")
    ax1.grid(True, alpha=0.25)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    plt.title(f"Dynamic scene: mean adaptive control trace")
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(outpath, dpi=180)
    plt.close()


def add_bar_labels(ax, fmt: str = "{:.2f}") -> None:
    for container in ax.containers:
        labels = []
        for bar in container:
            height = bar.get_height()
            labels.append("" if pd.isna(height) else fmt.format(height))
        ax.bar_label(container, labels=labels, padding=2, fontsize=8, rotation=0)


def plot_overall_metric_bar(df: pd.DataFrame, metric_col: str, ylabel: str, outpath: Path, scale: float = 1.0, value_fmt: str = "{:.2f}") -> None:
    if metric_col not in df.columns or df.empty:
        return
    plot_df = df.copy()
    plot_df[metric_col] = plot_df[metric_col] * scale
    plot_df["label"] = plot_df.apply(lambda row: f"{row['scene_id']}\n{prettify_method_label(row['method'])}", axis=1)
    ax = plot_df.plot(x="label", y=metric_col, kind="bar", figsize=(9, 5), legend=False, color="#4C78A8")
    ax.set_xlabel("Scene / method")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    add_bar_labels(ax, value_fmt)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_transition_metric_bar(df: pd.DataFrame, metric_col: str, ylabel: str, outpath: Path, scale: float = 1.0, value_fmt: str = "{:.2f}") -> None:
    if metric_col not in df.columns or df.empty:
        return
    plot_df = df.copy()
    plot_df[metric_col] = plot_df[metric_col] * scale
    plot_df["label"] = plot_df.apply(lambda row: f"{row['scene_id']}\n{prettify_method_label(row['method'])}", axis=1)
    ax = plot_df.plot(x="label", y=metric_col, kind="bar", figsize=(9, 5), legend=False, color="#E45756")
    ax.set_xlabel("Scene / method")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    add_bar_labels(ax, value_fmt)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_stage_metric_bar(scene_id: str, stage_df: pd.DataFrame, metric_col: str, ylabel: str, outpath: Path, scale: float = 1.0, value_fmt: str = "{:.2f}") -> None:
    scene_df = stage_df[stage_df["scene_id"].astype(str) == str(scene_id)].copy()
    if scene_df.empty or metric_col not in scene_df.columns:
        return
    scene_df[metric_col] = scene_df[metric_col] * scale
    pivot = scene_df.pivot_table(index="stage_index", columns="method", values=metric_col, aggfunc="mean")
    if pivot.empty:
        return
    pivot = pivot.rename(columns={col: prettify_method_label(col) for col in pivot.columns})
    ax = pivot.plot(kind="bar", figsize=(10, 5), width=0.8)
    ax.set_xlabel("Stage index")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    add_bar_labels(ax, value_fmt)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def summarize_overall(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["method", "scene_id"], as_index=False)
        .agg(
            iperf_mean_mbps=("iperf_mean_mbps", "mean"),
            throughput_std_mbps=("iperf_mean_mbps", "std"),
            iperf_p95_mbps=("iperf_p95_mbps", "mean"),
            iperf_duration_s=("iperf_duration_s", "mean"),
            mean_rtt_s=("mean_rtt_s", "mean"),
            p95_rtt_s=("p95_rtt_s", "mean"),
            mean_queue_delay_s=("mean_queue_delay_s", "mean"),
            p95_queue_delay_s=("p95_queue_delay_s", "mean"),
            mean_fec_ratio=("mean_fec_ratio", "mean"),
            num_experiments=("exp_id", "nunique"),
        )
    )


def summarize_stage(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["method", "scene_id", "stage_index"], as_index=False)
        .agg(
            true_bw_mbps=("true_bw_mbps", "mean"),
            true_loss_rate=("true_loss_rate", "mean"),
            stage_duration_s=("stage_duration_s", "mean"),
            start_rel_s=("start_rel_s", "mean"),
            end_rel_s=("end_rel_s", "mean"),
            iperf_mean_mbps=("iperf_mean_mbps", "mean"),
            iperf_p95_mbps=("iperf_p95_mbps", "mean"),
            target_utilization=("target_utilization", "mean"),
            mean_rtt_s=("mean_rtt_s", "mean"),
            p95_rtt_s=("p95_rtt_s", "mean"),
            mean_queue_delay_s=("mean_queue_delay_s", "mean"),
            p95_queue_delay_s=("p95_queue_delay_s", "mean"),
            mean_fec_ratio=("mean_fec_ratio", "mean"),
            brake_fraction=("brake_fraction", "mean"),
            hold_fraction=("hold_fraction", "mean"),
            accel_fraction=("accel_fraction", "mean"),
            sense_bw_mae_mbps=("sense_bw_mae_mbps", "mean"),
            sense_loss_mae=("sense_loss_mae", "mean"),
            num_experiments=("exp_id", "nunique"),
        )
    )


def summarize_transition(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["method", "scene_id", "transition_index"], as_index=False)
        .agg(
            from_bw_mbps=("from_bw_mbps", "mean"),
            to_bw_mbps=("to_bw_mbps", "mean"),
            from_loss_rate=("from_loss_rate", "mean"),
            to_loss_rate=("to_loss_rate", "mean"),
            transition_type=("transition_type", "first"),
            switch_rel_s=("switch_rel_s", "mean"),
            pre_3s_mean_mbps=("pre_3s_mean_mbps", "mean"),
            post_horizon_mean_mbps=("post_horizon_mean_mbps", "mean"),
            recovery_time_s=("recovery_time_s", "mean"),
            deficit_area_mbps_s=("deficit_area_mbps_s", "mean"),
            num_experiments=("exp_id", "nunique"),
        )
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze dynamic PEPesc system experiment results")
    parser.add_argument("--root", required=True, help="Root dir containing <method>/<scene_id>/ or <method>/<scene_id>/exp_x/")
    parser.add_argument("--outdir", default="dynamic_analysis_outputs")
    parser.add_argument("--scenes", default=None, help="Comma-separated scene ids to analyze")
    parser.add_argument("--transition-horizon-s", type=float, default=8.0)
    parser.add_argument("--recovery-ratio", type=float, default=0.85)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    requested = {item.strip() for item in args.scenes.split(",") if item.strip()} if args.scenes else None

    overall_rows = []
    stage_frames = []
    transition_frames = []
    plot_runs_by_scene = {}

    for method, scene_id, exp_id, run_dir in discover_runs(root):
        if requested is not None and scene_id not in requested:
            continue
        overall, stages, transitions, scenario, iperf, link_state = analyze_run(method, scene_id, exp_id, run_dir, args.transition_horizon_s, args.recovery_ratio)
        overall_rows.append(overall)
        stage_frames.append(stages)
        transition_frames.append(transitions)
        plot_runs_by_scene.setdefault(scene_id, {}).setdefault(method, []).append((run_dir, scenario, iperf, link_state))

    if not overall_rows:
        raise SystemExit(f"No dynamic experiment results found under {root}")

    overall_per_exp_df = pd.DataFrame(overall_rows)
    stage_per_exp_df = pd.concat(stage_frames, ignore_index=True) if stage_frames else pd.DataFrame()
    transition_per_exp_df = pd.concat(transition_frames, ignore_index=True) if transition_frames else pd.DataFrame()

    overall_df = summarize_overall(overall_per_exp_df)
    stage_df = summarize_stage(stage_per_exp_df) if not stage_per_exp_df.empty else pd.DataFrame()
    transition_df = summarize_transition(transition_per_exp_df) if not transition_per_exp_df.empty else pd.DataFrame()

    overall_per_exp_df.to_csv(outdir / "dynamic_per_experiment_overall.csv", index=False)
    stage_per_exp_df.to_csv(outdir / "dynamic_per_experiment_stage.csv", index=False)
    transition_per_exp_df.to_csv(outdir / "dynamic_per_experiment_transition.csv", index=False)

    overall_df.to_csv(outdir / "dynamic_overall_results.csv", index=False)
    stage_df.to_csv(outdir / "dynamic_stage_results.csv", index=False)
    transition_df.to_csv(outdir / "dynamic_transition_results.csv", index=False)

    if not transition_df.empty:
        transition_summary = (
            transition_df.groupby(["method", "scene_id"], as_index=False)
            .agg(
                mean_recovery_time_s=("recovery_time_s", "mean"),
                mean_deficit_area_mbps_s=("deficit_area_mbps_s", "mean"),
                mean_post_horizon_mbps=("post_horizon_mean_mbps", "mean"),
                num_experiments=("num_experiments", "max"),
            )
        )
        transition_summary.to_csv(outdir / "dynamic_transition_summary.csv", index=False)
    else:
        transition_summary = pd.DataFrame()

    for scene_id, runs_by_method in plot_runs_by_scene.items():
        plot_scene_timeseries(scene_id, runs_by_method, outdir / f"scene_{scene_id}_throughput_trace.png")
        plot_scene_control(scene_id, runs_by_method, outdir / f"scene_{scene_id}_control_trace.png")
        plot_stage_metric_bar(scene_id, stage_df, "iperf_mean_mbps", "Receiver throughput (Mbps)", outdir / f"scene_{scene_id}_stage_throughput_bar.png", scale=1.0, value_fmt="{:.2f}")
        plot_stage_metric_bar(scene_id, stage_df, "p95_rtt_s", "p95 RTT (ms)", outdir / f"scene_{scene_id}_stage_p95_rtt_bar.png", scale=1000.0, value_fmt="{:.1f}")
        plot_stage_metric_bar(scene_id, stage_df, "p95_queue_delay_s", "p95 queue delay (ms)", outdir / f"scene_{scene_id}_stage_p95_queue_delay_bar.png", scale=1000.0, value_fmt="{:.1f}")

    plot_overall_metric_bar(overall_df, "iperf_mean_mbps", "Average receiver throughput (Mbps)", outdir / "dynamic_avg_throughput_bar.png")
    plot_overall_metric_bar(overall_df, "p95_rtt_s", "p95 RTT (ms)", outdir / "dynamic_p95_rtt_bar.png", scale=1000.0, value_fmt="{:.1f}")
    plot_overall_metric_bar(overall_df, "p95_queue_delay_s", "p95 queue delay (ms)", outdir / "dynamic_p95_queue_delay_bar.png", scale=1000.0, value_fmt="{:.1f}")
    plot_transition_metric_bar(transition_summary, "mean_recovery_time_s", "Mean recovery time (s)", outdir / "dynamic_recovery_time_bar.png", scale=1.0, value_fmt="{:.2f}")
    plot_transition_metric_bar(transition_summary, "mean_deficit_area_mbps_s", "Mean deficit area (Mbps*s)", outdir / "dynamic_deficit_area_bar.png", scale=1.0, value_fmt="{:.2f}")

    with open(outdir / "dynamic_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "root": str(root),
                "methods": sorted(overall_df["method"].unique().tolist()),
                "scenes": sorted(overall_df["scene_id"].unique().tolist()),
                "transition_horizon_s": args.transition_horizon_s,
                "recovery_ratio": args.recovery_ratio,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Dynamic analysis outputs saved to: {outdir}")


if __name__ == "__main__":
    main()
