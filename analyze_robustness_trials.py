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

METHOD_LABELS = {
    "original": "Original PEPesc",
    "hybrid_adaptive_v2": "Hybrid Adaptive",
    "hybrid_adaptive": "Hybrid Adaptive",
}

METHOD_COLORS = {
    "original": "#F28E2B",
    "hybrid_adaptive_v2": "#4E79A7",
    "hybrid_adaptive": "#4E79A7",
}


def prettify_method(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def method_color(method: str) -> str:
    return METHOD_COLORS.get(method, "#4E79A7")


def load_meta(run_dir: Path):
    path = run_dir / "meta.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def load_trial_result(trial_dir: Path):
    path = trial_dir / "trial_result.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def load_scenario(run_dir: Path):
    path = run_dir / "scenario.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_link_state(run_dir: Path):
    path = run_dir / "link_state.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def parse_iperf(log_path: Path):
    rows = []
    if not log_path.exists():
        return pd.DataFrame(columns=["start_s", "end_s", "throughput_mbps"])
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = IPERF_INTERVAL_RE.search(line)
        if not match:
            continue
        start_s = float(match.group("start"))
        end_s = float(match.group("end"))
        value = float(match.group("bw"))
        unit = match.group("unit").upper()
        if unit == "G":
            mbps = value * 1000.0
        elif unit == "M":
            mbps = value
        else:
            mbps = value / 1000.0
        rows.append({"start_s": start_s, "end_s": end_s, "throughput_mbps": mbps})
    return pd.DataFrame(rows)


def discover_runs(root: Path):
    for method_dir in sorted(root.iterdir()):
        if not method_dir.is_dir():
            continue
        summary_path = method_dir / "trial_summary.csv"
        summary_df = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
        for trial_dir in sorted(method_dir.glob("trial_*")):
            if not trial_dir.is_dir():
                continue
            if (trial_dir / "meta.json").exists() or (trial_dir / "iperf_d.log").exists() or (trial_dir / "trial_result.json").exists():
                run_dir = trial_dir
            else:
                run_dirs = [child for child in trial_dir.iterdir() if child.is_dir() and ((child / "meta.json").exists() or (child / "iperf_d.log").exists())]
                if not run_dirs:
                    continue
                run_dir = run_dirs[0]
            trial_result = load_trial_result(trial_dir)
            summary_row = None
            if not summary_df.empty and "trial_id" in summary_df.columns:
                try:
                    trial_id = int(trial_dir.name.split("_")[-1])
                    matched = summary_df.loc[summary_df["trial_id"] == trial_id]
                    if not matched.empty:
                        summary_row = matched.iloc[0].to_dict()
                except Exception:
                    summary_row = None
            yield method_dir.name, trial_dir, run_dir, summary_row or trial_result


def analyze_run(method: str, trial_dir: Path, run_dir: Path, trial_info: dict):
    meta = load_meta(run_dir)
    scenario = load_scenario(run_dir)
    iperf = parse_iperf(run_dir / "iperf_d.log")
    link = load_link_state(run_dir)

    planned_time_s = float(trial_info.get("planned_time_s") or 0.0)
    completion_time_s = float(trial_info.get("completion_time_s") or (float(iperf["end_s"].max()) if not iperf.empty else 0.0))
    zero_throughput_duration_s = float((iperf["throughput_mbps"] <= 0.05).sum()) if not iperf.empty else 0.0
    row = {
        "method": method,
        "trial_id": int(str(trial_dir.name).split("_")[-1]),
        "scene_name": meta.get("scene_name") or trial_info.get("scene_name") or run_dir.name,
        "status": trial_info.get("status") or ("completed" if completion_time_s >= max(planned_time_s - 2.0, 0.0) else "aborted"),
        "abort_reason": trial_info.get("abort_reason") or "",
        "planned_time_s": planned_time_s,
        "completion_time_s": completion_time_s,
        "completion_ratio": float(trial_info.get("completion_ratio") or ((completion_time_s / planned_time_s) if planned_time_s > 0 else 0.0)),
        "zero_throughput_duration_s": zero_throughput_duration_s,
        "max_rtt_s": float(link["rtt"].max()) if "rtt" in link.columns and not link.empty else float("nan"),
        "max_queue_delay_s": float(link["queue_delay"].max()) if "queue_delay" in link.columns and not link.empty else float("nan"),
        "max_ack_gap_ms": float(link["ack_gap_ms"].max()) if "ack_gap_ms" in link.columns and not link.empty else float("nan"),
        "max_streamc_queue_size": float(link["streamc_queue_size"].max()) if "streamc_queue_size" in link.columns and not link.empty else float("nan"),
    }
    if not scenario.empty and "stage_duration_s" in scenario.columns:
        row["scenario_stage_count"] = int(len(scenario))
        row["scenario_duration_s"] = float(scenario["stage_duration_s"].astype(float).sum())
    return row


def add_bar_labels(ax, fmt: str = "{:.1f}"):
    for container in ax.containers:
        labels = []
        for bar in container:
            height = bar.get_height()
            labels.append("" if pd.isna(height) else fmt.format(height))
        ax.bar_label(container, labels=labels, padding=2, fontsize=8)


def save_bar_completion(df: pd.DataFrame, outdir: Path):
    if df.empty:
        return
    for method in sorted(df["method"].drop_duplicates()):
        sub = df[df["method"] == method].sort_values("trial_id")
        plt.figure(figsize=(12, 4.5))
        ax = plt.gca()
        ax.bar([str(t) for t in sub["trial_id"]], sub["completion_time_s"], color=method_color(method), width=0.22)
        ax.set_xlabel("Trial ID")
        ax.set_ylabel("Completion Time (s)")
        ax.set_title("{} Completion Time by Trial".format(prettify_method(method)))
        ax.grid(True, axis="y", alpha=0.25)
        add_bar_labels(ax, "{:.1f}")
        plt.tight_layout()
        plt.savefig(outdir / "completion_time_by_trial_{}.png".format(method), dpi=200)
        plt.close()


def save_completion_rate(df: pd.DataFrame, outdir: Path):
    if df.empty:
        return
    summary = df.assign(completed_flag=(df["status"] == "completed").astype(float)).groupby("method", as_index=False).agg(completion_rate=("completed_flag", "mean"))
    plt.figure(figsize=(6.5, 4.5))
    ax = plt.gca()
    ax.bar([prettify_method(m) for m in summary["method"]], summary["completion_rate"], color=[method_color(m) for m in summary["method"]])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Completion Rate")
    ax.set_title("Robustness Trial Completion Rate")
    ax.grid(True, axis="y", alpha=0.25)
    add_bar_labels(ax, "{:.2f}")
    plt.tight_layout()
    plt.savefig(outdir / "completion_rate_bar.png", dpi=200)
    plt.close()


def save_queue_and_ack(df: pd.DataFrame, outdir: Path):
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    if df["max_streamc_queue_size"].notna().any():
        grouped = df.groupby("method", as_index=False)["max_streamc_queue_size"].mean()
        axes[0].bar([prettify_method(m) for m in grouped["method"]], grouped["max_streamc_queue_size"], color=[method_color(m) for m in grouped["method"]])
        axes[0].set_title("Mean Peak Stream Queue Size")
        axes[0].set_ylabel("Packets")
        axes[0].grid(True, axis="y", alpha=0.25)
        add_bar_labels(axes[0], "{:.1f}")
    else:
        axes[0].set_visible(False)
    if df["max_ack_gap_ms"].notna().any():
        grouped = df.groupby("method", as_index=False)["max_ack_gap_ms"].mean()
        axes[1].bar([prettify_method(m) for m in grouped["method"]], grouped["max_ack_gap_ms"], color=[method_color(m) for m in grouped["method"]])
        axes[1].set_title("Mean Peak ACK Gap")
        axes[1].set_ylabel("ms")
        axes[1].grid(True, axis="y", alpha=0.25)
        add_bar_labels(axes[1], "{:.1f}")
    else:
        axes[1].set_visible(False)
    plt.tight_layout()
    plt.savefig(outdir / "queue_ack_summary_bar.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze robustness trials produced by run_robustness_trials.py")
    parser.add_argument("--root", required=True, help="Root directory containing per-method robustness outputs")
    parser.add_argument("--outdir", required=True, help="Output analysis directory")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for method, trial_dir, run_dir, trial_info in discover_runs(root):
        rows.append(analyze_run(method, trial_dir, run_dir, trial_info))
    if not rows:
        raise SystemExit("No robustness trials discovered under {}".format(root))

    df = pd.DataFrame(rows).sort_values(["method", "trial_id"]).reset_index(drop=True)
    df.to_csv(outdir / "robustness_trial_results.csv", index=False, encoding="utf-8-sig")

    summary = df.assign(completed_flag=(df["status"] == "completed").astype(float)).groupby("method", as_index=False).agg(
        num_trials=("trial_id", "count"),
        completion_rate=("completed_flag", "mean"),
        mean_completion_time_s=("completion_time_s", "mean"),
        mean_completion_ratio=("completion_ratio", "mean"),
        mean_zero_throughput_duration_s=("zero_throughput_duration_s", "mean"),
        mean_peak_ack_gap_ms=("max_ack_gap_ms", "mean"),
        mean_peak_streamc_queue_size=("max_streamc_queue_size", "mean"),
    )
    summary.to_csv(outdir / "robustness_method_summary.csv", index=False, encoding="utf-8-sig")

    save_bar_completion(df, outdir)
    save_completion_rate(df, outdir)
    save_queue_and_ack(df, outdir)


if __name__ == "__main__":
    main()
