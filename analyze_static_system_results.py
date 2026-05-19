import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CASE_RE = re.compile(r"bw(?P<bw>\d+(?:\.\d+)?)_loss(?P<loss>\d+(?:p\d+)?)", re.IGNORECASE)
IPERF_SUMMARY_RE = re.compile(
    r"\[\s*\d+\]\s+0\.0-\s*(?P<seconds>\d+(?:\.\d+)?)\s+sec\s+.+?\s+(?P<bw>\d+(?:\.\d+)?)\s+(?P<unit>[KMG])bits/sec",
    re.IGNORECASE,
)
IPERF_INTERVAL_RE = re.compile(
    r"\[\s*\d+\]\s+"
    r"(?P<start>\d+(?:\.\d+)?)\s*-\s*(?P<end>\d+(?:\.\d+)?)\s+sec\s+"
    r".+?\s+(?P<bw>\d+(?:\.\d+)?)\s+(?P<unit>[KMG])bits/sec",
    re.IGNORECASE,
)


def parse_case(case_id: str):
    match = CASE_RE.fullmatch(case_id)
    if not match:
        raise ValueError(f"Invalid static case id: {case_id}")
    bw = float(match.group("bw"))
    loss = float(match.group("loss").replace("p", "."))
    return bw, loss


def unit_to_mbps(value: float, unit: str) -> float:
    unit = unit.upper()
    if unit == "G":
        return value * 1000.0
    if unit == "M":
        return value
    if unit == "K":
        return value / 1000.0
    return value / 1_000_000.0


def parse_iperf_summary(log_path: Path):
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = list(IPERF_SUMMARY_RE.finditer(text))
    if not matches:
        raise ValueError(f"Could not parse iperf summary from {log_path}")
    match = matches[-1]
    return {
        "iperf_duration_s": float(match.group("seconds")),
        "throughput_mbps": unit_to_mbps(float(match.group("bw")), match.group("unit")),
    }


def parse_iperf_intervals(log_path: Path) -> pd.DataFrame:
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


def parse_link_state(link_state_path: Path):
    df = pd.read_csv(link_state_path)
    return {
        "link_rows": int(len(df)),
        "mean_rtt_s": float(df["rtt"].mean()) if "rtt" in df.columns else float("nan"),
        "p95_rtt_s": float(df["rtt"].quantile(0.95)) if "rtt" in df.columns else float("nan"),
        "mean_queue_delay_s": float(df["queue_delay"].mean()) if "queue_delay" in df.columns else float("nan"),
        "p95_queue_delay_s": float(df["queue_delay"].quantile(0.95)) if "queue_delay" in df.columns else float("nan"),
    }


def add_bar_labels(ax, fmt: str = "{:.2f}") -> None:
    for container in ax.containers:
        labels = []
        for bar in container:
            height = bar.get_height()
            labels.append("" if pd.isna(height) else fmt.format(height))
        ax.bar_label(container, labels=labels, padding=2, fontsize=8)


def discover_experiment_dirs(case_dir: Path):
    exp_dirs = sorted(
        [
            path
            for path in case_dir.iterdir()
            if path.is_dir() and path.name.lower().startswith("exp_")
        ]
    )
    return exp_dirs if exp_dirs else [case_dir]


def collect_rows(root: Path):
    rows = []
    for method_dir in sorted(root.iterdir()):
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        for case_dir in sorted(method_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            case_id = case_dir.name
            try:
                true_bw, true_loss = parse_case(case_id)
            except ValueError:
                continue
            for exp_dir in discover_experiment_dirs(case_dir):
                iperf_path = exp_dir / "iperf_d.log"
                link_state_path = exp_dir / "link_state.csv"
                if not iperf_path.exists():
                    continue
                row = {
                    "method": method,
                    "case_id": case_id,
                    "exp_id": exp_dir.name if exp_dir != case_dir else "exp_1",
                    "true_bw_mbps": true_bw,
                    "true_loss_pct": true_loss,
                    "exp_dir": str(exp_dir),
                }
                row.update(parse_iperf_summary(iperf_path))
                if link_state_path.exists():
                    row.update(parse_link_state(link_state_path))
                rows.append(row)
    if not rows:
        raise SystemExit(f"No static results found under {root}")
    return pd.DataFrame(rows)


def summarize_rows(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["method", "case_id", "true_bw_mbps", "true_loss_pct"], as_index=False)
        .agg(
            throughput_mbps=("throughput_mbps", "mean"),
            throughput_std_mbps=("throughput_mbps", "std"),
            iperf_duration_s=("iperf_duration_s", "mean"),
            mean_rtt_s=("mean_rtt_s", "mean"),
            p95_rtt_s=("p95_rtt_s", "mean"),
            mean_queue_delay_s=("mean_queue_delay_s", "mean"),
            p95_queue_delay_s=("p95_queue_delay_s", "mean"),
            num_experiments=("exp_id", "nunique"),
        )
    )


def plot_sweep(df: pd.DataFrame, x_col: str, x_label: str, fixed_filter, outpath: Path):
    filtered = df.loc[fixed_filter].copy()
    if filtered.empty:
        return
    pivot = (
        filtered.groupby(["method", x_col], as_index=False)["throughput_mbps"]
        .mean()
        .sort_values([x_col, "method"])
    )
    plt.figure(figsize=(8, 5))
    for method, group in pivot.groupby("method"):
        plt.plot(group[x_col], group["throughput_mbps"], marker="o", label=method)
    plt.xlabel(x_label)
    plt.ylabel("Average throughput (Mbps)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_metric_bar(summary: pd.DataFrame, metric_col: str, ylabel: str, outpath: Path, scale: float = 1.0, value_fmt: str = "{:.2f}"):
    if metric_col not in summary.columns:
        return
    pivot = summary.pivot_table(index="case_id", columns="method", values=metric_col, aggfunc="mean")
    if pivot.empty:
        return
    pivot = pivot * scale
    ax = pivot.plot(kind="bar", figsize=(10, 5), width=0.78)
    ax.set_xlabel("Static case")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    add_bar_labels(ax, value_fmt)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def build_mean_trace(exp_dirs: list[Path]) -> pd.DataFrame:
    traces = []
    for exp_dir in exp_dirs:
        log_path = exp_dir / "iperf_d.log"
        iperf = parse_iperf_intervals(log_path)
        if iperf.empty:
            continue
        traces.append(iperf[["mid_s", "throughput_mbps"]].copy())
    if not traces:
        return pd.DataFrame(columns=["mid_s", "throughput_mbps"])
    merged = pd.concat(traces, ignore_index=True)
    return merged.groupby("mid_s", as_index=False)["throughput_mbps"].mean().sort_values("mid_s")


def plot_case_traces(root: Path, summary: pd.DataFrame, outdir: Path):
    trace_dir = outdir / "static_case_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    cases = sorted(summary["case_id"].unique().tolist())
    methods = sorted(summary["method"].unique().tolist())
    for case_id in cases:
        true_bw = float(summary.loc[summary["case_id"] == case_id, "true_bw_mbps"].iloc[0])
        plt.figure(figsize=(10, 4.8))
        plotted = False
        for method in methods:
            exp_dirs = discover_experiment_dirs(root / method / case_id)
            mean_trace = build_mean_trace(exp_dirs)
            if mean_trace.empty:
                continue
            plt.plot(mean_trace["mid_s"], mean_trace["throughput_mbps"], linewidth=1.6, label=method)
            plotted = True
        if not plotted:
            plt.close()
            continue
        plt.axhline(true_bw, color="black", linestyle="--", linewidth=1.0, alpha=0.6, label="configured bandwidth")
        plt.xlabel("Time (s)")
        plt.ylabel("Receiver throughput (Mbps)")
        plt.title(f"Static case {case_id}: mean receiver throughput trace")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(trace_dir / f"{case_id}_throughput_trace.png", dpi=180)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze repeated static system experiment results")
    parser.add_argument("--root", required=True, help="Root dir like static_runs containing <method>/<case>/exp_x or <method>/<case>/...")
    parser.add_argument("--outdir", default="static_analysis_outputs")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = collect_rows(root)
    df.to_csv(outdir / "static_per_experiment_results.csv", index=False)

    summary = summarize_rows(df)
    summary.to_csv(outdir / "static_summary.csv", index=False)

    plot_sweep(
        summary,
        x_col="true_bw_mbps",
        x_label="Static bandwidth (Mbps)",
        fixed_filter=summary["true_loss_pct"].round(3) == 0.1,
        outpath=outdir / "static_bw_sweep_throughput.png",
    )
    plot_sweep(
        summary,
        x_col="true_loss_pct",
        x_label="Static loss (%)",
        fixed_filter=summary["true_bw_mbps"].round(3) == 20.0,
        outpath=outdir / "static_loss_sweep_throughput.png",
    )
    plot_metric_bar(summary, "throughput_mbps", "Receiver throughput (Mbps)", outdir / "static_throughput_by_case_bar.png", scale=1.0, value_fmt="{:.2f}")
    plot_metric_bar(summary, "p95_rtt_s", "p95 RTT (ms)", outdir / "static_p95_rtt_bar.png", scale=1000.0, value_fmt="{:.1f}")
    plot_metric_bar(summary, "p95_queue_delay_s", "p95 queue delay (ms)", outdir / "static_p95_queue_delay_bar.png", scale=1000.0, value_fmt="{:.1f}")
    plot_case_traces(root, summary, outdir)

    with open(outdir / "static_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_experiments": int(len(df)),
                "methods": sorted(df["method"].unique().tolist()),
                "cases": sorted(df["case_id"].unique().tolist()),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Static analysis outputs saved to: {outdir}")


if __name__ == "__main__":
    main()
