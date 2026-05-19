import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analyze_static_system_results import (
    add_bar_labels,
    build_mean_trace,
    collect_rows,
    discover_experiment_dirs,
    parse_case,
)


def plot_per_exp_bar(case_df: pd.DataFrame, metric_col: str, ylabel: str, outpath: Path, scale: float = 1.0, value_fmt: str = "{:.2f}"):
    if metric_col not in case_df.columns or case_df.empty:
        return
    pivot = case_df.pivot_table(index="exp_id", columns="method", values=metric_col, aggfunc="mean")
    if pivot.empty:
        return
    pivot = pivot * scale
    ax = pivot.plot(kind="bar", figsize=(10, 5), width=0.8)
    ax.set_xlabel("Experiment repeat")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    add_bar_labels(ax, value_fmt)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_case_mean_trace(root: Path, case_id: str, methods: list[str], outpath: Path):
    true_bw, _true_loss = parse_case(case_id)
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
        return
    plt.axhline(true_bw, color="black", linestyle="--", linewidth=1.0, alpha=0.6, label="configured bandwidth")
    plt.xlabel("Time (s)")
    plt.ylabel("Receiver throughput (Mbps)")
    plt.title(f"Static case {case_id}: mean receiver throughput trace")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def summarize_case(case_df: pd.DataFrame) -> pd.DataFrame:
    return (
        case_df.groupby(["method", "case_id"], as_index=False)
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


def main():
    parser = argparse.ArgumentParser(description="Analyze repeated experiments for one static case")
    parser.add_argument("--root", required=True, help="Root dir like static_runs containing <method>/<case>/exp_x or <method>/<case>/...")
    parser.add_argument("--case-id", required=True, help="Static case id like bw20_loss0p1")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    case_id = args.case_id
    outdir = Path(args.outdir).resolve() if args.outdir else (root / "_case_analysis" / case_id).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = collect_rows(root)
    case_df = df[df["case_id"] == case_id].copy()
    if case_df.empty:
        raise SystemExit(f"No repeated static results found for case: {case_id}")

    case_df.to_csv(outdir / "case_per_experiment_results.csv", index=False)
    summary = summarize_case(case_df)
    summary.to_csv(outdir / "case_summary.csv", index=False)

    methods = sorted(case_df["method"].unique().tolist())
    plot_case_mean_trace(root, case_id, methods, outdir / f"{case_id}_mean_throughput_trace.png")
    plot_per_exp_bar(case_df, "throughput_mbps", "Receiver throughput (Mbps)", outdir / "throughput_by_experiment_bar.png", scale=1.0, value_fmt="{:.2f}")
    plot_per_exp_bar(case_df, "p95_rtt_s", "p95 RTT (ms)", outdir / "p95_rtt_by_experiment_bar.png", scale=1000.0, value_fmt="{:.1f}")
    plot_per_exp_bar(case_df, "p95_queue_delay_s", "p95 queue delay (ms)", outdir / "p95_queue_delay_by_experiment_bar.png", scale=1000.0, value_fmt="{:.1f}")

    with open(outdir / "case_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "case_id": case_id,
                "methods": methods,
                "num_experiments": int(case_df["exp_id"].nunique()),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Single-case static analysis outputs saved to: {outdir}")


if __name__ == "__main__":
    main()
