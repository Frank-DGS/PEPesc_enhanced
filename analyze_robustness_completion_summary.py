import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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


def load_method_summary(method_dir: Path) -> pd.DataFrame:
    summary_path = method_dir / "trial_summary.csv"
    if not summary_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(summary_path)
    if df.empty:
        return df
    df = df.copy()
    df["method"] = method_dir.name
    if "trial_id" in df.columns:
        df["trial_id"] = pd.to_numeric(df["trial_id"], errors="coerce")
    if "completion_time_s" in df.columns:
        df["completion_time_s"] = pd.to_numeric(df["completion_time_s"], errors="coerce")
    return df


def add_bar_labels(ax, fmt: str = "{:.1f}") -> None:
    for container in ax.containers:
        labels = []
        for bar in container:
            height = bar.get_height()
            labels.append("" if pd.isna(height) else fmt.format(height))
        ax.bar_label(container, labels=labels, padding=2, fontsize=8)


def save_completion_time_by_trial(df: pd.DataFrame, method: str, outdir: Path) -> None:
    sub = df[df["method"] == method].sort_values("trial_id").copy()
    if sub.empty:
        return
    plt.figure(figsize=(12, 4.5))
    ax = plt.gca()
    ax.bar(
        [str(int(x)) if pd.notna(x) else "" for x in sub["trial_id"]],
        sub["completion_time_s"],
        color=method_color(method),
        width=0.22,
    )
    ax.set_xlabel("Trial ID")
    ax.set_ylabel("Completion Time (s)")
    ax.set_title("{} Completion Time by Trial".format(prettify_method(method)))
    ax.grid(True, axis="y", alpha=0.25)
    add_bar_labels(ax, "{:.1f}")
    plt.tight_layout()
    plt.savefig(outdir / "completion_time_by_trial_{}.png".format(method), dpi=200)
    plt.close()


def save_mean_completion_time(df: pd.DataFrame, outdir: Path) -> None:
    summary = (
        df.groupby("method", as_index=False)
        .agg(
            mean_completion_time_s=("completion_time_s", "mean"),
            std_completion_time_s=("completion_time_s", "std"),
            num_trials=("trial_id", "count"),
        )
    )
    if summary.empty:
        return
    plt.figure(figsize=(6.8, 4.5))
    ax = plt.gca()
    ax.bar(
        [prettify_method(m) for m in summary["method"]],
        summary["mean_completion_time_s"],
        color=[method_color(m) for m in summary["method"]],
    )
    ax.set_ylabel("Mean Completion Time (s)")
    ax.set_title("Mean Completion Time Comparison")
    ax.grid(True, axis="y", alpha=0.25)
    add_bar_labels(ax, "{:.1f}")
    plt.tight_layout()
    plt.savefig(outdir / "mean_completion_time_bar.png", dpi=200)
    plt.close()
    summary.to_csv(outdir / "mean_completion_time_summary.csv", index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description="Analyze robustness completion time from per-method trial_summary.csv files")
    parser.add_argument("--root", required=True, help="Root directory containing per-method robustness folders")
    parser.add_argument("--outdir", required=True, help="Output analysis directory")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    frames = []
    for method_dir in sorted(root.iterdir()):
        if not method_dir.is_dir():
            continue
        df = load_method_summary(method_dir)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise SystemExit("No trial_summary.csv found under {}".format(root))

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.dropna(subset=["trial_id", "completion_time_s"])
    all_df = all_df.sort_values(["method", "trial_id"]).reset_index(drop=True)
    all_df.to_csv(outdir / "robustness_completion_trials.csv", index=False, encoding="utf-8-sig")

    for method in sorted(all_df["method"].drop_duplicates()):
        save_completion_time_by_trial(all_df, method, outdir)
    save_mean_completion_time(all_df, outdir)


if __name__ == "__main__":
    main()
