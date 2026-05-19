import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adaptive_data import (
    add_loss_score_columns,
    build_training_keep_frame,
    resample_aligned_training_frame,
)
from adaptive_schema import DEFAULT_RESAMPLE_MS


def drop_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required_numeric_cols = [
        "ts",
        "true_bw_mbps",
        "est_bw_mbps",
        "est_bw_max_mbps",
        "probe_bw_mbps",
        "pred_bw_mbps",
        "true_loss_rate",
        "loss_rate",
        "pred_loss_rate",
    ]

    for col in required_numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    mask = np.ones(len(out), dtype=bool)
    for col in required_numeric_cols:
        if col in out.columns:
            mask &= np.isfinite(out[col].to_numpy(dtype=float))

    return out.loc[mask].reset_index(drop=True)


def load_link_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, engine="python", on_bad_lines="skip")


def sanitize_ts(df: pd.DataFrame, col: str = "ts") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=[col])
    out = out[(out[col] > 1e8) & (out[col] < 1e11)]
    return out.sort_values(col).reset_index(drop=True)


def align_with_scenario(link: pd.DataFrame, scenario: pd.DataFrame) -> pd.DataFrame:
    link = sanitize_ts(link, "ts")
    scenario = sanitize_ts(scenario, "ts")
    scenario = scenario.dropna(subset=["true_bw", "true_loss_rate"]).reset_index(drop=True)

    if len(scenario) == 0:
        return link.iloc[0:0].copy()

    merged = pd.merge_asof(
        link.sort_values("ts"),
        scenario.sort_values("ts"),
        on="ts",
        direction="backward",
    )
    merged = merged.dropna(subset=["true_bw", "true_loss_rate"]).copy()

    labeled_end_ts = None
    if "stage_duration_s" in scenario.columns:
        durations = pd.to_numeric(scenario["stage_duration_s"], errors="coerce").to_numpy(dtype=float)
        valid = durations[np.isfinite(durations) & (durations > 0)]
        if len(valid) == len(scenario):
            labeled_end_ts = float(scenario["ts"].iloc[-1] + durations[-1])

    if labeled_end_ts is None and "end_ts" in scenario.columns:
        end_ts = pd.to_numeric(scenario["end_ts"], errors="coerce").to_numpy(dtype=float)
        valid = end_ts[np.isfinite(end_ts)]
        if len(valid) == len(scenario):
            labeled_end_ts = float(end_ts[-1])

    if labeled_end_ts is None:
        scenario_ts = scenario["ts"].to_numpy(dtype=float)
        if len(scenario_ts) >= 2:
            stage_durations = np.diff(scenario_ts)
            stage_durations = stage_durations[np.isfinite(stage_durations) & (stage_durations > 0)]
            inferred_stage_duration = float(np.median(stage_durations)) if len(stage_durations) else 0.0
        else:
            inferred_stage_duration = 0.0
        if inferred_stage_duration > 0:
            labeled_end_ts = float(scenario_ts[-1] + inferred_stage_duration)

    if labeled_end_ts is not None:
        merged = merged[merged["ts"] < labeled_end_ts].copy()

    merged["t_rel"] = merged["ts"] - merged["ts"].iloc[0]
    return merged


def add_unit_conversion(df: pd.DataFrame, sc_packet_size_bytes: int) -> pd.DataFrame:
    out = df.copy()
    for col in ["est_bw", "est_bw_max", "probe_bw", "pred_bw"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[f"{col}_mbps"] = out[col] * sc_packet_size_bytes * 8.0 / 1e6
    out["true_bw_mbps"] = pd.to_numeric(out["true_bw"], errors="coerce")

    for col in ["loss_rate", "pred_loss_rate", "true_loss_rate"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def compute_metrics(df: pd.DataFrame) -> dict:
    out = {}

    for col in ["est_bw", "est_bw_max", "probe_bw", "pred_bw"]:
        valid = df[["true_bw_mbps", f"{col}_mbps"]].apply(pd.to_numeric, errors="coerce")
        valid = valid.replace([np.inf, -np.inf], np.nan).dropna()
        true_bw = valid["true_bw_mbps"].to_numpy(dtype=float)
        pred = valid[f"{col}_mbps"].to_numpy(dtype=float)
        if len(valid) == 0:
            out[col] = {"type": "bandwidth", "RMSE": float("nan"), "MAE": float("nan"), "MAPE": float("nan")}
            continue
        out[col] = {
            "type": "bandwidth",
            "RMSE": float(np.sqrt(np.mean((pred - true_bw) ** 2))),
            "MAE": float(np.mean(np.abs(pred - true_bw))),
            "MAPE": float(np.mean(np.abs((pred - true_bw) / np.maximum(true_bw, 1e-9)))),
        }

    for col in ["loss_rate", "pred_loss_rate"]:
        valid = df[["true_loss_rate", col]].apply(pd.to_numeric, errors="coerce")
        valid = valid.replace([np.inf, -np.inf], np.nan).dropna()
        true_loss = valid["true_loss_rate"].to_numpy(dtype=float)
        pred = valid[col].to_numpy(dtype=float)
        if len(valid) == 0:
            out[col] = {"type": "loss_rate", "RMSE": float("nan"), "MAE": float("nan"), "MAPE": float("nan")}
            continue
        out[col] = {
            "type": "loss_rate",
            "RMSE": float(np.sqrt(np.mean((pred - true_loss) ** 2))),
            "MAE": float(np.mean(np.abs(pred - true_loss))),
            "MAPE": float(np.mean(np.abs((pred - true_loss) / np.maximum(true_loss, 1e-9)))),
        }

    return out


def compute_bw_segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("true_bw_mbps")
        .agg(
            n=("true_bw_mbps", "size"),
            est_mean=("est_bw_mbps", "mean"),
            estmax_mean=("est_bw_max_mbps", "mean"),
            probe_mean=("probe_bw_mbps", "mean"),
            pred_mean=("pred_bw_mbps", "mean"),
        )
        .reset_index()
        .sort_values("true_bw_mbps")
    )


def compute_loss_segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("true_loss_rate")
        .agg(
            n=("true_loss_rate", "size"),
            loss_mean=("loss_rate", "mean"),
            pred_loss_mean=("pred_loss_rate", "mean"),
        )
        .reset_index()
        .sort_values("true_loss_rate")
    )


def make_plots(
    df: pd.DataFrame,
    scenario: pd.DataFrame,
    metrics: dict,
    bw_seg: pd.DataFrame,
    loss_seg: pd.DataFrame,
    outdir: Path,
) -> None:
    plot_df = df.copy()
    for col in ["est_bw_mbps", "est_bw_max_mbps", "probe_bw_mbps", "pred_bw_mbps"]:
        plot_df[f"{col}_smooth"] = plot_df[col].rolling(window=5, min_periods=1).mean()
    for col in ["loss_rate", "pred_loss_rate"]:
        plot_df[f"{col}_smooth"] = plot_df[col].rolling(window=5, min_periods=1).mean()

    scenario = scenario.sort_values("ts")
    t0 = scenario["ts"].iloc[0]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(plot_df["t_rel"], plot_df["true_bw_mbps"], label="true_bw")
    ax.plot(plot_df["t_rel"], plot_df["est_bw_mbps_smooth"], label="est_bw")
    ax.plot(plot_df["t_rel"], plot_df["probe_bw_mbps_smooth"], label="probe_bw")
    ax.plot(plot_df["t_rel"], plot_df["pred_bw_mbps_smooth"], label="pred_bw")
    for ts in scenario["ts"]:
        ax.axvline(ts - t0, color="gray", linestyle="--", alpha=0.25)
    ax.set_xlabel("Time since first scenario label (s)")
    ax.set_ylabel("Bandwidth (Mbps)")
    ax.set_title("Dynamic bandwidth tracking")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "bw_timeseries.png", dpi=180)
    plt.close(fig)

    bw_metric_df = pd.DataFrame({k: v for k, v in metrics.items() if v["type"] == "bandwidth"}).T
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    bw_metric_df["RMSE"].plot(kind="bar", ax=axs[0], title="Bandwidth RMSE (Mbps)")
    bw_metric_df["MAE"].plot(kind="bar", ax=axs[1], title="Bandwidth MAE (Mbps)")
    bw_metric_df["MAPE"].plot(kind="bar", ax=axs[2], title="Bandwidth MAPE")
    for ax in axs:
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(outdir / "bw_metric_bars.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = bw_seg["true_bw_mbps"].to_numpy(dtype=float)
    ax.plot(x, x, linestyle="--", label="ideal")
    ax.plot(x, bw_seg["est_mean"], marker="o", label="est_mean")
    ax.plot(x, bw_seg["estmax_mean"], marker="o", label="est_bw_max_mean")
    ax.plot(x, bw_seg["probe_mean"], marker="o", label="probe_mean")
    ax.plot(x, bw_seg["pred_mean"], marker="o", label="pred_mean")
    ax.set_xlabel("True bandwidth (Mbps)")
    ax.set_ylabel("Estimated mean bandwidth (Mbps)")
    ax.set_title("Per-stage mean bandwidth estimates")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "bw_segment_means.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(plot_df["t_rel"], plot_df["true_loss_rate"], label="true_loss_rate")
    ax.plot(plot_df["t_rel"], plot_df["loss_rate_smooth"], label="loss_rate")
    ax.plot(plot_df["t_rel"], plot_df["pred_loss_rate_smooth"], label="pred_loss_rate")
    for ts in scenario["ts"]:
        ax.axvline(ts - t0, color="gray", linestyle="--", alpha=0.25)
    ax.set_xlabel("Time since first scenario label (s)")
    ax.set_ylabel("Loss rate")
    ax.set_title("Dynamic loss-rate tracking")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "loss_timeseries.png", dpi=180)
    plt.close(fig)

    loss_metric_df = pd.DataFrame({k: v for k, v in metrics.items() if v["type"] == "loss_rate"}).T
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    loss_metric_df["RMSE"].plot(kind="bar", ax=axs[0], title="Loss-rate RMSE")
    loss_metric_df["MAE"].plot(kind="bar", ax=axs[1], title="Loss-rate MAE")
    loss_metric_df["MAPE"].plot(kind="bar", ax=axs[2], title="Loss-rate MAPE")
    for ax in axs:
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(outdir / "loss_metric_bars.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = loss_seg["true_loss_rate"].to_numpy(dtype=float)
    ax.plot(x, x, linestyle="--", label="ideal")
    ax.plot(x, loss_seg["loss_mean"], marker="o", label="loss_rate_mean")
    ax.plot(x, loss_seg["pred_loss_mean"], marker="o", label="pred_loss_rate_mean")
    ax.set_xlabel("True loss rate")
    ax.set_ylabel("Estimated mean loss rate")
    ax.set_title("Per-stage mean loss-rate estimates")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "loss_segment_means.png", dpi=180)
    plt.close(fig)


def build_training_outputs(df: pd.DataFrame, interval_ms: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    enriched = add_loss_score_columns(df)
    raw_frame = build_training_keep_frame(enriched)
    resampled_frame = build_training_keep_frame(resample_aligned_training_frame(enriched, interval_ms=interval_ms))
    return raw_frame, resampled_frame


def run_analysis(
    link_path: Path,
    scenario_path: Path,
    outdir: Path,
    sc_packet_size: int,
    resample_ms: int,
) -> Dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)

    link = load_link_csv(link_path)
    scenario = pd.read_csv(scenario_path)
    merged = align_with_scenario(link, scenario)
    merged = add_unit_conversion(merged, sc_packet_size)
    before_rows = len(merged)
    merged = drop_invalid_rows(merged)
    after_rows = len(merged)

    metrics = compute_metrics(merged)
    bw_seg = compute_bw_segment_summary(merged)
    loss_seg = compute_loss_segment_summary(merged)
    make_plots(merged, scenario, metrics, bw_seg, loss_seg, outdir)

    raw_train_df, resampled_train_df = build_training_outputs(merged, interval_ms=resample_ms)

    metrics_path = outdir / "metrics_summary.csv"
    raw_csv_path = outdir / "aligned_link_state_raw.csv"
    resampled_csv_path = outdir / f"aligned_link_state_{resample_ms}ms.csv"
    default_csv_path = outdir / "aligned_link_state.csv"

    pd.DataFrame(metrics).T.to_csv(metrics_path)
    raw_train_df.to_csv(raw_csv_path, index=False)
    resampled_train_df.to_csv(resampled_csv_path, index=False)
    resampled_train_df.to_csv(default_csv_path, index=False)

    return {
        "before_rows": before_rows,
        "after_rows": after_rows,
        "dropped_rows": before_rows - after_rows,
        "metrics_path": metrics_path,
        "raw_csv_path": raw_csv_path,
        "resampled_csv_path": resampled_csv_path,
        "default_csv_path": default_csv_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--link", default="link_state.csv")
    parser.add_argument("--scenario", default="scenario.csv")
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory. Defaults to the directory containing --link.",
    )
    parser.add_argument(
        "--sc-packet-size",
        type=int,
        default=1465,
        help="ScPacketSize in bytes; if your protocol.py differs, change this value",
    )
    parser.add_argument(
        "--resample-ms",
        type=int,
        default=DEFAULT_RESAMPLE_MS,
        help="固定重采样周期，默认 50ms",
    )
    args = parser.parse_args()

    link_path = Path(args.link)
    scenario_path = Path(args.scenario)
    outdir = Path(args.outdir) if args.outdir else link_path.resolve().parent
    result = run_analysis(
        link_path=link_path,
        scenario_path=scenario_path,
        outdir=outdir,
        sc_packet_size=args.sc_packet_size,
        resample_ms=args.resample_ms,
    )

    print("Analysis done.")
    print(
        f"Rows kept for analysis: {result['after_rows']}/{result['before_rows']} "
        f"(dropped {result['dropped_rows']} invalid rows)"
    )
    print(f"Raw training csv: {Path(result['raw_csv_path']).resolve()}")
    print(f"Resampled training csv: {Path(result['resampled_csv_path']).resolve()}")
    print(f"Backward-compatible default csv: {Path(result['default_csv_path']).resolve()}")


if __name__ == "__main__":
    main()
