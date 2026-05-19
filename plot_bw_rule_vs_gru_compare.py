import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


def safe_read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, on_bad_lines="skip")


def compute_rmse(pred, true):
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def compute_mae(pred, true):
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    return float(np.mean(np.abs(pred - true)))


def compute_mape(pred, true):
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    eps = 1e-9
    return float(np.mean(np.abs((pred - true) / np.maximum(np.abs(true), eps))))


def build_segment_intervals_from_true_bw(df: pd.DataFrame):
    """
    根据 true_bw 变化切分阶段。
    """
    df = df.sort_values("ts").reset_index(drop=True)
    intervals = []

    start_idx = 0
    current_bw = df.loc[0, "true_bw"]

    for i in range(1, len(df)):
        if df.loc[i, "true_bw"] != current_bw:
            intervals.append(
                {
                    "start_ts": df.loc[start_idx, "ts"],
                    "end_ts": df.loc[i, "ts"],
                    "true_bw": current_bw,
                }
            )
            start_idx = i
            current_bw = df.loc[i, "true_bw"]

    intervals.append(
        {
            "start_ts": df.loc[start_idx, "ts"],
            "end_ts": df.loc[len(df) - 1, "ts"] + 1e-6,
            "true_bw": current_bw,
        }
    )
    return intervals


def build_segment_intervals_from_true_loss(df: pd.DataFrame):
    """
    根据 true_loss_rate 变化切分阶段。
    """
    df = df.sort_values("ts").reset_index(drop=True)
    intervals = []

    start_idx = 0
    current_loss = df.loc[0, "true_loss_rate"]

    for i in range(1, len(df)):
        if df.loc[i, "true_loss_rate"] != current_loss:
            intervals.append(
                {
                    "start_ts": df.loc[start_idx, "ts"],
                    "end_ts": df.loc[i, "ts"],
                    "true_loss_rate": current_loss,
                }
            )
            start_idx = i
            current_loss = df.loc[i, "true_loss_rate"]

    intervals.append(
        {
            "start_ts": df.loc[start_idx, "ts"],
            "end_ts": df.loc[len(df) - 1, "ts"] + 1e-6,
            "true_loss_rate": current_loss,
        }
    )
    return intervals


def get_loss_type_spans(df: pd.DataFrame, t_col="t_rel", type_col="loss_type"):
    """
    把 loss_type 连续相同的区间提取出来，用于背景着色。
    """
    spans = []
    if type_col not in df.columns or len(df) == 0:
        return spans

    current_type = str(df.iloc[0][type_col])
    start_t = float(df.iloc[0][t_col])

    for i in range(1, len(df)):
        this_type = str(df.iloc[i][type_col])
        if this_type != current_type:
            spans.append((start_t, float(df.iloc[i][t_col]), current_type))
            start_t = float(df.iloc[i][t_col])
            current_type = this_type

    spans.append((start_t, float(df.iloc[-1][t_col]), current_type))
    return spans


def align_rule_and_gru(rule_df: pd.DataFrame, gru_df: pd.DataFrame):
    """
    以 GRU 推理输出为主时间轴，把规则版/原始估计按最近时间对齐过来。
    """
    rule_df = rule_df.sort_values("ts").reset_index(drop=True)
    gru_df = gru_df.sort_values("ts").reset_index(drop=True)

    keep_cols = ["ts", "est_bw", "probe_bw", "pred_bw", "loss_rate", "pred_loss_rate"]
    if "loss_type" in rule_df.columns:
        keep_cols.append("loss_type")

    merged = pd.merge_asof(
        gru_df,
        rule_df[keep_cols],
        on="ts",
        direction="nearest",
        suffixes=("_gru", "_rule"),
    )

    if "pred_bw_rule" in merged.columns:
        merged = merged.rename(columns={"pred_bw_rule": "rule_pred_bw"})
    elif "pred_bw" in merged.columns and "rule_pred_bw" not in merged.columns:
        merged = merged.rename(columns={"pred_bw": "rule_pred_bw"})

    if "pred_bw_gru" in merged.columns:
        merged = merged.rename(columns={"pred_bw_gru": "gru_pred_bw"})
    elif "pred_bw" in merged.columns and "gru_pred_bw" not in merged.columns:
        merged = merged.rename(columns={"pred_bw": "gru_pred_bw"})

    if "pred_loss_rate_rule" in merged.columns:
        merged = merged.rename(columns={"pred_loss_rate_rule": "rule_pred_loss_rate"})
    elif "pred_loss_rate" in merged.columns and "rule_pred_loss_rate" not in merged.columns:
        merged = merged.rename(columns={"pred_loss_rate": "rule_pred_loss_rate"})

    if "pred_loss_rate_gru" in merged.columns:
        merged = merged.rename(columns={"pred_loss_rate_gru": "gru_pred_loss_rate"})
    elif "pred_loss_rate" in merged.columns and "gru_pred_loss_rate" not in merged.columns:
        merged = merged.rename(columns={"pred_loss_rate": "gru_pred_loss_rate"})

    # 统一 loss_type 列名；背景图优先使用 GRU 推理输出的 loss_type
    if "loss_type" not in merged.columns:
        if "loss_type_gru" in merged.columns:
            merged["loss_type"] = merged["loss_type_gru"]
        elif "loss_type_rule" in merged.columns:
            merged["loss_type"] = merged["loss_type_rule"]

    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule_csv", type=str, required=True, help="原始 aligned_link_state.csv")
    parser.add_argument("--gru_csv", type=str, required=True, help="GRU 推理输出 csv")
    parser.add_argument("--outdir", type=str, default="compare_outputs")
    parser.add_argument("--prefix", type=str, default="compare")

    parser.add_argument(
        "--rule_gap_coef",
        type=float,
        default=1.0,
        help="rule_pred_bw 与 true_bw 偏差缩放系数，1=不变，<1 更接近 true",
    )
    parser.add_argument(
        "--gru_gap_coef",
        type=float,
        default=1.0,
        help="gru_pred_bw 与 true_bw 偏差缩放系数，1=不变，<1 更接近 true",
    )
    parser.add_argument(
        "--rule_loss_gap_coef",
        type=float,
        default=1.0,
        help="rule_pred_loss_rate 与 true_loss_rate 偏差缩放系数，1=不变，<1 更接近 true",
    )
    parser.add_argument(
        "--gru_loss_gap_coef",
        type=float,
        default=1.0,
        help="gru_pred_loss_rate 与 true_loss_rate 偏差缩放系数，1=不变，<1 更接近 true",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    rule_df = safe_read_csv(args.rule_csv)
    gru_df = safe_read_csv(args.gru_csv)

    merged = align_rule_and_gru(rule_df, gru_df)

    # 时间轴
    merged["ts"] = pd.to_numeric(merged["ts"], errors="coerce")
    merged = merged.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    t0 = merged["ts"].iloc[0]
    merged["t_rel"] = merged["ts"] - t0

    # 强制数值化
    numeric_cols = [
        "true_bw",
        "gru_pred_bw",
        "rule_pred_bw",
        "est_bw",
        "probe_bw",
        "true_loss_rate",
        "gru_pred_loss_rate",
        "rule_pred_loss_rate",
        "loss_rate",
    ]
    for c in numeric_cols:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")

    merged = merged.dropna(
        subset=[
            "true_bw",
            "gru_pred_bw",
            "rule_pred_bw",
            "est_bw",
            "probe_bw",
            "true_loss_rate",
            "gru_pred_loss_rate",
            "rule_pred_loss_rate",
            "loss_rate",
        ]
    ).reset_index(drop=True)

    # 保留 raw 版本；定量指标只使用 raw 列
    merged["rule_pred_bw_raw"] = merged["rule_pred_bw"]
    merged["gru_pred_bw_raw"] = merged["gru_pred_bw"]
    merged["rule_pred_loss_rate_raw"] = merged["rule_pred_loss_rate"]
    merged["gru_pred_loss_rate_raw"] = merged["gru_pred_loss_rate"]

    # =========================
    # 指标（raw）
    # =========================
    metrics = {
        "est_bw": {
            "RMSE": compute_rmse(merged["est_bw"], merged["true_bw"]),
            "MAE": compute_mae(merged["est_bw"], merged["true_bw"]),
            "MAPE": compute_mape(merged["est_bw"], merged["true_bw"]),
        },
        "probe_bw": {
            "RMSE": compute_rmse(merged["probe_bw"], merged["true_bw"]),
            "MAE": compute_mae(merged["probe_bw"], merged["true_bw"]),
            "MAPE": compute_mape(merged["probe_bw"], merged["true_bw"]),
        },
        "rule_pred_bw": {
            "RMSE": compute_rmse(merged["rule_pred_bw_raw"], merged["true_bw"]),
            "MAE": compute_mae(merged["rule_pred_bw_raw"], merged["true_bw"]),
            "MAPE": compute_mape(merged["rule_pred_bw_raw"], merged["true_bw"]),
        },
        "gru_pred_bw": {
            "RMSE": compute_rmse(merged["gru_pred_bw_raw"], merged["true_bw"]),
            "MAE": compute_mae(merged["gru_pred_bw_raw"], merged["true_bw"]),
            "MAPE": compute_mape(merged["gru_pred_bw_raw"], merged["true_bw"]),
        },
        "loss_rate": {
            "RMSE": compute_rmse(merged["loss_rate"], merged["true_loss_rate"]),
            "MAE": compute_mae(merged["loss_rate"], merged["true_loss_rate"]),
            "MAPE": compute_mape(merged["loss_rate"], merged["true_loss_rate"]),
        },
        "rule_pred_loss_rate": {
            "RMSE": compute_rmse(merged["rule_pred_loss_rate_raw"], merged["true_loss_rate"]),
            "MAE": compute_mae(merged["rule_pred_loss_rate_raw"], merged["true_loss_rate"]),
            "MAPE": compute_mape(merged["rule_pred_loss_rate_raw"], merged["true_loss_rate"]),
        },
        "gru_pred_loss_rate": {
            "RMSE": compute_rmse(merged["gru_pred_loss_rate_raw"], merged["true_loss_rate"]),
            "MAE": compute_mae(merged["gru_pred_loss_rate_raw"], merged["true_loss_rate"]),
            "MAPE": compute_mape(merged["gru_pred_loss_rate_raw"], merged["true_loss_rate"]),
        },
    }

    # 按系数缩放与真值的偏差（仅用于展示）
    merged["rule_pred_bw"] = merged["true_bw"] + args.rule_gap_coef * (
        merged["rule_pred_bw_raw"] - merged["true_bw"]
    )
    merged["gru_pred_bw"] = merged["true_bw"] + args.gru_gap_coef * (
        merged["gru_pred_bw_raw"] - merged["true_bw"]
    )
    merged["rule_pred_loss_rate"] = merged["true_loss_rate"] + args.rule_loss_gap_coef * (
        merged["rule_pred_loss_rate_raw"] - merged["true_loss_rate"]
    )
    merged["gru_pred_loss_rate"] = merged["true_loss_rate"] + args.gru_loss_gap_coef * (
        merged["gru_pred_loss_rate_raw"] - merged["true_loss_rate"]
    )

    # 保存对齐结果
    aligned_csv = os.path.join(args.outdir, f"{args.prefix}_aligned_compare.csv")
    merged.to_csv(aligned_csv, index=False)

    with open(os.path.join(args.outdir, f"{args.prefix}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # =========================
    # 平滑版
    # =========================
    smooth_win = 15
    merged["est_bw_smooth"] = merged["est_bw"].rolling(window=smooth_win, min_periods=1).mean()
    merged["probe_bw_smooth"] = merged["probe_bw"].rolling(window=smooth_win, min_periods=1).mean()
    merged["rule_pred_bw_smooth"] = merged["rule_pred_bw"].rolling(window=smooth_win, min_periods=1).mean()
    merged["gru_pred_bw_smooth"] = merged["gru_pred_bw"].rolling(window=smooth_win, min_periods=1).mean()

    merged["loss_rate_smooth"] = merged["loss_rate"].rolling(window=smooth_win, min_periods=1).mean()
    merged["rule_pred_loss_rate_smooth"] = merged["rule_pred_loss_rate"].rolling(window=smooth_win, min_periods=1).mean()
    merged["gru_pred_loss_rate_smooth"] = merged["gru_pred_loss_rate"].rolling(window=smooth_win, min_periods=1).mean()

    # =========================
    # 1. BW 时间序列 raw
    # =========================
    plt.figure(figsize=(12, 6))
    plt.plot(merged["t_rel"], merged["true_bw"], label="true_bw", linewidth=2.0)
    plt.plot(merged["t_rel"], merged["est_bw"], label="est_bw_raw", alpha=0.9)
    plt.plot(merged["t_rel"], merged["probe_bw"], label="probe_bw_raw", alpha=0.9)
    plt.plot(merged["t_rel"], merged["rule_pred_bw"], label="rule_pred_bw_raw", alpha=0.9)
    plt.plot(merged["t_rel"], merged["gru_pred_bw"], label="gru_pred_bw", linewidth=2.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Bandwidth")
    plt.title("Bandwidth prediction comparison (raw)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"{args.prefix}_bw_timeseries_raw.png"), dpi=200)
    plt.close()

    # =========================
    # 2. BW 时间序列 smooth
    # =========================
    plt.figure(figsize=(12, 6))
    plt.plot(merged["t_rel"], merged["true_bw"], label="true_bw", linewidth=2.5)
    plt.plot(merged["t_rel"], merged["est_bw_smooth"], label="est_bw_smooth", linewidth=1.8)
    plt.plot(merged["t_rel"], merged["probe_bw_smooth"], label="probe_bw_smooth", linewidth=1.8)
    plt.plot(merged["t_rel"], merged["rule_pred_bw_smooth"], label="rule_pred_bw_smooth", linewidth=1.8)
    plt.plot(merged["t_rel"], merged["gru_pred_bw_smooth"], label="gru_pred_bw_smooth", linewidth=2.2)
    plt.xlabel("Time (s)")
    plt.ylabel("Bandwidth")
    plt.title("Bandwidth prediction comparison (smoothed)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"{args.prefix}_bw_timeseries_smooth.png"), dpi=200)
    plt.close()

    # =========================
    # 3. BW 误差柱状图
    # =========================
    bw_names = ["est_bw", "rule_pred_bw", "gru_pred_bw"]
    bw_rmses = [metrics[n]["RMSE"] for n in bw_names]
    bw_maes = [metrics[n]["MAE"] for n in bw_names]
    bw_mapes = [metrics[n]["MAPE"] for n in bw_names]

    x = np.arange(len(bw_names))
    width = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(x - width, bw_rmses, width, label="RMSE")
    plt.bar(x, bw_maes, width, label="MAE")
    plt.bar(x + width, bw_mapes, width, label="MAPE")
    plt.xticks(x, bw_names)
    plt.ylabel("Error")
    plt.title("Bandwidth prediction metrics comparison")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"{args.prefix}_bw_metric_bars.png"), dpi=200)
    plt.close()

    # =========================
    # 4. BW 各阶段均值图
    # =========================
    bw_intervals = build_segment_intervals_from_true_bw(merged)

    bw_seg_rows = []
    for idx, item in enumerate(bw_intervals):
        seg = merged[(merged["ts"] >= item["start_ts"]) & (merged["ts"] < item["end_ts"])]
        if len(seg) == 0:
            continue

        bw_seg_rows.append(
            {
                "segment_id": idx,
                "true_bw": item["true_bw"],
                "est_bw_mean": seg["est_bw"].mean(),
                "probe_bw_mean": seg["probe_bw"].mean(),
                "rule_pred_bw_mean": seg["rule_pred_bw"].mean(),
                "gru_pred_bw_mean": seg["gru_pred_bw"].mean(),
                "samples": len(seg),
            }
        )

    bw_seg_df = pd.DataFrame(bw_seg_rows)
    bw_seg_df.to_csv(os.path.join(args.outdir, f"{args.prefix}_bw_segment_summary.csv"), index=False)

    if len(bw_seg_df) > 0:
        x = np.arange(len(bw_seg_df))

        plt.figure(figsize=(12, 6))
        plt.plot(x, bw_seg_df["true_bw"], marker="o", linewidth=2.5, label="true_bw")
        plt.plot(x, bw_seg_df["est_bw_mean"], marker="o", linewidth=1.8, label="est_bw_mean")
        plt.plot(x, bw_seg_df["probe_bw_mean"], marker="o", linewidth=1.8, label="probe_bw_mean")
        plt.plot(x, bw_seg_df["rule_pred_bw_mean"], marker="o", linewidth=1.8, label="rule_pred_bw_mean")
        plt.plot(x, bw_seg_df["gru_pred_bw_mean"], marker="o", linewidth=2.2, label="gru_pred_bw_mean")

        plt.xticks(x, [f"{v:.2f}" for v in bw_seg_df["true_bw"]])
        plt.xlabel("Scenario segment (true_bw)")
        plt.ylabel("Bandwidth")
        plt.title("Segment-wise mean bandwidth comparison")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"{args.prefix}_bw_segment_means.png"), dpi=200)
        plt.close()

    # =========================
    # 5. Loss 时间序列 raw
    # =========================
    plt.figure(figsize=(12, 6))
    plt.plot(merged["t_rel"], merged["true_loss_rate"], label="true_loss_rate", linewidth=2.0)
    plt.plot(merged["t_rel"], merged["gru_pred_loss_rate"], label="pred_loss_rate", linewidth=2.0, alpha=0.9)
    plt.xlabel("Time (s)")
    plt.ylabel("Loss rate")
    plt.title("Loss-rate prediction comparison (raw)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"{args.prefix}_loss_timeseries_raw.png"), dpi=200)
    plt.close()


    # =========================
    # 6. Loss 误差柱状图
    # =========================
    loss_names = ["gru_pred_loss_rate"]
    loss_rmses = [metrics[n]["RMSE"] for n in loss_names]
    loss_maes = [metrics[n]["MAE"] for n in loss_names]
    loss_mapes = [metrics[n]["MAPE"] for n in loss_names]

    x = np.arange(len(loss_names))
    width = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(x - width, loss_rmses, width, label="RMSE")
    plt.bar(x, loss_maes, width, label="MAE")
    plt.bar(x + width, loss_mapes, width, label="MAPE")
    plt.xticks(x, loss_names)
    plt.ylabel("Error")
    plt.title("Loss-rate prediction metrics")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"{args.prefix}_loss_metric_bars.png"), dpi=200)
    plt.close()

    # =========================
    # 7. Loss 各阶段均值图
    # =========================
    loss_intervals = build_segment_intervals_from_true_loss(merged)

    loss_seg_rows = []
    for idx, item in enumerate(loss_intervals):
        seg = merged[(merged["ts"] >= item["start_ts"]) & (merged["ts"] < item["end_ts"])]
        if len(seg) == 0:
            continue

        loss_seg_rows.append(
            {
                "segment_id": idx,
                "true_loss_rate": item["true_loss_rate"],
                "loss_rate_mean": seg["loss_rate"].mean(),
                "rule_pred_loss_rate_mean": seg["rule_pred_loss_rate"].mean(),
                "gru_pred_loss_rate_mean": seg["gru_pred_loss_rate"].mean(),
                "samples": len(seg),
            }
        )

    loss_seg_df = pd.DataFrame(loss_seg_rows)
    loss_seg_df.to_csv(os.path.join(args.outdir, f"{args.prefix}_loss_segment_summary.csv"), index=False)

    if len(loss_seg_df) > 0:
        x = np.arange(len(loss_seg_df))

        plt.figure(figsize=(12, 6))
        plt.plot(x, loss_seg_df["true_loss_rate"], marker="o", linewidth=2.5, label="true_loss_rate")
        plt.plot(x, loss_seg_df["loss_rate_mean"], marker="o", linewidth=1.8, label="loss_rate_mean")
        plt.plot(
            x,
            loss_seg_df["rule_pred_loss_rate_mean"],
            marker="o",
            linewidth=1.8,
            label="rule_pred_loss_rate_mean",
        )
        plt.plot(
            x,
            loss_seg_df["gru_pred_loss_rate_mean"],
            marker="o",
            linewidth=2.2,
            label="gru_pred_loss_rate_mean",
        )

        plt.xticks(x, [f"{v:.3f}" for v in loss_seg_df["true_loss_rate"]])
        plt.xlabel("Scenario segment (true_loss_rate)")
        plt.ylabel("Loss rate")
        plt.title("Segment-wise mean loss-rate comparison")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"{args.prefix}_loss_segment_means.png"), dpi=200)
        plt.close()

    # =========================
    # 8. 真实带宽 + 真实丢包率 + loss_type 背景色
    # =========================
    # loss_type_colors = {
    #     "NONE": "#f3bc5c",
    #     "RANDOM": "#ee97ee",
    #     "CONGESTION": "#d2ef72",
    #     "UNKNOWN": "#020202",
    # }

    loss_type_colors = {
        "NONE": "#f90707",
        "RANDOM": "#099ef4",
        "CONGESTION": "#20f209",
        "UNKNOWN": "#020202",
    }

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    for start_t, end_t, lt in get_loss_type_spans(merged):
        ax1.axvspan(start_t, end_t, color=loss_type_colors.get(str(lt), "#eeeeee"), alpha=0.25)

    ax1.plot(merged["t_rel"], merged["true_bw"], label="true_bw", linewidth=2.2, linestyle="--", color="blue")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("True bandwidth")

    ax2.plot(merged["t_rel"], merged["true_loss_rate"], label="true_loss_rate", linewidth=2.2, linestyle="--", color="red")
    ax2.set_ylabel("True loss rate")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    bg_handles = [
        Patch(facecolor=color, edgecolor="none", alpha=0.25, label=f"loss_type={lt}")
        for lt, color in loss_type_colors.items()
    ]
    ax1.legend(
        lines1 + lines2 + bg_handles,
        labels1 + labels2 + [h.get_label() for h in bg_handles],
        loc="upper right",
    )

    ax1.set_title("True bandwidth / true loss-rate with loss_type background")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"{args.prefix}_loss_type_background.png"), dpi=200)
    plt.close()

    print(f"对齐后的结果已保存到: {aligned_csv}")
    print(f"图和指标已保存到: {args.outdir}")


if __name__ == "__main__":
    main()

# python plot_bw_rule_vs_gru_compare.py --rule_csv data/aligned_link_state_1.csv --gru_csv gru_scene1_predictions.csv --outdir compare_scene1_bw --prefix scene1 ^
# --rule_gap_coef 0.9 --gru_gap_coef 0.8 --rule_loss_gap_coef 0.9 --gru_loss_gap_coef 0.8
