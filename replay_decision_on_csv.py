import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from adaptive_data import add_loss_score_columns, resample_aligned_training_frame, safe_read_csv
from adaptive_schema import DecisionController, EmaFilter, LossSignalTracker, normalize_loss_type


def maybe_resample(df: pd.DataFrame, interval_ms: int) -> pd.DataFrame:
    if len(df) < 3:
        return df.copy()
    dt_ms = (df["ts"].diff().dropna().median()) * 1000.0
    if dt_ms < 0.5 * interval_ms:
        return resample_aligned_training_frame(df, interval_ms=interval_ms)
    return df.copy()


def replay(
    df: pd.DataFrame,
    bw_col: str,
    loss_col: str,
    recompute_loss_type: bool,
    sense_bw_ema_alpha: float,
    sense_loss_ema_alpha: float,
) -> pd.DataFrame:
    controller = DecisionController()
    loss_tracker = LossSignalTracker()
    bw_ema = EmaFilter(alpha=sense_bw_ema_alpha)
    loss_ema = EmaFilter(alpha=sense_loss_ema_alpha)
    rows = []

    for _, row in df.iterrows():
        if recompute_loss_type:
            loss_view = loss_tracker.update(
                loss_rate=float(row.get(loss_col, 0.0)),
                queue_delay=float(row.get("queue_delay", 0.0)),
                rtt_gradient=float(row.get("rtt_gradient", 0.0)),
                queue_pressure=float(row.get("queue_pressure", 0.0)),
                recent_burst=float(row.get("recent_burst", 0.0)),
                burst_degree=float(row.get("burst_degree", 0.0)),
                ack_gap_ms=float(row.get("ack_gap_ms", 0.0)),
                streamc_queue_size=float(row.get("streamc_queue_size", 0.0)),
                decoder_active=float(row.get("decoder_active", 0.0)),
                rtt_min=float(row.get("rtt_min", 0.0)),
            )
            loss_type = loss_view["loss_type"]
            loss_random_score = loss_view["loss_random_score"]
            loss_congestion_score = loss_view["loss_congestion_score"]
        else:
            loss_type = normalize_loss_type(row.get("loss_type", "NONE"))
            loss_random_score = float(row.get("loss_random_score", 0.0))
            loss_congestion_score = float(row.get("loss_congestion_score", 0.0))

        sensed_bw_raw = max(0.0, float(row.get(bw_col, 0.0)))
        sensed_loss_raw = max(0.0, float(row.get(loss_col, 0.0)))
        sensed_bw = max(0.0, bw_ema.update(sensed_bw_raw))
        sensed_loss_rate = max(0.0, loss_ema.update(sensed_loss_raw))

        decision = controller.update(
            sensed_bw_mbps=sensed_bw,
            sensed_loss_rate=sensed_loss_rate,
            loss_type=loss_type,
            est_bw_mbps=float(row.get("est_bw", 0.0)),
            probe_bw_mbps=float(row.get("probe_bw", 0.0)),
            queue_delay=float(row.get("queue_delay", 0.0)),
            rtt_min=float(row.get("rtt_min", 0.0)),
            rtt_gradient=float(row.get("rtt_gradient", 0.0)),
            queue_pressure=float(row.get("queue_pressure", 0.0)),
            ack_gap_ms=float(row.get("ack_gap_ms", 0.0)),
            recent_burst=float(row.get("recent_burst", 0.0)),
            burst_degree=float(row.get("burst_degree", 0.0)),
            streamc_queue_size=float(row.get("streamc_queue_size", 0.0)),
            decoder_active=float(row.get("decoder_active", 0.0)),
        )

        rows.append(
            {
                "ts": float(row["ts"]),
                "true_bw": float(row.get("true_bw", 0.0)),
                "true_loss_rate": float(row.get("true_loss_rate", 0.0)),
                "sensed_bw_raw": sensed_bw_raw,
                "sensed_loss_rate_raw": sensed_loss_raw,
                "sensed_bw": sensed_bw,
                "sensed_loss_rate": sensed_loss_rate,
                "loss_type": loss_type,
                "loss_random_score": loss_random_score,
                "loss_congestion_score": loss_congestion_score,
                **decision,
            }
        )

    return pd.DataFrame(rows)


def make_plots(df: pd.DataFrame, outdir: Path) -> None:
    if len(df) == 0:
        return
    t_rel = df["ts"] - df["ts"].iloc[0]

    plt.figure(figsize=(12, 5))
    plt.plot(t_rel, df["sensed_bw"], label="sensed_bw")
    plt.plot(t_rel, df["decision_pacing_mbps"], label="decision_pacing_mbps")
    if "true_bw" in df.columns:
        plt.plot(t_rel, df["true_bw"], label="true_bw", linestyle="--", alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Bandwidth / pacing (Mbps)")
    plt.title("Replay pacing trajectory")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "replay_pacing.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(t_rel, df["sensed_loss_rate"], label="sensed_loss_rate")
    plt.plot(t_rel, df["decision_fec_ratio"], label="decision_fec_ratio")
    if "true_loss_rate" in df.columns:
        plt.plot(t_rel, df["true_loss_rate"], label="true_loss_rate", linestyle="--", alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Loss / FEC ratio")
    plt.title("Replay FEC trajectory")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "replay_fec.png", dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Replay DecisionController on aligned csv")
    parser.add_argument("--input_csv", required=True, help="Input aligned_link_state / prediction csv")
    parser.add_argument("--outdir", default="decision_replay_outputs")
    parser.add_argument("--bw_col", default="pred_bw", help="Bandwidth column used as sensed_bw")
    parser.add_argument("--loss_col", default="pred_loss_rate", help="Loss column used as sensed_loss_rate")
    parser.add_argument("--resample_ms", type=int, default=50)
    parser.add_argument("--recompute_loss_type", action="store_true", help="Recompute loss_type from numeric features")
    parser.add_argument("--sense_bw_ema_alpha", type=float, default=0.35, help="EMA alpha for sensed bandwidth before replay controller")
    parser.add_argument("--sense_loss_ema_alpha", type=float, default=0.25, help="EMA alpha for sensed loss rate before replay controller")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = safe_read_csv(args.input_csv)
    df = add_loss_score_columns(df)
    df = df.sort_values("ts").reset_index(drop=True)
    df = maybe_resample(df, args.resample_ms)

    replay_df = replay(
        df,
        args.bw_col,
        args.loss_col,
        args.recompute_loss_type,
        args.sense_bw_ema_alpha,
        args.sense_loss_ema_alpha,
    )
    replay_df.to_csv(outdir / "decision_replay.csv", index=False)
    make_plots(replay_df, outdir)

    summary = {
        "rows": int(len(replay_df)),
        "state_counts": replay_df["decision_state"].value_counts().to_dict(),
        "mean_pacing_mbps": float(replay_df["decision_pacing_mbps"].mean()),
        "mean_fec_ratio": float(replay_df["decision_fec_ratio"].mean()),
    }
    with open(outdir / "decision_replay_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Replay outputs saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
