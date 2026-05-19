from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd

from adaptive_schema import (
    DEFAULT_RESAMPLE_MS,
    DERIVED_FEATURE_COLS,
    HYBRID_EXTRA_FEATURE_COLS,
    TRAINING_KEEP_COLS,
    loss_type_to_scores,
)


BANDWIDTH_FEATURE_COLS = ["est_bw", "est_bw_max", "probe_bw", "pred_bw"]


def safe_read_csv(path: Union[str, Path]) -> pd.DataFrame:
    return pd.read_csv(path, on_bad_lines="skip")


def _ensure_numeric_column(df: pd.DataFrame, col: str, default: float) -> None:
    if col not in df.columns:
        df[col] = default
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)


def add_loss_score_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "loss_type" not in out.columns:
        out["loss_type"] = "NONE"

    random_scores = []
    congestion_scores = []
    for loss_type in out["loss_type"].astype(str):
        random_score, congestion_score = loss_type_to_scores(loss_type)
        random_scores.append(random_score)
        congestion_scores.append(congestion_score)

    if "loss_random_score" not in out.columns:
        out["loss_random_score"] = random_scores
    if "loss_congestion_score" not in out.columns:
        out["loss_congestion_score"] = congestion_scores

    return out


def add_derived_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    base_defaults = {
        "est_bw": 0.0,
        "probe_bw": 0.0,
        "pred_bw": 0.0,
        "loss_rate": 0.0,
        "queue_delay": 0.0,
        "ack_gap_ms": 0.0,
        "rtt": 0.0,
        "rtt_min": 0.0,
        "packets_in_flight": 0.0,
        "cwnd": 0.0,
    }
    for col, default in base_defaults.items():
        _ensure_numeric_column(out, col, default)

    out["est_bw_delta"] = out["est_bw"].diff().fillna(0.0)
    out["pred_bw_delta"] = out["pred_bw"].diff().fillna(0.0)
    out["loss_rate_delta"] = out["loss_rate"].diff().fillna(0.0)
    out["queue_delay_delta"] = out["queue_delay"].diff().fillna(0.0)
    out["ack_gap_delta_ms"] = out["ack_gap_ms"].diff().fillna(0.0)

    rtt_min_safe = out["rtt_min"].replace(0.0, np.nan)
    out["rtt_over_min"] = (out["rtt"] / rtt_min_safe).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["ack_gap_over_rttmin"] = (
        out["ack_gap_ms"] / (1000.0 * rtt_min_safe)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["probe_minus_est"] = out["probe_bw"] - out["est_bw"]
    out["pred_minus_est"] = out["pred_bw"] - out["est_bw"]

    cwnd_safe = out["cwnd"].replace(0.0, np.nan)
    out["inflight_over_cwnd"] = (
        out["packets_in_flight"] / cwnd_safe
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for col in DERIVED_FEATURE_COLS:
        _ensure_numeric_column(out, col, 0.0)

    return out


def ensure_model_feature_columns(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    out = add_loss_score_columns(df.copy())

    for col in BANDWIDTH_FEATURE_COLS:
        mbps_col = f"{col}_mbps"
        if mbps_col in out.columns:
            out[col] = pd.to_numeric(out[mbps_col], errors="coerce")

    if "pred_bw" in feature_cols and "pred_bw" not in out.columns and "est_bw" in out.columns:
        out["pred_bw"] = out["est_bw"]
    if "pred_loss_rate" in feature_cols and "pred_loss_rate" not in out.columns and "loss_rate" in out.columns:
        out["pred_loss_rate"] = out["loss_rate"]

    default_fill = {
        "ack_gap_ms": 0.0,
        "streamc_queue_size": 0.0,
        "decoder_active": 0.0,
        "loss_random_score": 0.0,
        "loss_congestion_score": 0.0,
    }

    for col, default in default_fill.items():
        if col in out.columns or col in feature_cols:
            _ensure_numeric_column(out, col, default)

    out = add_derived_feature_columns(out)

    for col in feature_cols:
        _ensure_numeric_column(out, col, default_fill.get(col, 0.0))

    for col in HYBRID_EXTRA_FEATURE_COLS:
        if col in out.columns:
            _ensure_numeric_column(out, col, default_fill.get(col, 0.0))

    return out


def sanitize_ts(df: pd.DataFrame, col: str = "ts") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=[col])
    out = out[(out[col] > 1e8) & (out[col] < 1e11)]
    return out.sort_values(col).reset_index(drop=True)


def resample_aligned_training_frame(df: pd.DataFrame, interval_ms: int = DEFAULT_RESAMPLE_MS) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()

    out = sanitize_ts(df, "ts")
    interval_s = interval_ms / 1000.0
    t0 = out["ts"].iloc[0]
    t_last = out["ts"].iloc[-1]
    out["bin_idx"] = np.floor((out["ts"] - t0) / interval_s).astype(int)
    grouped = out.groupby("bin_idx", sort=True).last()

    full_index = np.arange(grouped.index.min(), grouped.index.max() + 1)
    grouped = grouped.reindex(full_index)
    grouped = grouped.ffill()
    grouped["ts"] = t0 + full_index * interval_s
    grouped = grouped.reset_index(drop=True)
    return grouped


def build_training_keep_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = add_loss_score_columns(df.copy())
    for col in BANDWIDTH_FEATURE_COLS:
        mbps_col = f"{col}_mbps"
        if mbps_col in out.columns:
            out[col] = pd.to_numeric(out[mbps_col], errors="coerce")
    if "true_bw_mbps" in out.columns:
        out["true_bw"] = pd.to_numeric(out["true_bw_mbps"], errors="coerce")
    numeric_defaults = {
        "ack_gap_ms": 0.0,
        "streamc_queue_size": 0.0,
        "decoder_active": 0.0,
        "loss_random_score": 0.0,
        "loss_congestion_score": 0.0,
    }
    for col in TRAINING_KEEP_COLS:
        if col == "loss_type":
            if col not in out.columns:
                out[col] = "NONE"
            continue
        if col not in out.columns:
            out[col] = numeric_defaults.get(col, np.nan)
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = add_derived_feature_columns(out)
    out = out[TRAINING_KEEP_COLS].sort_values("ts").reset_index(drop=True)
    return out
