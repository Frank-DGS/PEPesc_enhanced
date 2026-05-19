from dataclasses import dataclass
from typing import Tuple


TIME_COL = "ts"
TARGET_BW = "true_bw"
TARGET_LOSS = "true_loss_rate"

LEGACY_FEATURE_COLS = [
    "est_bw",
    "est_bw_max",
    "probe_bw",
    "rtt",
    "rtt_min",
    "queue_delay",
    "rtt_gradient",
    "loss_rate",
    "packets_in_flight",
    "cwnd",
    "queue_pressure",
    "recent_burst",
    "burst_degree",
]

HYBRID_EXTRA_FEATURE_COLS = [
    "pred_bw",
    "pred_loss_rate",
    "loss_random_score",
    "loss_congestion_score",
    "ack_gap_ms",
    "streamc_queue_size",
    "decoder_active",
]

DERIVED_FEATURE_COLS = [
    "est_bw_delta",
    "pred_bw_delta",
    "loss_rate_delta",
    "queue_delay_delta",
    "ack_gap_delta_ms",
    "rtt_over_min",
    "ack_gap_over_rttmin",
    "probe_minus_est",
    "pred_minus_est",
    "inflight_over_cwnd",
]

HYBRID_FEATURE_COLS = LEGACY_FEATURE_COLS + HYBRID_EXTRA_FEATURE_COLS + DERIVED_FEATURE_COLS
DEFAULT_RESAMPLE_MS = 50
LOSS_TYPE_LABELS = ("NONE", "RANDOM", "CONGESTION")

TRAINING_KEEP_COLS = [
    TIME_COL,
    *HYBRID_FEATURE_COLS,
    "loss_type",
    TARGET_BW,
    TARGET_LOSS,
]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def normalize_loss_type(value) -> str:
    text = str(value).strip().upper()
    if text not in LOSS_TYPE_LABELS:
        return "NONE"
    return text


def loss_type_to_scores(loss_type: str) -> Tuple[float, float]:
    normalized = normalize_loss_type(loss_type)
    if normalized == "RANDOM":
        return 1.0, 0.0
    if normalized == "CONGESTION":
        return 0.0, 1.0
    return 0.0, 0.0


@dataclass
class EmaFilter:
    alpha: float = 0.35
    value: float = 0.0
    initialized: bool = False

    def update(self, sample: float) -> float:
        alpha = clamp(float(self.alpha), 0.01, 1.0)
        sample = float(sample)
        if not self.initialized:
            self.value = sample
            self.initialized = True
            return self.value
        self.value = alpha * sample + (1.0 - alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = 0.0
        self.initialized = False


@dataclass
class LossSignalTracker:
    hysteresis_ticks: int = 2
    stable_label: str = "NONE"
    pending_label: str = "NONE"
    pending_ticks: int = 0
    random_score: float = 0.0
    congestion_score: float = 0.0

    def _choose_candidate(self, loss_rate: float) -> str:
        if loss_rate < 0.003 and max(self.random_score, self.congestion_score) < 0.35:
            return "NONE"
        if self.congestion_score >= self.random_score + 0.05:
            return "CONGESTION"
        if self.random_score >= self.congestion_score + 0.05:
            return "RANDOM"
        return self.stable_label

    def update(
        self,
        *,
        loss_rate: float,
        queue_delay: float,
        rtt_gradient: float,
        queue_pressure: float,
        recent_burst: float,
        burst_degree: float,
        ack_gap_ms: float,
        streamc_queue_size: float,
        decoder_active: float,
        rtt_min: float,
    ) -> dict:
        q_thresh = max(0.020, 0.20 * max(rtt_min, 0.0))
        g_thresh = max(0.004, 0.05 * max(rtt_min, 0.0))
        ack_thresh_ms = max(20.0, 1.25 * max(rtt_min, 0.0) * 1000.0)

        random_score = 0.0
        congestion_score = 0.0

        if loss_rate >= 0.003:
            random_score += clamp((loss_rate - 0.003) / 0.05, 0.0, 0.45)
            congestion_score += clamp((loss_rate - 0.003) / 0.05, 0.0, 0.25)

        if queue_delay <= 0.6 * q_thresh:
            random_score += 0.18
        else:
            congestion_score += clamp(queue_delay / max(q_thresh, 1e-6), 0.0, 0.30)

        if rtt_gradient <= 0.5 * g_thresh:
            random_score += 0.12
        else:
            congestion_score += clamp(rtt_gradient / max(g_thresh, 1e-6), 0.0, 0.20)

        if queue_pressure >= 1.0:
            congestion_score += clamp(queue_pressure - 0.95, 0.0, 0.15)
        else:
            random_score += 0.05

        if ack_gap_ms >= ack_thresh_ms:
            congestion_score += 0.12
        else:
            random_score += 0.05

        if recent_burst >= 0.5:
            random_score += min(0.20, 0.02 * max(burst_degree, 0.0))

        if streamc_queue_size >= 10:
            congestion_score += min(0.12, 0.01 * streamc_queue_size)

        if decoder_active >= 0.5:
            congestion_score += 0.08

        self.random_score = clamp(random_score, 0.0, 1.0)
        self.congestion_score = clamp(congestion_score, 0.0, 1.0)
        candidate = self._choose_candidate(loss_rate)

        if candidate == self.stable_label:
            self.pending_label = candidate
            self.pending_ticks = 0
        else:
            if candidate != self.pending_label:
                self.pending_label = candidate
                self.pending_ticks = 1
            else:
                self.pending_ticks += 1
            if self.pending_ticks >= self.hysteresis_ticks:
                self.stable_label = candidate
                self.pending_ticks = 0

        return {
            "loss_type": self.stable_label,
            "loss_random_score": self.random_score,
            "loss_congestion_score": self.congestion_score,
        }


@dataclass
class DecisionController:
    state: str = "HOLD"
    last_pacing_mbps: float = 0.0
    last_fec_ratio: float = 0.02
    brake_signal_ticks: int = 0
    clean_ticks: int = 0
    accel_ticks: int = 0
    brake_hold_ticks: int = 0

    def update(
        self,
        *,
        sensed_bw_mbps: float,
        sensed_loss_rate: float,
        loss_type: str,
        est_bw_mbps: float,
        probe_bw_mbps: float,
        queue_delay: float,
        rtt_min: float,
        rtt_gradient: float,
        queue_pressure: float,
        ack_gap_ms: float,
        recent_burst: float,
        burst_degree: float,
        streamc_queue_size: float,
        decoder_active: float,
    ) -> dict:
        q_thresh = max(0.020, 0.20 * max(rtt_min, 0.0))
        g_thresh = max(0.004, 0.05 * max(rtt_min, 0.0))
        ack_thresh_ms = max(20.0, 1.25 * max(rtt_min, 0.0) * 1000.0)

        floor_mbps = max(0.1, 0.8 * max(est_bw_mbps, 0.1))
        ceiling_mbps = max(0.2, 1.2 * max(est_bw_mbps, probe_bw_mbps, sensed_bw_mbps, 0.2))
        bw_ref = clamp(sensed_bw_mbps, floor_mbps, ceiling_mbps)

        if self.last_pacing_mbps <= 0:
            self.last_pacing_mbps = bw_ref

        brake_signal = normalize_loss_type(loss_type) == "CONGESTION" and (
            queue_delay > q_thresh
            or rtt_gradient > g_thresh
            or ack_gap_ms > ack_thresh_ms
            or queue_pressure > 1.10
            or streamc_queue_size > 12
            or decoder_active >= 0.5
        )

        clean_signal = (
            normalize_loss_type(loss_type) != "CONGESTION"
            and queue_delay < 0.60 * q_thresh
            and rtt_gradient < 0.60 * g_thresh
            and ack_gap_ms < 0.90 * ack_thresh_ms
            and queue_pressure < 1.0
        )

        accel_signal = clean_signal and bw_ref > 1.02 * self.last_pacing_mbps

        self.brake_signal_ticks = self.brake_signal_ticks + 1 if brake_signal else 0
        self.clean_ticks = self.clean_ticks + 1 if clean_signal else 0
        self.accel_ticks = self.accel_ticks + 1 if accel_signal else 0

        if self.state != "BRAKE" and self.brake_signal_ticks >= 2:
            self.state = "BRAKE"
            self.brake_hold_ticks = 0
        elif self.state == "BRAKE":
            self.brake_hold_ticks += 1
            if self.brake_hold_ticks >= 3 and self.clean_ticks >= 3:
                self.state = "HOLD"
                self.brake_hold_ticks = 0
        elif self.accel_ticks >= 2:
            self.state = "ACCEL"
        else:
            self.state = "HOLD"

        prev_pacing = self.last_pacing_mbps
        if self.state == "BRAKE":
            target_pacing = max(0.70 * prev_pacing, 0.85 * bw_ref)
        elif self.state == "ACCEL":
            target_pacing = min(1.08 * prev_pacing, 1.05 * bw_ref)
        else:
            target_pacing = 0.80 * prev_pacing + 0.20 * (0.95 * bw_ref)

        applied_pacing = clamp(0.70 * prev_pacing + 0.30 * target_pacing, floor_mbps, ceiling_mbps)

        burst_boost = min(0.05, 0.005 * max(burst_degree, 0.0)) if recent_burst >= 0.5 else 0.0
        normalized_type = normalize_loss_type(loss_type)
        if normalized_type == "RANDOM":
            target_fec = clamp(sensed_loss_rate + 0.025 + burst_boost, 0.01, 0.18)
        elif normalized_type == "CONGESTION":
            # Streaming-code decoding still needs loss compensation; pacing handles congestion.
            target_fec = clamp(sensed_loss_rate + 0.020, 0.0, 0.12)
        else:
            target_fec = clamp(0.50 * sensed_loss_rate + 0.002, 0.0, 0.03)

        applied_fec = clamp(0.70 * self.last_fec_ratio + 0.30 * target_fec, 0.0, 0.18)

        self.last_pacing_mbps = applied_pacing
        self.last_fec_ratio = applied_fec

        return {
            "decision_state": self.state,
            "decision_bw_ref_mbps": bw_ref,
            "decision_pacing_mbps": applied_pacing,
            "decision_fec_ratio": applied_fec,
            "decision_brake_signal": 1 if brake_signal else 0,
        }
