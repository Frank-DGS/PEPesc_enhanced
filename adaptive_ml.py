from collections import deque
from dataclasses import dataclass
from typing import Dict

import numpy as np

from adaptive_schema import HYBRID_FEATURE_COLS, LEGACY_FEATURE_COLS

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - runtime fallback
    torch = None
    nn = None


@dataclass
class RunningStandardizer:
    count: int
    mean: np.ndarray
    m2: np.ndarray
    eps: float = 1e-8

    @classmethod
    def create(cls, dim: int):
        return cls(
            count=0,
            mean=np.zeros(dim, dtype=np.float64),
            m2=np.zeros(dim, dtype=np.float64),
            eps=1e-8,
        )

    def partial_fit(self, x: np.ndarray) -> None:
        if x.ndim != 2:
            raise ValueError("partial_fit expects x with shape [N, D]")
        for row in x:
            self.count += 1
            delta = row - self.mean
            self.mean += delta / self.count
            delta2 = row - self.mean
            self.m2 += delta * delta2

    @property
    def var(self) -> np.ndarray:
        if self.count < 2:
            return np.ones_like(self.mean, dtype=np.float64)
        v = self.m2 / max(self.count - 1, 1)
        return np.where(v < self.eps, 1.0, v)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def state_dict(self) -> Dict:
        return {
            "count": self.count,
            "mean": self.mean.tolist(),
            "m2": self.m2.tolist(),
            "eps": self.eps,
        }

    @classmethod
    def from_state_dict(cls, state: Dict):
        return cls(
            count=int(state["count"]),
            mean=np.array(state["mean"], dtype=np.float64),
            m2=np.array(state["m2"], dtype=np.float64),
            eps=float(state.get("eps", 1e-8)),
        )


if nn is not None:
    class _GRUBackbone(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
            super().__init__()
            gru_dropout = dropout if num_layers > 1 else 0.0
            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=gru_dropout,
            )
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            )

        def forward(self, x):
            out, _ = self.gru(x)
            feat = self.dropout(out[:, -1, :])
            return self.head(feat)


    class GRUAbsoluteRegressor(_GRUBackbone):
        pass


    class HybridResidualGRU(_GRUBackbone):
        pass


def resolve_model_type(config: Dict) -> str:
    return config.get("model_type", "gru_absolute")


def default_feature_cols_for_model(model_type: str):
    if model_type == "hybrid_residual":
        return HYBRID_FEATURE_COLS
    return LEGACY_FEATURE_COLS


def clip_nonnegative_prediction(pred_bw: float, pred_loss_rate: float):
    return max(0.0, float(pred_bw)), max(0.0, float(pred_loss_rate))


def add_runtime_derived_features(feature_map: Dict[str, float], prev_feature_map: Dict[str, float] = None) -> Dict[str, float]:
    prev_feature_map = prev_feature_map or {}
    out = dict(feature_map)

    def cur(name: str) -> float:
        return float(feature_map.get(name, 0.0))

    def prv(name: str) -> float:
        if name in prev_feature_map:
            return float(prev_feature_map.get(name, 0.0))
        return cur(name)

    def safe_ratio(numer: float, denom: float) -> float:
        return 0.0 if abs(denom) < 1e-9 else float(numer / denom)

    out["est_bw_delta"] = cur("est_bw") - prv("est_bw")
    out["pred_bw_delta"] = cur("pred_bw") - prv("pred_bw")
    out["loss_rate_delta"] = cur("loss_rate") - prv("loss_rate")
    out["queue_delay_delta"] = cur("queue_delay") - prv("queue_delay")
    out["ack_gap_delta_ms"] = cur("ack_gap_ms") - prv("ack_gap_ms")
    out["rtt_over_min"] = safe_ratio(cur("rtt"), cur("rtt_min"))
    out["ack_gap_over_rttmin"] = safe_ratio(cur("ack_gap_ms"), cur("rtt_min") * 1000.0)
    out["probe_minus_est"] = cur("probe_bw") - cur("est_bw")
    out["pred_minus_est"] = cur("pred_bw") - cur("est_bw")
    out["inflight_over_cwnd"] = safe_ratio(cur("packets_in_flight"), cur("cwnd"))
    return out


def build_model(config: Dict):
    if torch is None:
        raise ImportError("PyTorch is required to build the GRU model.")

    model_type = resolve_model_type(config)
    input_dim = int(config["input_dim"])
    hidden_dim = int(config["hidden_dim"])
    num_layers = int(config["num_layers"])
    dropout = float(config["dropout"])

    if model_type == "hybrid_residual":
        return HybridResidualGRU(input_dim, hidden_dim, num_layers, dropout)
    return GRUAbsoluteRegressor(input_dim, hidden_dim, num_layers, dropout)


class HybridSenseRuntime:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        if torch is None:
            raise ImportError("PyTorch is required for hybrid GRU runtime.")

        self.device = torch.device(device)
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = self.checkpoint["config"]
        self.model_type = resolve_model_type(self.config)
        self.feature_cols = self.config.get(
            "feature_cols",
            default_feature_cols_for_model(self.model_type),
        )
        self.seq_len = int(self.config["seq_len"])
        self.scaler_x = RunningStandardizer.from_state_dict(self.checkpoint["scaler_x"])
        self.scaler_y = RunningStandardizer.from_state_dict(self.checkpoint["scaler_y"])
        self.model = build_model(self.config).to(self.device)
        self.model.load_state_dict(self.checkpoint["model_state_dict"], strict=True)
        self.model.eval()
        self.history = deque(maxlen=self.seq_len)
        self.last_prediction = None
        self.last_feature_map = None

    def push(self, feature_map: Dict[str, float]) -> None:
        feature_map = add_runtime_derived_features(feature_map, self.last_feature_map)
        feature_row = np.array(
            [float(feature_map.get(col, 0.0)) for col in self.feature_cols],
            dtype=np.float32,
        )
        baseline = np.array(
            [
                float(feature_map.get("pred_bw", 0.0)),
                float(feature_map.get("pred_loss_rate", 0.0)),
            ],
            dtype=np.float32,
        )
        self.history.append((feature_row, baseline))
        self.last_feature_map = dict(feature_map)

    def predict(self) -> Dict[str, float]:
        if not self.history:
            return {
                "ready": False,
                "pred_bw": 0.0,
                "pred_loss_rate": 0.0,
                "residual_bw": 0.0,
                "residual_loss_rate": 0.0,
            }

        last_baseline = self.history[-1][1]
        if len(self.history) < self.seq_len:
            return {
                "ready": False,
                "pred_bw": float(last_baseline[0]),
                "pred_loss_rate": float(last_baseline[1]),
                "residual_bw": 0.0,
                "residual_loss_rate": 0.0,
            }

        x = np.stack([row for row, _ in self.history], axis=0)
        x = self.scaler_x.transform(x)
        x_tensor = torch.tensor(x[None, :, :], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            pred = self.model(x_tensor).cpu().numpy()

        pred = self.scaler_y.inverse_transform(pred)[0]
        residual_bw = float(pred[0])
        residual_loss = float(pred[1])

        if self.model_type == "hybrid_residual":
            pred_bw = float(last_baseline[0] + residual_bw)
            pred_loss = float(last_baseline[1] + residual_loss)
        else:
            pred_bw = residual_bw
            pred_loss = residual_loss
            residual_bw = pred_bw - float(last_baseline[0])
            residual_loss = pred_loss - float(last_baseline[1])

        pred_bw, pred_loss = clip_nonnegative_prediction(pred_bw, pred_loss)

        self.last_prediction = {
            "ready": True,
            "pred_bw": pred_bw,
            "pred_loss_rate": pred_loss,
            "residual_bw": residual_bw,
            "residual_loss_rate": residual_loss,
        }
        return self.last_prediction
