import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from adaptive_data import ensure_model_feature_columns, safe_read_csv
from adaptive_ml import (
    RunningStandardizer,
    build_model,
    clip_nonnegative_prediction,
    default_feature_cols_for_model,
    resolve_model_type,
)
from adaptive_schema import TARGET_BW, TARGET_LOSS, TIME_COL


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    return float(np.mean(np.abs((pred - true) / np.maximum(np.abs(true), 1e-9))))


def resolve_scene_path(data_dir: str, scene_idx: int, preferred_ms: int) -> str:
    candidates = [
        os.path.join(data_dir, f"aligned_link_state_{scene_idx}_{preferred_ms}ms.csv"),
        os.path.join(data_dir, f"aligned_link_state_{scene_idx}.csv"),
        os.path.join(data_dir, f"aligned_link_state_{scene_idx}_raw.csv"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Scene {scene_idx} csv not found under {data_dir}")


def parse_scene_ids(text: str) -> List[int]:
    ids = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        ids.append(int(token))
    if not ids:
        raise ValueError(f"No scene ids parsed from: {text}")
    return ids


def load_split_by_files(data_dir: str, preferred_ms: int, train_ids: List[int], val_ids: List[int], test_ids: List[int]):
    split = {
        "train": train_ids,
        "test": test_ids,
        "val": val_ids,
    }

    def read_list(indices):
        result = []
        for idx in indices:
            path = resolve_scene_path(data_dir, idx, preferred_ms)
            df = safe_read_csv(path).sort_values(TIME_COL).reset_index(drop=True)
            result.append((path, df))
        return result

    return read_list(split["train"]), read_list(split["val"]), read_list(split["test"])


def fit_standardizers_from_pairs(
    pairs,
    feature_cols: List[str],
    model_type: str,
    scaler_x: RunningStandardizer,
    scaler_y: RunningStandardizer,
) -> None:
    for _, df in pairs:
        aligned = ensure_model_feature_columns(df, feature_cols)
        required_cols = [TIME_COL, *feature_cols, TARGET_BW, TARGET_LOSS]
        aligned = aligned[required_cols].dropna().sort_values(TIME_COL).reset_index(drop=True)
        x_all = aligned[feature_cols].astype(float).to_numpy()
        true_abs = aligned[[TARGET_BW, TARGET_LOSS]].astype(float).to_numpy()
        if model_type == "hybrid_residual":
            baseline = aligned[["pred_bw", "pred_loss_rate"]].astype(float).to_numpy()
            y_all = true_abs - baseline
        else:
            y_all = true_abs
        scaler_x.partial_fit(x_all)
        scaler_y.partial_fit(y_all)


class SequenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int,
        feature_cols: List[str],
        model_type: str,
        scaler_x: Optional[RunningStandardizer] = None,
        scaler_y: Optional[RunningStandardizer] = None,
        fit_scaler: bool = False,
        transition_window_s: float = 2.0,
        transition_weight: float = 2.0,
    ):
        super().__init__()
        df = ensure_model_feature_columns(df, feature_cols)
        required_cols = [TIME_COL, *feature_cols, TARGET_BW, TARGET_LOSS]
        df = df[required_cols].dropna().sort_values(TIME_COL).reset_index(drop=True)

        self.ts = df[TIME_COL].astype(float).to_numpy()
        x_all = df[feature_cols].astype(float).to_numpy()
        true_abs = df[[TARGET_BW, TARGET_LOSS]].astype(float).to_numpy()

        if model_type == "hybrid_residual":
            baseline = df[["pred_bw", "pred_loss_rate"]].astype(float).to_numpy()
            y_all = true_abs - baseline
        else:
            y_all = true_abs

        if scaler_x is not None and fit_scaler:
            scaler_x.partial_fit(x_all)
        if scaler_y is not None and fit_scaler:
            scaler_y.partial_fit(y_all)
        if scaler_x is not None:
            x_all = scaler_x.transform(x_all)
        if scaler_y is not None:
            y_all = scaler_y.transform(y_all)

        self.x = x_all
        self.y = y_all
        self.seq_len = seq_len
        self.sample_weights = np.ones(len(self.x), dtype=np.float32)

        if transition_window_s > 0 and transition_weight > 1.0 and len(self.ts) > 1:
            bw_all = df[TARGET_BW].astype(float).to_numpy()
            loss_all = df[TARGET_LOSS].astype(float).to_numpy()
            change_mask = np.zeros(len(self.ts), dtype=bool)
            change_mask[1:] = (np.abs(np.diff(bw_all)) > 1e-9) | (np.abs(np.diff(loss_all)) > 1e-12)
            last_change_ts = None
            for i, ts_value in enumerate(self.ts):
                if change_mask[i]:
                    last_change_ts = ts_value
                if last_change_ts is not None and (ts_value - last_change_ts) <= transition_window_s:
                    self.sample_weights[i] = float(transition_weight)

        if len(self.x) < seq_len:
            raise ValueError(f"Data length {len(self.x)} smaller than seq_len={seq_len}")

    def __len__(self):
        return len(self.x) - self.seq_len + 1

    def __getitem__(self, idx: int):
        end_idx = idx + self.seq_len - 1
        return (
            torch.tensor(self.x[idx : idx + self.seq_len], dtype=torch.float32),
            torch.tensor(self.y[end_idx], dtype=torch.float32),
            torch.tensor(self.ts[end_idx], dtype=torch.float64),
            torch.tensor(self.sample_weights[end_idx], dtype=torch.float32),
        )


class PredictionSequenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int,
        feature_cols: List[str],
        model_type: str,
        scaler_x: RunningStandardizer,
    ):
        super().__init__()
        df = ensure_model_feature_columns(df, feature_cols)
        keep_cols = [TIME_COL, *feature_cols, TARGET_BW, TARGET_LOSS]
        if "loss_type" in df.columns:
            keep_cols.append("loss_type")
        df = df[keep_cols].dropna().sort_values(TIME_COL).reset_index(drop=True)

        self.ts = df[TIME_COL].astype(float).to_numpy()
        self.true_abs = df[[TARGET_BW, TARGET_LOSS]].astype(float).to_numpy()
        self.loss_type = df["loss_type"].astype(str).to_numpy() if "loss_type" in df.columns else np.array(["NONE"] * len(df))
        self.baseline = (
            df[["pred_bw", "pred_loss_rate"]].astype(float).to_numpy()
            if model_type == "hybrid_residual"
            else np.zeros((len(df), 2), dtype=np.float64)
        )
        self.x = scaler_x.transform(df[feature_cols].astype(float).to_numpy())
        self.seq_len = seq_len
        self.model_type = model_type

        if len(self.x) < seq_len:
            raise ValueError(f"Data length {len(self.x)} smaller than seq_len={seq_len}")

    def __len__(self):
        return len(self.x) - self.seq_len + 1

    def __getitem__(self, idx: int):
        end_idx = idx + self.seq_len - 1
        return (
            torch.tensor(self.x[idx : idx + self.seq_len], dtype=torch.float32),
            torch.tensor(self.ts[end_idx], dtype=torch.float64),
            torch.tensor(self.true_abs[end_idx], dtype=torch.float32),
            torch.tensor(self.baseline[end_idx], dtype=torch.float32),
            self.loss_type[end_idx],
        )


def train_one_epoch(model, loader, optimizer, device, loss_fn):
    model.train()
    total_loss = 0.0
    total_weight = 0.0
    for x, y, _, sample_weight in loader:
        x = x.to(device)
        y = y.to(device)
        sample_weight = sample_weight.to(device)
        optimizer.zero_grad()
        pred = model(x)
        per_sample_loss = loss_fn(pred, y).mean(dim=1)
        loss = (per_sample_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)
        loss.backward()
        optimizer.step()
        total_loss += float((per_sample_loss * sample_weight).sum().item())
        total_weight += float(sample_weight.sum().item())
    return total_loss / max(total_weight, 1e-6)


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    for x, y, _, sample_weight in loader:
        x = x.to(device)
        y = y.to(device)
        sample_weight = sample_weight.to(device)
        pred = model(x)
        per_sample_loss = loss_fn(pred, y).mean(dim=1)
        total_loss += float((per_sample_loss * sample_weight).sum().item())
        total_weight += float(sample_weight.sum().item())
    return total_loss / max(total_weight, 1e-6)


@torch.no_grad()
def predict_dataset(model, loader, device, scaler_y: RunningStandardizer, model_type: str):
    model.eval()
    rows = []
    for x, ts, true_abs, baseline_abs, loss_type in loader:
        x = x.to(device)
        pred_model = model(x).cpu().numpy()
        pred_model = scaler_y.inverse_transform(pred_model)
        true_abs_np = true_abs.numpy()
        baseline_np = baseline_abs.numpy()
        ts_np = ts.numpy()

        if model_type == "hybrid_residual":
            pred_abs = baseline_np + pred_model
        else:
            pred_abs = pred_model

        for idx in range(len(ts_np)):
            pred_bw, pred_loss_rate = clip_nonnegative_prediction(pred_abs[idx, 0], pred_abs[idx, 1])
            rows.append(
                {
                    "ts": float(ts_np[idx]),
                    "true_bw": float(true_abs_np[idx, 0]),
                    "pred_bw": pred_bw,
                    "true_loss_rate": float(true_abs_np[idx, 1]),
                    "pred_loss_rate": pred_loss_rate,
                    "base_bw": float(baseline_np[idx, 0]),
                    "base_loss_rate": float(baseline_np[idx, 1]),
                    "pred_residual_bw": float(pred_model[idx, 0]),
                    "pred_residual_loss_rate": float(pred_model[idx, 1]),
                    "loss_type": str(loss_type[idx]),
                }
            )

    return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)


def save_checkpoint(path, model, optimizer, epoch, scaler_x, scaler_y, config):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_x": scaler_x.state_dict(),
            "scaler_y": scaler_y.state_dict(),
            "config": config,
        },
        path,
    )


def load_checkpoint(path, device):
    return torch.load(path, map_location=device)


def plot_regression_timeseries(df: pd.DataFrame, outdir: str, split_name: str):
    if len(df) == 0:
        return
    t_rel = df["ts"] - df["ts"].iloc[0]

    plt.figure(figsize=(12, 5))
    plt.plot(t_rel, df["true_bw"], label="true_bw")
    plt.plot(t_rel, df["pred_bw"], label="pred_bw")
    if "base_bw" in df.columns:
        plt.plot(t_rel, df["base_bw"], label="rule_base_bw", alpha=0.6)
    plt.xlabel("Time (s)")
    plt.ylabel("Bandwidth (Mbps)")
    plt.title(f"{split_name}: true_bw vs pred_bw")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{split_name}_bw_timeseries.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(t_rel, df["true_loss_rate"], label="true_loss_rate")
    plt.plot(t_rel, df["pred_loss_rate"], label="pred_loss_rate")
    if "base_loss_rate" in df.columns:
        plt.plot(t_rel, df["base_loss_rate"], label="rule_base_loss", alpha=0.6)
    plt.xlabel("Time (s)")
    plt.ylabel("Loss rate")
    plt.title(f"{split_name}: true_loss_rate vs pred_loss_rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{split_name}_loss_timeseries.png"), dpi=200)
    plt.close()


def plot_scatter(df: pd.DataFrame, outdir: str, split_name: str):
    if len(df) == 0:
        return

    plt.figure(figsize=(5, 5))
    plt.scatter(df["true_bw"], df["pred_bw"], alpha=0.5)
    mn = min(df["true_bw"].min(), df["pred_bw"].min())
    mx = max(df["true_bw"].max(), df["pred_bw"].max())
    plt.plot([mn, mx], [mn, mx], "--")
    plt.xlabel("true_bw")
    plt.ylabel("pred_bw")
    plt.title(f"{split_name}: bw scatter")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{split_name}_bw_scatter.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.scatter(df["true_loss_rate"], df["pred_loss_rate"], alpha=0.5)
    mn = min(df["true_loss_rate"].min(), df["pred_loss_rate"].min())
    mx = max(df["true_loss_rate"].max(), df["pred_loss_rate"].max())
    plt.plot([mn, mx], [mn, mx], "--")
    plt.xlabel("true_loss_rate")
    plt.ylabel("pred_loss_rate")
    plt.title(f"{split_name}: loss scatter")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{split_name}_loss_scatter.png"), dpi=200)
    plt.close()


def save_prediction_metrics(df: pd.DataFrame, outdir: str, split_name: str):
    metrics = {
        "bw_rmse": compute_rmse(df["pred_bw"], df["true_bw"]),
        "bw_mae": compute_mae(df["pred_bw"], df["true_bw"]),
        "bw_mape": compute_mape(df["pred_bw"], df["true_bw"]),
        "loss_rmse": compute_rmse(df["pred_loss_rate"], df["true_loss_rate"]),
        "loss_mae": compute_mae(df["pred_loss_rate"], df["true_loss_rate"]),
        "loss_mape": compute_mape(df["pred_loss_rate"], df["true_loss_rate"]),
    }
    with open(os.path.join(outdir, f"{split_name}_prediction_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics


def plot_loss_curve(history: list, outdir: str):
    if not history:
        return
    hist_df = pd.DataFrame(history)
    plt.figure(figsize=(8, 5))
    plt.plot(hist_df["epoch"], hist_df["train_loss"], label="train_loss")
    plt.plot(hist_df["epoch"], hist_df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training / Validation Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_curve.png"), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Incremental GRU training with absolute and hybrid residual modes")
    parser.add_argument("--data_dir", type=str, required=True, help="Folder storing aligned_link_state_<scene>_<ms>.csv files")
    parser.add_argument("--checkpoint", type=str, default="gru_reg_checkpoint.pt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model_type", type=str, default="hybrid_residual", choices=["gru_absolute", "hybrid_residual"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_best_only", action="store_true")
    parser.add_argument("--outdir", type=str, default="gru_outputs")
    parser.add_argument("--preferred_ms", type=int, default=50, help="Prefer *_50ms.csv if present")
    parser.add_argument("--train_ids", type=str, default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16")
    parser.add_argument("--val_ids", type=str, default="17")
    parser.add_argument("--test_ids", type=str, default="18")
    parser.add_argument("--transition_window_s", type=float, default=2.0)
    parser.add_argument("--transition_weight", type=float, default=2.0)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

    train_ids = parse_scene_ids(args.train_ids)
    val_ids = parse_scene_ids(args.val_ids)
    test_ids = parse_scene_ids(args.test_ids)

    train_pairs, val_pairs, test_pairs = load_split_by_files(
        args.data_dir,
        args.preferred_ms,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
    )
    print("Using scene files:")
    print(f"  train_ids={train_ids}")
    print(f"  val_ids={val_ids}")
    print(f"  test_ids={test_ids}")
    for split_name, pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        print(f"  [{split_name}]")
        for path, _ in pairs:
            print(f"    {Path(path).resolve()}")

    start_epoch = 0
    best_val = float("inf")
    history = []

    if args.resume and os.path.exists(args.checkpoint):
        ckpt = load_checkpoint(args.checkpoint, device)
        config = ckpt["config"]
        model_type = resolve_model_type(config)
        feature_cols = config.get("feature_cols", default_feature_cols_for_model(model_type))
        scaler_x = RunningStandardizer.from_state_dict(ckpt["scaler_x"])
        scaler_y = RunningStandardizer.from_state_dict(ckpt["scaler_y"])
        model = build_model(config).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception:
            print("Warning: optimizer state load failed, use fresh optimizer.")
        start_epoch = ckpt["epoch"] + 1
        print(f"[Resume] loaded checkpoint from {args.checkpoint}, start_epoch={start_epoch}")
    else:
        model_type = args.model_type
        feature_cols = default_feature_cols_for_model(model_type)
        scaler_x = RunningStandardizer.create(dim=len(feature_cols))
        scaler_y = RunningStandardizer.create(dim=2)
        config = {
            "model_type": model_type,
            "input_dim": len(feature_cols),
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "seq_len": args.seq_len,
            "feature_cols": feature_cols,
            "targets": [TARGET_BW, TARGET_LOSS],
        }
        model = build_model(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if not (args.resume and os.path.exists(args.checkpoint)):
        fit_standardizers_from_pairs(train_pairs, feature_cols, model_type, scaler_x, scaler_y)

    train_sets = [
        SequenceDataset(
            df,
            config["seq_len"],
            feature_cols,
            model_type,
            scaler_x,
            scaler_y,
            fit_scaler=False,
            transition_window_s=args.transition_window_s,
            transition_weight=args.transition_weight,
        )
        for _, df in train_pairs
    ]
    val_sets = [
        SequenceDataset(
            df,
            config["seq_len"],
            feature_cols,
            model_type,
            scaler_x,
            scaler_y,
            fit_scaler=False,
            transition_window_s=args.transition_window_s,
            transition_weight=args.transition_weight,
        )
        for _, df in val_pairs
    ]
    test_sets = [
        SequenceDataset(
            df,
            config["seq_len"],
            feature_cols,
            model_type,
            scaler_x,
            scaler_y,
            fit_scaler=False,
            transition_window_s=args.transition_window_s,
            transition_weight=args.transition_weight,
        )
        for _, df in test_pairs
    ]

    train_loader = DataLoader(ConcatDataset(train_sets), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ConcatDataset(val_sets), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(ConcatDataset(test_sets), batch_size=args.batch_size, shuffle=False)

    loss_fn = nn.MSELoss(reduction="none")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        val_loss = evaluate(model, val_loader, device, loss_fn)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[Epoch {epoch}] train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if args.save_best_only:
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(args.checkpoint, model, optimizer, epoch, scaler_x, scaler_y, config)
        else:
            save_checkpoint(args.checkpoint, model, optimizer, epoch, scaler_x, scaler_y, config)

    if os.path.exists(args.checkpoint):
        best_ckpt = load_checkpoint(args.checkpoint, device)
        model = build_model(best_ckpt["config"]).to(device)
        model.load_state_dict(best_ckpt["model_state_dict"], strict=True)
        scaler_x = RunningStandardizer.from_state_dict(best_ckpt["scaler_x"])
        scaler_y = RunningStandardizer.from_state_dict(best_ckpt["scaler_y"])
        config = best_ckpt["config"]
        model_type = resolve_model_type(config)
        feature_cols = config.get("feature_cols", default_feature_cols_for_model(model_type))

    val_pred_loader = DataLoader(
        ConcatDataset([PredictionSequenceDataset(df, config["seq_len"], feature_cols, model_type, scaler_x) for _, df in val_pairs]),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_pred_loader = DataLoader(
        ConcatDataset([PredictionSequenceDataset(df, config["seq_len"], feature_cols, model_type, scaler_x) for _, df in test_pairs]),
        batch_size=args.batch_size,
        shuffle=False,
    )

    val_loss = evaluate(model, val_loader, device, loss_fn)
    test_loss = evaluate(model, test_loader, device, loss_fn)
    val_pred_df = predict_dataset(model, val_pred_loader, device, scaler_y, model_type)
    test_pred_df = predict_dataset(model, test_pred_loader, device, scaler_y, model_type)

    with open(os.path.join(args.outdir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "history": history,
                "final_val_metrics": {"loss": val_loss},
                "final_test_metrics": {"loss": test_loss},
                "effective_checkpoint": os.path.abspath(args.checkpoint),
                "model_type": model_type,
                "train_ids": train_ids,
                "val_ids": val_ids,
                "test_ids": test_ids,
                "transition_window_s": args.transition_window_s,
                "transition_weight": args.transition_weight,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    plot_loss_curve(history, args.outdir)
    val_pred_df.to_csv(os.path.join(args.outdir, "val_predictions.csv"), index=False)
    test_pred_df.to_csv(os.path.join(args.outdir, "test_predictions.csv"), index=False)

    val_pred_metrics = save_prediction_metrics(val_pred_df, args.outdir, "val")
    test_pred_metrics = save_prediction_metrics(test_pred_df, args.outdir, "test")

    plot_regression_timeseries(val_pred_df, args.outdir, "val")
    plot_regression_timeseries(test_pred_df, args.outdir, "test")
    plot_scatter(val_pred_df, args.outdir, "val")
    plot_scatter(test_pred_df, args.outdir, "test")

    print("\nValidation prediction metrics:")
    print(json.dumps(val_pred_metrics, indent=2, ensure_ascii=False))
    print("\nTest prediction metrics:")
    print(json.dumps(test_pred_metrics, indent=2, ensure_ascii=False))
    print(f"\nAll outputs saved to: {os.path.abspath(args.outdir)}")
    print(f"Checkpoint used for final outputs: {os.path.abspath(args.checkpoint)}")


if __name__ == "__main__":
    main()
