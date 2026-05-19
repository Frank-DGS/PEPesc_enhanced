import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from adaptive_data import ensure_model_feature_columns, safe_read_csv
from adaptive_ml import (
    RunningStandardizer,
    build_model,
    clip_nonnegative_prediction,
    default_feature_cols_for_model,
    resolve_model_type,
)
from adaptive_schema import TARGET_BW, TARGET_LOSS, TIME_COL


class InferenceSequenceDataset(Dataset):
    def __init__(self, df, seq_len, feature_cols, model_type, scaler_x):
        super().__init__()
        df = ensure_model_feature_columns(df, feature_cols)
        keep_cols = [TIME_COL, *feature_cols, TARGET_BW, TARGET_LOSS]
        if "loss_type" in df.columns:
            keep_cols.append("loss_type")
        df = df[keep_cols].dropna().sort_values(TIME_COL).reset_index(drop=True)

        self.ts = df[TIME_COL].astype(float).values
        self.true_abs = df[[TARGET_BW, TARGET_LOSS]].astype(float).values
        self.loss_type = df["loss_type"].astype(str).values if "loss_type" in df.columns else ["NONE"] * len(df)
        self.baseline = (
            df[["pred_bw", "pred_loss_rate"]].astype(float).values
            if model_type == "hybrid_residual"
            else None
        )
        self.x = scaler_x.transform(df[feature_cols].astype(float).values)
        self.seq_len = seq_len
        self.model_type = model_type

        if len(self.x) < seq_len:
            raise ValueError(f"Data length {len(self.x)} smaller than seq_len={seq_len}")

    def __len__(self):
        return len(self.x) - self.seq_len + 1

    def __getitem__(self, idx):
        end_idx = idx + self.seq_len - 1
        baseline = self.baseline[end_idx] if self.baseline is not None else [0.0, 0.0]
        return (
            torch.tensor(self.x[idx : idx + self.seq_len], dtype=torch.float32),
            torch.tensor(self.ts[end_idx], dtype=torch.float64),
            torch.tensor(self.true_abs[end_idx], dtype=torch.float32),
            torch.tensor(baseline, dtype=torch.float32),
            self.loss_type[end_idx],
        )


@torch.no_grad()
def predict(model, loader, device, scaler_y, model_type):
    model.eval()
    rows = []
    for x, ts, true_abs, baseline_abs, loss_type in loader:
        x = x.to(device)
        pred_model = model(x).cpu().numpy()
        pred_model = scaler_y.inverse_transform(pred_model)
        baseline_np = baseline_abs.numpy()
        true_np = true_abs.numpy()
        ts_np = ts.numpy()
        pred_abs = pred_model + baseline_np if model_type == "hybrid_residual" else pred_model

        for idx in range(len(ts_np)):
            pred_bw, pred_loss_rate = clip_nonnegative_prediction(pred_abs[idx, 0], pred_abs[idx, 1])
            rows.append(
                {
                    "ts": float(ts_np[idx]),
                    "true_bw": float(true_np[idx, 0]),
                    "pred_bw": pred_bw,
                    "true_loss_rate": float(true_np[idx, 1]),
                    "pred_loss_rate": pred_loss_rate,
                    "base_bw": float(baseline_np[idx, 0]),
                    "base_loss_rate": float(baseline_np[idx, 1]),
                    "pred_residual_bw": float(pred_model[idx, 0]),
                    "pred_residual_loss_rate": float(pred_model[idx, 1]),
                    "loss_type": str(loss_type[idx]),
                }
            )
    return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Input aligned_link_state csv")
    parser.add_argument("--checkpoint", type=str, required=True, help="Trained checkpoint")
    parser.add_argument("--output_csv", type=str, required=True, help="Output prediction csv")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]
    model_type = resolve_model_type(config)
    feature_cols = config.get("feature_cols", default_feature_cols_for_model(model_type))
    scaler_x = RunningStandardizer.from_state_dict(ckpt["scaler_x"])
    scaler_y = RunningStandardizer.from_state_dict(ckpt["scaler_y"])

    model = build_model(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    df = safe_read_csv(args.csv)
    dataset = InferenceSequenceDataset(df, config["seq_len"], feature_cols, model_type, scaler_x)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    pred_df = predict(model, loader, device, scaler_y, model_type)
    pred_df.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
