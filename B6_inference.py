#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B5 - Inference for ACLA + ANODE (Hướng 2: sau phiên sạc)
--------------------------------------------------------
• Dùng checkpoint đã train (acla_anode_best.pt) từ các script B3/B4 của bạn.
• Nhận dữ liệu mới ở dạng NPZ từ pipeline tiền xử lý (giống training):
    - X: (N, 61, C)
    - features: list tên kênh (để kiểm tra thứ tự)
    - (tùy chọn) y: nếu có thì sẽ tính RMSE để backtest
• Áp dụng chuẩn hoá per-feature bằng thống kê TRAIN lưu sẵn trong checkpoint.
• Xuất kết quả ra CSV.

Cách dùng:
  1) Sửa CONFIG bên dưới (đường dẫn ckpt & npz input)
  2) Chạy:  python B5_infer_acla_anode.py

Yêu cầu:
  pip install torch numpy pandas torchdiffeq
"""

# =========================
# CONFIG (edit here)
CKPT_PATH  = r"E:\ACLA + ANODE\weight\acla_anode_best_v1.pt"         # checkpoint đã train
NPZ_INPUT  = r"E:\ACLA + ANODE\output_infer\ocpp_lstm_infer.npz"     # npz mới
META_PATH  = r"E:\ACLA + ANODE\output_infer\ocpp_lstm_sessions_with_user_car.csv"   # (tuỳ chọn)
OUT_CSV    = r"E:\ACLA + ANODE\soh_predictions_mc_with_ids_v2.csv"
USE_CPU    = False

# MC Dropout
MC_SAMPLES  = 20      # số lần forward; 0/1 = tắt MC (mean=pred, std=0)
BATCH_SIZE  = 256     # batch inference
# =========================

import os, math, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---- torchdiffeq (ANODE) ----
try:
    from torchdiffeq import odeint_adjoint as odeint
    ADJOINT_DEFAULT = True
except Exception:
    try:
        from torchdiffeq import odeint
        ADJOINT_DEFAULT = False
    except Exception as e:
        raise RuntimeError("Cần cài torchdiffeq: pip install torchdiffeq") from e


# ===== Model pieces (khớp với training) =====
class ODEFunc(nn.Module):
    def __init__(self, dim: int, hidden_mlp: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_mlp), nn.Tanh(),
            nn.Linear(hidden_mlp, hidden_mlp), nn.Tanh(),
            nn.Linear(hidden_mlp, dim),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, t, z):
        return self.net(z)


class ODEBlock(nn.Module):
    def __init__(self, odefunc: nn.Module, t_end: float = 1.0, method: str = "dopri5", rtol: float = 1e-4, atol: float = 1e-5):
        super().__init__()
        self.odefunc = odefunc
        self.t_end = t_end
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def forward(self, z0):
        t = torch.tensor([0.0, self.t_end], dtype=z0.dtype, device=z0.device)
        zt = odeint(self.odefunc, z0, t, method=self.method, rtol=self.rtol, atol=self.atol)
        return zt[-1]


class ACLA_ANODE(nn.Module):
    def __init__(self, input_channels=10, conv_channels=(64, 32), lstm_hidden=64,
                 aug_dim=20, ode_hidden_mlp=128, ode_method="dopri5", t_end=1.0):
        super().__init__()
        self.attn = nn.Linear(input_channels, input_channels)
        c1, c2 = conv_channels
        self.conv1 = nn.Conv1d(input_channels, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=c2, hidden_size=lstm_hidden, batch_first=True)
        ode_dim = lstm_hidden + aug_dim
        self.odefunc = ODEFunc(ode_dim, hidden_mlp=ode_hidden_mlp)
        self.odeblock = ODEBlock(self.odefunc, t_end=t_end, method=ode_method)
        self.readout = nn.Linear(lstm_hidden + aug_dim, 1)
        self.aux_head = nn.Conv1d(c2, 1, kernel_size=1)

        for m in [self.conv1, self.conv2, self.readout]:
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        attn_w = torch.sigmoid(self.attn(x))  # (B,61,C)
        x = x * attn_w
        x_cnn = x.permute(0,2,1)
        x_cnn = torch.relu(self.conv1(x_cnn))
        x_cnn = torch.relu(self.conv2(x_cnn))
        aux_seq = self.aux_head(x_cnn).squeeze(1)  # (B,61)
        x_lstm = x_cnn.permute(0,2,1)
        _, (h_n, _) = self.lstm(x_lstm)
        h = h_n[-1]
        B = h.shape[0]
        aug_dim = self.odefunc.net[-1].out_features - h.shape[1]
        z0 = torch.cat([h, torch.zeros(B, aug_dim, device=h.device)], dim=1)
        z1 = self.odeblock(z0)
        soh = self.readout(z1).squeeze(1)
        return soh, aux_seq


def load_checkpoint(path: str | Path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    required = ["model", "config", "norm"]
    for k in required:
        if k not in ckpt:
            raise RuntimeError(f"Checkpoint thiếu khoá '{k}'. Hãy dùng ckpt từ B3/B4 (norm + config).")
    return ckpt


def load_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"].astype(np.float32)               # (N,61,C)
    y = data["y"].astype(np.float32) if "y" in data.files else None
    features = list(data["features"]) if "features" in data.files else None

    # Các id/meta nếu có trong npz
    # Lưu ý: tuỳ B1_preprocess mà có/không, nên cần .get() an toàn
    id_fields = {}
    for key in ["session_id", "car_id", "user_id", "car_model_id", "car_model_name", "car_type"]:
        if key in data.files:
            id_fields[key] = data[key]

    return X, y, features, id_fields


def apply_normalization(X: torch.Tensor, mu: np.ndarray, std: np.ndarray) -> torch.Tensor:
    mu_t  = torch.tensor(mu, dtype=torch.float32, device=X.device)
    std_t = torch.tensor(std, dtype=torch.float32, device=X.device)
    return (X - mu_t) / torch.clamp(std_t, min=1e-6)


def rmse_np(a, b) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def predict_mc(model: nn.Module, X: torch.Tensor, mc_samples: int, batch_size: int):
    """
    Trả về:
      mean_pred: (N,)
      std_pred:  (N,)
    """
    N = X.shape[0]
    all_mean = np.zeros(N, dtype=np.float32)
    all_std  = np.zeros(N, dtype=np.float32)

    if mc_samples <= 1:
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, N, batch_size):
                xb = X[i:i+batch_size]
                yh = model(xb)[0].cpu().numpy()
                preds.append(yh)
        y_pred = np.concatenate(preds, axis=0)
        return y_pred, np.zeros_like(y_pred)

    # MC Dropout: để dropout hoạt động, bật train() nhưng tắt grad
    model.train()
    with torch.no_grad():
        for i in range(0, N, batch_size):
            xb = X[i:i+batch_size]
            bag = []
            for _ in range(mc_samples):
                yh = model(xb)[0].cpu().numpy()
                bag.append(yh)
            bag = np.stack(bag, axis=0)     # (S, B)
            all_mean[i:i+batch_size] = bag.mean(axis=0)
            all_std[i:i+batch_size]  = bag.std(axis=0)
    return all_mean, all_std


def merge_meta(df_pred: pd.DataFrame, meta_path: str | None, id_fields: dict) -> pd.DataFrame:
    if id_fields:
        # Đưa các id từ npz vào ngay từ đầu (nếu có)
        for k, v in id_fields.items():
            try:
                # v có thể là object/bytes → chuyển sang Series an toàn
                df_pred[k] = pd.Series(v).reset_index(drop=True)
            except Exception:
                pass

    if not meta_path:
        return df_pred

    try:
        meta = pd.read_csv(meta_path)
    except Exception as e:
        print(f"[WARN] Không đọc được META_PATH: {e}")
        return df_pred

    # Ưu tiên 1: merge theo 'session_id' nếu có ở cả hai
    if ("session_id" in meta.columns) and ("session_id" in df_pred.columns):
        before = len(df_pred)
        df_merged = df_pred.merge(meta, on="session_id", how="left", suffixes=("", "_meta"))
        print(f"[META] Merge by session_id: {before} rows -> {len(df_merged)} rows")
        return df_merged

    # Ưu tiên 2: merge theo 'row_idx' nếu có trong meta
    if "row_idx" in meta.columns:
        meta2 = meta.copy()
        meta2["index"] = meta2["row_idx"].astype(int)
        before = len(df_pred)
        df_merged = df_pred.merge(meta2.drop(columns=["row_idx"]), on="index", how="left", suffixes=("", "_meta"))
        print(f"[META] Merge by row_idx/index: {before} rows -> {len(df_merged)} rows")
        return df_merged

    # Fall-back: nếu số dòng khớp → ghép theo thứ tự
    if len(meta) == len(df_pred):
        df_merged = pd.concat([df_pred.reset_index(drop=True), meta.reset_index(drop=True)], axis=1)
        print(f"[META] Concat by order (no key). Rows={len(df_merged)}")
        return df_merged

    print("[WARN] META không có khoá (session_id/row_idx) và số dòng không khớp → bỏ qua merge.")
    return df_pred


def main():
    device = torch.device("cpu" if USE_CPU or not torch.cuda.is_available() else "cuda")

    # 1) Load checkpoint & config
    ckpt = load_checkpoint(CKPT_PATH, device)
    cfg = ckpt["config"]; norm = ckpt["norm"]
    print(f"[LOAD] ckpt: {CKPT_PATH}")
    print(f"[INFO] val_RMSE(best) = {ckpt.get('val_rmse', 'N/A')} | epoch = {ckpt.get('epoch', 'N/A')}")

    # 2) Load new data
    X_np, y_np, features, id_fields = load_npz(NPZ_INPUT)
    N, T, C = X_np.shape
    print(f"[LOAD] NPZ: {NPZ_INPUT} | X shape={X_np.shape} | y={'yes' if y_np is not None else 'no'}")
    if id_fields:
        print(f"[INFO] Found id fields in NPZ: {list(id_fields.keys())}")

    # 3) Build model with config from ckpt
    model = ACLA_ANODE(
        input_channels=C,
        conv_channels=(64, 32),
        lstm_hidden=int(cfg.get("lstm_hidden", 64)),
        aug_dim=int(cfg.get("aug_dim", 20)),
        ode_hidden_mlp=int(cfg.get("ode_hidden", 128)),
        ode_method=str(cfg.get("ode_method", "dopri5")),
        t_end=float(cfg.get("t_end", 1.0))
    ).to(device)
    model.load_state_dict(ckpt["model"])

    # 4) Apply normalization from ckpt (TRAIN stats)
    mu = norm["mu"]; std = norm["std"]
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    X = apply_normalization(X, mu, std)

    # 5) MC predictions
    mean_pred, std_pred = predict_mc(model, X, mc_samples=MC_SAMPLES, batch_size=BATCH_SIZE)

    # 6) Build output table (kèm SOH true nếu có)
    out = {"index": np.arange(N), "soh_pred_mean": mean_pred, "soh_pred_std": std_pred}
    if y_np is not None:
        out["soh_true"] = y_np
        out["abs_err"]  = np.abs(mean_pred - y_np)
        out["squared_err"] = (mean_pred - y_np) ** 2

    df = pd.DataFrame(out)

    # 7) Merge meta/session/vehicle
    df = merge_meta(df, META_PATH, id_fields)

    # 8) Summaries
    if "soh_true" in df.columns:
        rmse = rmse_np(df["soh_pred_mean"].values, df["soh_true"].values)
        mae  = float(np.mean(np.abs(df["soh_pred_mean"].values - df["soh_true"].values)))
        print(f"[METRIC] RMSE={rmse:.4f} | MAE={mae:.4f}")
        print(f"[NOTE] Trung bình std (uncertainty): {df['soh_pred_std'].mean():.4f}")
    else:
        print("[METRIC] y_true không có trong NPZ mới → chỉ xuất dự báo.")
        print(f"[NOTE] Trung bình std (uncertainty): {df['soh_pred_std'].mean():.4f}")

    # 9) Save CSV
    out_path = Path(OUT_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[SAVE] Predictions -> {out_path.resolve()}")

if __name__ == "__main__":
    main()