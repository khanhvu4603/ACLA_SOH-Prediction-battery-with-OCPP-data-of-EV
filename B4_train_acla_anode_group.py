#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B4 - Train ACLA + ANODE with:
  • Per-feature Z-score normalization (fit on TRAIN)
  • Best checkpoint by val_RMSE(SOH)
  • **Group split** by car or user to avoid leakage (each group appears in only one split)

Run:
  python B4_train_acla_anode_group.py

Edit the CONFIG block below before running.
"""

# =========================
# CONFIG (edit here)
# =========================
NPZ_PATH      = r"/path/to/ocpp_lstm_dataset.npz"       # dataset (npz)
META_PATH     = r"/path/to/ocpp_lstm_sessions.csv"      # meta csv from preprocessing (same order as X)
GROUP_FIELD   = "car_name"    # one of columns in META (e.g., "car_name", "user_id", "tag_id")
GROUP_SPLIT   = (0.7, 0.15, 0.15)  # train, val, test ratios by **group**

OUT_DIR       = r"./acla_out_group"
AUX_FEATURE   = "dSOC_dt"
USE_CPU       = False

# Model
LSTM_HIDDEN   = 64
AUG_DIM       = 10
ODE_HIDDEN    = 128
ODE_METHOD    = "dopri5"
T_END         = 1.0

# Training
EPOCHS        = 200
BATCH_SIZE    = 32
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
LAMBDA_AUX    = 0.0     # recommend start with 0 for noisy OCPP aux
PATIENCE      = 20
SEED          = 1337
# =========================

import os, math, random, csv
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---- torchdiffeq (ANODE) ----
try:
    from torchdiffeq import odeint_adjoint as odeint
    ADJOINT_DEFAULT = True
except Exception:
    try:
        from torchdiffeq import odeint
        ADJOINT_DEFAULT = False
    except Exception as e:
        raise RuntimeError("Please install torchdiffeq: pip install torchdiffeq") from e


def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rmse_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def rmse_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((a - b) ** 2) + 1e-12)


def assert_exists(path: str | Path, kind="file") -> Path:
    p = Path(path)
    if kind == "file":
        ok = p.exists() and p.is_file()
    else:
        ok = p.exists()
    if not ok:
        raise FileNotFoundError(f"{kind.capitalize()} not found: {p}")
    return p.resolve()


def load_npz(npz_path: str, aux_feature_name: str = "dSOC_dt"):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]            # (N, 61, C)
    y = data["y"]            # (N,)
    features = list(data["features"]) if "features" in data.files else None
    soc_grid = data["soc_grid"] if "soc_grid" in data.files else None

    aux_target = None
    aux_idx = None
    if features is not None and aux_feature_name in features:
        aux_idx = features.index(aux_feature_name)
        aux_target = X[:, :, aux_idx]     # (N, 61)
    return X, y, features, soc_grid, aux_target, aux_idx


# ===== ODE pieces =====
class ODEFunc(nn.Module):
    def __init__(self, dim: int, hidden_mlp: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_mlp),
            nn.Tanh(),
            nn.Linear(hidden_mlp, hidden_mlp),
            nn.Tanh(),
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


# ===== Model =====
class ACLA_ANODE(nn.Module):
    def __init__(self, input_channels=10, conv_channels=(64, 32), lstm_hidden=64,
                 aug_dim=20, ode_hidden_mlp=128, ode_method="dopri5", t_end=1.0):
        super().__init__()

        # Attention (channel-wise gating)
        self.attn = nn.Linear(input_channels, input_channels)

        # CNN feature extractor
        c1, c2 = conv_channels
        self.conv1 = nn.Conv1d(input_channels, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=3, padding=1)

        # LSTM over time
        self.lstm = nn.LSTM(input_size=c2, hidden_size=lstm_hidden, batch_first=True)

        # ANODE
        ode_dim = lstm_hidden + aug_dim
        self.odefunc = ODEFunc(ode_dim, hidden_mlp=ode_hidden_mlp)
        self.odeblock = ODEBlock(self.odefunc, t_end=t_end, method=ode_method)

        # Readout
        self.readout = nn.Linear(lstm_hidden + aug_dim, 1)

        # Auxiliary head: reconstruct a temporal signal
        self.aux_head = nn.Conv1d(c2, 1, kernel_size=1)  # -> (B,1,61)

        for m in [self.conv1, self.conv2, self.readout]:
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, 61, C)
        attn_w = torch.sigmoid(self.attn(x))  # (B,61,C)
        x = x * attn_w

        x_cnn = x.permute(0, 2, 1)           # (B,C,61)
        x_cnn = torch.relu(self.conv1(x_cnn))
        x_cnn = torch.relu(self.conv2(x_cnn)) # (B,32,61)

        aux_seq = self.aux_head(x_cnn).squeeze(1)  # (B,61)

        x_lstm = x_cnn.permute(0, 2, 1)      # (B,61,32)
        _, (h_n, _) = self.lstm(x_lstm)
        h = h_n[-1]                           # (B,lstm_hidden)

        B = h.shape[0]
        aug_dim = self.odefunc.net[-1].out_features - h.shape[1]
        z0 = torch.cat([h, torch.zeros(B, aug_dim, device=h.device)], dim=1)
        z1 = self.odeblock(z0)

        soh = self.readout(z1).squeeze(1)     # (B,)
        return soh, aux_seq


def make_group_indices(meta: pd.DataFrame, field: str, ratios=(0.7,0.15,0.15), seed=1337):
    assert field in meta.columns, f"'{field}' không có trong META: {meta.columns.tolist()}"
    groups = meta[field].fillna("UNK").astype(str).values
    uniq = np.unique(groups)
    # shuffle groups
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)

    n = len(uniq)
    n_train = int(ratios[0] * n)
    n_val   = int(ratios[1] * n)
    grp_train = set(uniq[:n_train])
    grp_val   = set(uniq[n_train:n_train+n_val])
    grp_test  = set(uniq[n_train+n_val:])

    idx_train = np.flatnonzero(np.isin(groups, list(grp_train)))
    idx_val   = np.flatnonzero(np.isin(groups, list(grp_val)))
    idx_test  = np.flatnonzero(np.isin(groups, list(grp_test)))

    print(f"[GROUP SPLIT] field={field} | groups: total={n} "
          f"| train={len(grp_train)} | val={len(grp_val)} | test={len(grp_test)}")
    print(f"[GROUP SPLIT] samples: train={len(idx_train)} | val={len(idx_val)} | test={len(idx_test)}")
    return idx_train, idx_val, idx_test, grp_train, grp_val, grp_test


def main():
    set_seed(SEED)
    device = torch.device("cpu" if USE_CPU or not torch.cuda.is_available() else "cuda")
    npz = assert_exists(NPZ_PATH)
    meta_p = assert_exists(META_PATH)

    # Load
    X, y, features, soc_grid, aux_target, aux_idx = load_npz(str(npz), aux_feature_name=AUX_FEATURE)
    X = torch.tensor(X, dtype=torch.float32)   # (N,61,C)
    y = torch.tensor(y, dtype=torch.float32)   # (N,)

    if aux_target is not None:
        aux_target = torch.tensor(aux_target, dtype=torch.float32)  # (N,61)
        print(f"[INFO] Aux feature '{AUX_FEATURE}' found at channel {aux_idx}")
    else:
        print(f"[WARN] Aux feature '{AUX_FEATURE}' NOT found. Recommend LAMBDA_AUX=0.")

    # Load meta (must align with X rows order from preprocessing)
    meta = pd.read_csv(meta_p)
    assert len(meta) == X.shape[0], f"Meta rows ({len(meta)}) != X samples ({X.shape[0]}). Ensure meta is from same preprocessing run."

    # ---- Group split by car/user ----
    idx_train, idx_val, idx_test, grp_tr, grp_vl, grp_te = make_group_indices(meta, GROUP_FIELD, GROUP_SPLIT, SEED)

    # ---- Per-feature normalization on TRAIN only ----
    mu  = X[idx_train].mean(dim=(0,1), keepdim=True)                  # (1,1,C)
    std = X[idx_train].std(dim=(0,1), keepdim=True).clamp_min(1e-6)   # (1,1,C)
    X = (X - mu) / std

    if aux_target is not None:
        mu_a  = aux_target[idx_train].mean()
        std_a = aux_target[idx_train].std().clamp_min(1e-6)
        aux_target = (aux_target - mu_a) / std_a
    else:
        mu_a = torch.tensor(0.0); std_a = torch.tensor(1.0)

    # Datasets
    if aux_target is not None:
        train_ds = TensorDataset(X[idx_train], y[idx_train], aux_target[idx_train])
        val_ds   = TensorDataset(X[idx_val],   y[idx_val],   aux_target[idx_val])
        test_ds  = TensorDataset(X[idx_test],  y[idx_test],  aux_target[idx_test])
    else:
        train_ds = TensorDataset(X[idx_train], y[idx_train])
        val_ds   = TensorDataset(X[idx_val],   y[idx_val])
        test_ds  = TensorDataset(X[idx_test],  y[idx_test])

    def make_loader(d, shuffle=False):
        return DataLoader(d, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0, pin_memory=True)

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader   = make_loader(val_ds, shuffle=False)
    test_loader  = make_loader(test_ds, shuffle=False)

    # Model
    model = ACLA_ANODE(
        input_channels=X.shape[2],
        conv_channels=(64, 32),
        lstm_hidden=LSTM_HIDDEN,
        aug_dim=AUG_DIM,
        ode_hidden_mlp=ODE_HIDDEN,
        ode_method=ODE_METHOD,
        t_end=T_END
    ).to(device)

    # Optimizer + warmup/cosine scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = EPOCHS * max(1, len(train_loader))
    warmup_steps = max(1, int(0.1 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    mse = nn.MSELoss()

    # Train (select best by val_RMSE)
    best_val_rmse = float("inf")
    best_state = None
    patience_count = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        train_rmse_list = []

        for batch in train_loader:
            if aux_target is not None:
                xb, yb, sb = batch; sb = sb.to(device)
            else:
                xb, yb = batch; sb = None

            xb = xb.to(device); yb = yb.to(device)

            soh_pred, seq_pred = model(xb)

            loss_soh = mse(soh_pred, yb)
            if (sb is not None) and (LAMBDA_AUX > 0):
                loss_aux = mse(seq_pred, sb)
            else:
                loss_aux = torch.tensor(0.0, device=device)
            loss = loss_soh + LAMBDA_AUX * loss_aux

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            train_rmse_list.append(rmse_torch(soh_pred, yb).item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_losses = []; val_rmse_list = []
            for batch in val_loader:
                if aux_target is not None:
                    xb, yb, sb = batch; sb = sb.to(device)
                else:
                    xb, yb = batch; sb = None

                xb = xb.to(device); yb = yb.to(device)
                soh_pred, seq_pred = model(xb)

                loss_soh = mse(soh_pred, yb)
                if (sb is not None) and (LAMBDA_AUX > 0):
                    loss_aux = mse(seq_pred, sb)
                else:
                    loss_aux = torch.tensor(0.0, device=device)
                loss = loss_soh + LAMBDA_AUX * loss_aux

                val_losses.append(loss.item())
                val_rmse_list.append(rmse_torch(soh_pred, yb).item())

            val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
            val_rmse = float(np.mean(val_rmse_list)) if val_rmse_list else float("inf")
            train_rmse = float(np.mean(train_rmse_list)) if train_rmse_list else float("inf")

        print(f"Epoch {epoch:03d} | train_loss={np.mean(train_losses):.5f} | val_loss={val_loss:.5f} "
              f"| train_RMSE={train_rmse:.4f} | val_RMSE={val_rmse:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

        if val_rmse < best_val_rmse - 1e-6:
            best_val_rmse = val_rmse
            best_state = {
                "model": model.state_dict(),
                "config": {
                    "features": features,
                    "lstm_hidden": LSTM_HIDDEN,
                    "aug_dim": AUG_DIM,
                    "ode_hidden": ODE_HIDDEN,
                    "ode_method": ODE_METHOD,
                    "t_end": T_END,
                },
                "norm": {
                    "mu": mu.cpu().numpy(),
                    "std": std.cpu().numpy(),
                    "mu_aux": float(mu_a.cpu().numpy()) if torch.is_tensor(mu_a) else float(mu_a),
                    "std_aux": float(std_a.cpu().numpy()) if torch.is_tensor(std_a) else float(std_a),
                },
                "group_split": {
                    "field": GROUP_FIELD,
                    "ratios": GROUP_SPLIT,
                    "groups_train": sorted(list(grp_tr)),
                    "groups_val":   sorted(list(grp_vl)),
                    "groups_test":  sorted(list(grp_te)),
                },
                "val_rmse": best_val_rmse,
                "epoch": epoch,
            }
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"[EarlyStopping] No val_RMSE improvement for {PATIENCE} epochs. Stop.")
                break

    # Save best
    out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "acla_anode_best.pt"
    torch.save(best_state, ckpt_path)
    print(f"[SAVE] Best checkpoint -> {ckpt_path} (val_RMSE={best_val_rmse:.6f})")

    # Test
    model.load_state_dict(best_state["model"])
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            if aux_target is not None:
                xb, yb, _ = batch
            else:
                xb, yb = batch
            xb = xb.to(device)
            yh = model(xb)[0]
            y_true.append(yb.numpy())
            y_pred.append(yh.cpu().numpy())
    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
    test_rmse = rmse_np(y_true, y_pred)
    print(f"[TEST] RMSE(SOH) = {test_rmse:.4f}")


if __name__ == "__main__":
    main()
