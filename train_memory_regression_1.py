#!/usr/bin/env python
# scripts_nuevos/train_memory_regression_1.py
"""
Regresión del retorno futuro r_H(t) (o su versión vol-aware z_H) con memoria (LSTM/Transformer + attention pooling).

- Features "rich" 100% retrospectivas (sin fuga): logp, ret1, MA(1h)/p, MA(6h)/p, momentum 1h,
  vol_1h, vol_6h, componentes diarios sen/cos, y opcionalmente lambda_proxy si existe.
  Normalización (z-score) ajustada SOLO en TRAIN y aplicada tal cual en VAL/TEST usando máscara de longitudes.
- Alternativamente, modo "windows" (legacy) con [price, sigma_rolling60m] tolerante a NaNs.

Objetivo:
- target=pct  -> r_H(t) en tanto por uno
- target=zscore -> z_H = r_H / sigma_1h(t)
- std_y: opcionalmente estandariza y en TRAIN; el reporte de RMSE/MAE se hace en unidades originales (se deshace std).

Métricas reportadas:
- RMSE, MAE, Pearson r, R2, Hit-rate de signo.

Early stopping:
- --early-by {rmse, mae, pearson_r, r2} (por defecto rmse)
- Criterio corregido:
    * rmse/mae -> minimizar
    * pearson_r/r2 -> maximizar
- warmup: no promueve “best” si no mejora; solo evita disparar el early stop hasta terminar el warmup.

"""

import argparse, math, json
from pathlib import Path
from typing import Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DT = np.float32
LOG2PI = math.log(2.0 * math.pi)


# ───────── Utils ─────────

def set_seed(seed: int, deterministic: bool = False):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try: torch.use_deterministic_algorithms(True)
        except Exception: pass

def load_price_series(world: Path) -> np.ndarray:
    for name in ["SYN_PRICE.npy","SYN_SERIE_5m.npy","PRICE_5m.npy","SERIE_5m.npy"]:
        fn = world / name
        if fn.exists(): return np.load(fn).astype(np.float64)
    raise FileNotFoundError(f"No price series in {world}")

def load_step_min(world: Path, default: int = 5) -> int:
    for fn in ["SYN_META.json", "meta.json"]:
        p = world / fn
        if p.exists():
            try:
                meta = json.loads(p.read_text())
                v = int(meta.get("dt_min", meta.get("step_min", default)))
                return v if v > 0 else default
            except Exception:
                pass
    return default

def maybe_load_lambda_proxy(world: Path) -> Optional[np.ndarray]:
    for name in ["SYN_LAMBDA.npy","HAWKES_LAMBDA.npy","LAMBDA.npy"]:
        fn = world / name
        if fn.exists():
            try:
                return np.load(fn).astype(np.float64)
            except Exception:
                return None
    return None

def split_indices_purged(N: int, steps_H: int, tr=0.70, va=0.15):
    """Purged split: deja hueco de steps_H entre bloques para evitar fuga temporal."""
    n_tr = int(tr * N); n_va = int(va * N)
    i_tr_end = n_tr
    i_va_start = min(N, i_tr_end + steps_H)
    i_va_end   = min(N, i_va_start + n_va)
    i_te_start = min(N, i_va_end + steps_H)
    return slice(0, i_tr_end), slice(i_va_start, i_va_end), slice(i_te_start, N)

def compute_pct_target(price: np.ndarray, horizon_min: int, step_min: int) -> np.ndarray:
    steps = max(1, horizon_min // max(1, step_min))
    p = np.clip(price.astype(np.float64), 1e-9, None)
    rH = (p[steps:] - p[:-steps]) / p[:-steps]
    return rH.astype(np.float32)

# ───────── Features ricas ─────────

def trailing_mean(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1: return x.copy()
    c = np.cumsum(np.insert(x, 0, 0.0))
    m = (c[w:] - c[:-w]) / w
    out = np.empty_like(x, dtype=np.float64)
    out[:w-1] = x[:w-1]; out[w-1:] = m
    return out

def rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1: return np.zeros_like(x)
    c1 = np.cumsum(np.insert(x, 0, 0.0))
    c2 = np.cumsum(np.insert(x*x, 0, 0.0))
    s = (c1[w:] - c1[:-w]) / w
    s2 = (c2[w:] - c2[:-w]) / w
    var = np.maximum(0.0, s2 - s*s)
    out = np.zeros_like(x); out[w-1:] = np.sqrt(var)
    return out

def build_raw_features(price: np.ndarray, step_min: int, lambda_proxy: Optional[np.ndarray]=None) -> np.ndarray:
    p = np.clip(price.astype(np.float64), 1e-9, None)
    logp = np.log(p)
    ret1 = np.diff(logp, prepend=logp[0])

    s1h = max(1, 60 // max(1, step_min))
    s6h = max(1, (6*60) // max(1, step_min))

    ma_fast = trailing_mean(p, s1h) / p
    ma_slow = trailing_mean(p, s6h) / p
    mom_1h  = (p - np.concatenate([p[:s1h], p[:-s1h]])) / np.concatenate([p[:s1h], p[:-s1h]])
    vol_1h  = rolling_std(ret1, s1h)
    vol_6h  = rolling_std(ret1, s6h)

    N = p.size
    day_period = float((24 * 60) // max(1, step_min))
    idx = np.arange(N, dtype=np.float64)
    sin_t = np.sin(2*np.pi * (idx % day_period) / day_period)
    cos_t = np.cos(2*np.pi * (idx % day_period) / day_period)

    feats = [logp, ret1, ma_fast, ma_slow, mom_1h, vol_1h, vol_6h, sin_t, cos_t]
    if lambda_proxy is not None and lambda_proxy.shape[0] == N:
        feats.append(lambda_proxy.astype(np.float64))

    X_all = np.stack(feats, axis=1)
    return np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

def build_windows_from_matrix(X_all: np.ndarray, T: int) -> Tuple[np.ndarray, np.ndarray]:
    """Devuelve (X_windows[T], lengths). Left-pad con 0; devolvemos lengths para máscara."""
    N, F = X_all.shape
    X = np.zeros((N, T, F), dtype=DT)
    lengths = np.zeros((N,), dtype=np.int32)
    for i in range(N):
        s = i - T + 1
        if s < 0:
            X[i, -i-1:, :] = X_all[:i+1, :]
            lengths[i] = i + 1
        else:
            X[i, :, :] = X_all[s:i+1, :]
            lengths[i] = T
    return X, lengths

# ───────── Atención / Modelos ─────────

class AttnPool(nn.Module):
    """Attention pooling con máscara de padding (True = ignorar)."""
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.w = nn.Linear(d, 1)
        self.drop = nn.Dropout(dropout)
    def forward(self, z, pad_mask):
        scores = self.w(self.drop(z)).squeeze(-1)       # (B,T)
        scores = scores.masked_fill(pad_mask, -1e9)
        attn = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (z * attn).sum(dim=1)                    # (B,D)

class LSTMRegressor(nn.Module):
    def __init__(self, feat_dim=10, hidden=128, layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(feat_dim, hidden, num_layers=layers,
                            batch_first=True, dropout=(dropout if layers>1 else 0.0))
        self.attnp = AttnPool(hidden, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
    def forward(self, x, lengths, pad_mask):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        z, _ = self.lstm(packed)
        z, _ = nn.utils.rnn.pad_packed_sequence(z, batch_first=True)
        h = self.attnp(z, pad_mask)
        return self.head(h).squeeze(-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe)
    def forward(self, x): return x + self.pe[:x.size(1)].unsqueeze(0)

class TransformerRegressor(nn.Module):
    def __init__(self, feat_dim=10, d_model=128, nhead=4, ff=256, layers=3, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(feat_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.attnp = AttnPool(d_model, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
    def forward(self, x, lengths, pad_mask):
        h = self.inp(x); h = self.pos(h)
        z = self.enc(h, src_key_padding_mask=pad_mask)
        pooled = self.attnp(z, pad_mask)
        return self.head(pooled).squeeze(-1)

# ───────── Métricas ─────────

def pearsonr_np(y_true, y_pred):
    y_true = np.asarray(y_true, np.float64)
    y_pred = np.asarray(y_pred, np.float64)
    if y_true.size == 0: return 0.0
    yt = y_true - y_true.mean()
    yp = y_pred - y_pred.mean()
    denom = np.sqrt((yt**2).sum()) * np.sqrt((yp**2).sum())
    if denom <= 1e-12: return 0.0
    return float((yt*yp).sum() / denom)

def r2_np(y_true, y_pred):
    y_true = np.asarray(y_true, np.float64)
    y_pred = np.asarray(y_pred, np.float64)
    if y_true.size == 0: return 0.0
    ss_res = float(((y_true - y_pred)**2).sum())
    ss_tot = float(((y_true - y_true.mean())**2).sum()) + 1e-12
    return float(1.0 - ss_res/ss_tot)

def metrics_reg(y_true, y_pred):
    y_true = np.asarray(y_true, np.float64)
    y_pred = np.asarray(y_pred, np.float64)
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    r    = pearsonr_np(y_true, y_pred)
    r2   = r2_np(y_true, y_pred)
    hit  = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    return {"rmse": rmse, "mae": mae, "r": r, "r2": r2, "hit": hit}

# ───────── Main ─────────

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--world", required=True)
    pa.add_argument("--horizon-min", type=int, required=True)
    pa.add_argument("--features", choices=["rich","windows"], default="rich")
    pa.add_argument("--context-len", type=int, default=256)
    pa.add_argument("--use-lambda-proxy", choices=["auto","off"], default="auto")

    # Objetivo
    pa.add_argument("--target", choices=["pct","zscore"], default="zscore",
                    help="pct=r_H; zscore=z_H=r_H/sigma_1h")
    pa.add_argument("--std-y", action="store_true",
                    help="Estandariza y con media/std de TRAIN (se deshace al reportar métricas)")

    # Modelo
    pa.add_argument("--model", choices=["lstm","transformer"], default="transformer")
    pa.add_argument("--dropout", type=float, default=0.1)

    # Entrenamiento
    pa.add_argument("--epochs", type=int, default=40)
    pa.add_argument("--batch-size", type=int, default=512)
    pa.add_argument("--lr", type=float, default=3e-4)
    pa.add_argument("--weight-decay", type=float, default=1e-4)
    pa.add_argument("--grad-clip", type=float, default=1.0)

    # Pérdida
    pa.add_argument("--loss", choices=["mse","huber"], default="mse")
    pa.add_argument("--huber-delta", type=float, default=1.0)

    # Early stopping
    pa.add_argument("--early-by", choices=["rmse","mae","pearson_r","r2"], default="rmse")
    pa.add_argument("--patience", type=int, default=8)
    pa.add_argument("--min-delta", type=float, default=1e-4)
    pa.add_argument("--warmup-epochs", type=int, default=3)

    # Embargo/otros
    pa.add_argument("--embargo-steps", type=str, default="auto")
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--device", default="cuda:0")
    pa.add_argument("--deterministic", action="store_true")
    args = pa.parse_args()

    set_seed(args.seed, deterministic=args.deterministic)
    device = torch.device(args.device)
    world  = Path(args.world).expanduser().resolve()
    STEP_MIN = load_step_min(world, default=5)

    # Serie y objetivos
    price = load_price_series(world)
    r_pct_full = compute_pct_target(price, args.horizon_min, STEP_MIN)

    # sigma_1h para zscore
    logp = np.log(np.clip(price.astype(np.float64), 1e-9, None))
    ret1 = np.diff(logp, prepend=logp[0])
    steps_1h = max(1, 60 // max(1, STEP_MIN))
    sig1h = rolling_std(ret1, steps_1h)
    sig1h = sig1h[:r_pct_full.shape[0]]
    sig1h = np.clip(sig1h, 1e-8, None)

    # Construcción de X,y según features
    use_rich = (args.features == "rich")
    if use_rich:
        lam = maybe_load_lambda_proxy(world) if args.use_lambda_proxy == "auto" else None
        X_all = build_raw_features(price, STEP_MIN, lambda_proxy=lam)
        steps_H = max(1, args.horizon_min // max(1, STEP_MIN))
        N_eff = min(X_all.shape[0] - steps_H, r_pct_full.shape[0])
        if N_eff <= 0: raise SystemExit("No hay muestras tras alinear horizonte.")
        X_all = X_all[:N_eff, :]
        r_pct = r_pct_full[:N_eff]
        z_H   = (r_pct / sig1h[:N_eff]).astype(np.float32)

        X_raw, lengths = build_windows_from_matrix(X_all, T=int(args.context_len))
        Fdim = X_raw.shape[2]

        # Embargo y splits
        if args.embargo_steps.strip().lower() == "auto":
            embargo = steps_H
        else:
            embargo = max(0, int(args.embargo_steps))

        idx_tr, idx_va, idx_te = split_indices_purged(X_raw.shape[0], steps_H, 0.70, 0.15)

        # Normalización SOLO en TRAIN, ignorando padding
        Tctx = X_raw.shape[1]
        valid = np.zeros((X_raw.shape[0], Tctx), dtype=bool)
        for i, L in enumerate(lengths):
            if L > 0:
                valid[i, Tctx-int(L):] = True
        valid_tr = valid[idx_tr]
        mask3 = np.repeat(valid_tr[:, :, None], Fdim, axis=2)
        Xtr_valid = X_raw[idx_tr][mask3].reshape(-1, Fdim).astype(np.float64)
        mu = np.mean(Xtr_valid, axis=0)
        sd = np.std(Xtr_valid, axis=0) + 1e-9
        X_np = (X_raw - mu.astype(DT)) / sd.astype(DT)

        pad_mask_full = ~valid
        lengths_full  = lengths

        feat_tag = f"richT{int(args.context_len)}"
        has_lambda = (lam is not None)
    else:
        # WINDOWS (legacy): [price, sigma_rolling60m] con NaNs iniciales
        steps_H = max(1, args.horizon_min // max(1, STEP_MIN))
        X_np = np.load(world / f"SYN_WINDOWS_{args.horizon_min}m_X.npy", mmap_mode="r").astype(DT)
        N_eff = min(X_np.shape[0] - steps_H, r_pct_full.shape[0])
        if N_eff <= 0: raise SystemExit("No hay muestras tras alinear horizonte.")
        X_np = X_np[:N_eff]
        r_pct = r_pct_full[:N_eff]
        z_H   = (r_pct / sig1h[:N_eff]).astype(np.float32)

        # splits purgados
        idx_tr, idx_va, idx_te = split_indices_purged(N_eff, steps_H, 0.70, 0.15)

        # Normalización solo TRAIN ignorando NaNs
        Xtr = X_np[idx_tr]
        price_tr = Xtr[..., 0]  # puede tener NaNs
        sigma_tr = Xtr[..., 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            logp_tr = np.log(np.clip(price_tr, 1e-6, None))
        m0 = float(np.nanmean(logp_tr)); s0 = float(np.nanstd(logp_tr) + 1e-9)
        m1 = float(np.nanmean(sigma_tr)); s1 = float(np.nanstd(sigma_tr) + 1e-9)
        legacy_stats = (m0, s0, m1, s1)

        Fdim = X_np.shape[2]
        pad_mask_full = None
        lengths_full  = None
        feat_tag = "windows"
        has_lambda = False

    # y según target
    y_full = z_H if args.target == "zscore" else r_pct

    # Estandarización de y (se deshace para métricas)
    if args.std_y:
        y_tr = y_full[idx_tr]
        y_mean = float(np.mean(y_tr))
        y_std  = float(np.std(y_tr) + 1e-9)
    else:
        y_mean, y_std = 0.0, 1.0

    # Tensores
    X_t = torch.from_numpy(np.asarray(X_np, np.float32))
    y_t = torch.from_numpy(((y_full - y_mean) / y_std).astype(np.float32))
    if device.type == "cuda":
        X_t = X_t.pin_memory(); y_t = y_t.pin_memory()

    if use_rich:
        pad_mask_full_t = torch.from_numpy(pad_mask_full)
        lengths_full_t  = torch.from_numpy(lengths_full.astype(np.int64))

    # Modelo
    if args.model == "lstm":
        net = LSTMRegressor(feat_dim=Fdim, hidden=128, layers=2, dropout=args.dropout).to(device)
        model_cfg = {"input_dim": int(Fdim), "hidden_dim":128, "num_layers":2, "dropout":args.dropout}
        model_name = "LSTM"
    else:
        net = TransformerRegressor(feat_dim=Fdim, d_model=128, nhead=4, ff=256, layers=3, dropout=args.dropout).to(device)
        model_cfg = {"input_dim": int(Fdim), "d_model":128, "nhead":4, "dim_feedforward":256, "num_layers":3, "dropout":args.dropout}
        model_name = "TRANSFORMER"

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    outdir = (world / "trained").resolve(); outdir.mkdir(parents=True, exist_ok=True)
    ckpt = outdir / f"REG_{model_name}_{feat_tag}_{args.target}_stdY{int(args.std_y)}_H{int(args.horizon_min)}m_{world.name}.pt"

    print(f"\n=== Train REG ({args.model}) world={world.name} H={int(args.horizon_min)}m ===")
    print(f"Device={device} | step_min={STEP_MIN} | features={args.features} | F={Fdim} | target={args.target} | std_y={args.std_y}")
    if use_rich:
        print(f"Rich feats: T={int(args.context_len)} | lambda_proxy={'ON' if has_lambda else 'OFF'}")

    BS = args.batch_size

    def huber_loss(pred, target, delta=1.0):
        err = pred - target
        abs_e = torch.abs(err)
        quad = torch.minimum(abs_e, torch.tensor(delta, device=err.device))
        lin  = abs_e - quad
        return (0.5 * quad**2 + delta * lin).mean()

    @torch.no_grad()
    def eval_slice(sl: slice):
        net.eval()
        start = sl.start or 0; stop = sl.stop if sl.stop is not None else X_t.size(0)
        preds = []; gts = []
        for i in range(start, stop, BS):
            end = min(i+BS, stop)
            xb = X_t[i:end]
            if use_rich:
                pad_b = pad_mask_full_t[i:end]
                len_b = lengths_full_t[i:end]
                x = xb.to(device, non_blocking=True)
                pad_mask = pad_b.to(device, non_blocking=True)
                lengths  = len_b.to(device, non_blocking=True)
            else:
                # windows: re-normaliza robustamente (igual que en train)
                pad_mask = torch.isnan(xb[..., 0])
                x0 = xb[..., 0].clone()
                val0 = ~torch.isnan(x0)
                x0[val0] = torch.log(x0[val0].clamp_min(1e-6))
                m0, s0, m1, s1 = legacy_stats
                x0[val0] = (x0[val0] - m0) / s0
                x1 = xb[..., 1].clone()
                val1 = ~torch.isnan(x1)
                x1[val1] = (x1[val1] - m1) / s1
                x = torch.stack([x0, x1], dim=-1)
                x = torch.nan_to_num(x, nan=0.0).to(device, non_blocking=True)
                lengths = (~pad_mask).sum(dim=1).to(device)

            pred = net(x, lengths, pad_mask).detach().cpu().numpy()
            yb = y_t[i:end].cpu().numpy()
            preds.append(pred); gts.append(yb)
        if not preds:
            return {"rmse":0.0,"mae":0.0,"r":0.0,"r2":0.0,"hit":0.0}
        y_pred = (np.concatenate(preds) * y_std) + y_mean
        y_true = (np.concatenate(gts)   * y_std) + y_mean
        return metrics_reg(y_true, y_pred)

    # Early-stopping helpers (corregido)
    def score_from_metrics(m):
        return m["rmse"] if args.early_by=="rmse" else (
               m["mae"]  if args.early_by=="mae"  else (
               m["r"]    if args.early_by=="pearson_r" else m["r2"]))

    def better(old, new):
        # old: best_val_score; new: current_val_score
        if old is None: return True
        scale = max(1.0, abs(old))
        if args.early_by in ("rmse","mae"):       # minimizar
            return (old - new) > args.min_delta * scale
        else:                                      # maximizar
            return (new - old) > args.min_delta * scale

    best_val_score = None
    best_ep = 0
    bad = 0

    # Slices
    start_tr, stop_tr = (idx_tr.start or 0), (idx_tr.stop if idx_tr.stop is not None else X_t.size(0))

    for ep in range(1, args.epochs+1):
        net.train()
        # barajar TRAIN
        perm = torch.randperm(stop_tr - start_tr) + start_tr
        total_loss = 0.0; nb = 0

        for off in range(0, perm.numel(), BS):
            b_idx = perm[off: off+BS]
            xb = X_t[b_idx]
            yb = y_t[b_idx].to(device, non_blocking=True)

            if use_rich:
                pad_b = pad_mask_full_t[b_idx]
                len_b = lengths_full_t[b_idx]
                x = xb.to(device, non_blocking=True)
                pad_mask = pad_b.to(device, non_blocking=True)
                lengths = len_b.to(device, non_blocking=True)
            else:
                pad_mask = torch.isnan(xb[..., 0])
                x0 = xb[..., 0].clone()
                val0 = ~torch.isnan(x0)
                x0[val0] = torch.log(x0[val0].clamp_min(1e-6))
                m0, s0, m1, s1 = legacy_stats
                x0[val0] = (x0[val0] - m0) / s0
                x1 = xb[..., 1].clone()
                val1 = ~torch.isnan(x1)
                x1[val1] = (x1[val1] - m1) / s1
                x = torch.stack([x0, x1], dim=-1)
                x = torch.nan_to_num(x, nan=0.0).to(device, non_blocking=True)
                lengths = (~pad_mask).sum(dim=1).to(device)

            y_hat = net(x, lengths, pad_mask)
            if args.loss == "mse":
                loss = F.mse_loss(y_hat, yb)
            else:
                loss = huber_loss(y_hat, yb, delta=args.huber_delta)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            opt.step()
            total_loss += float(loss.detach().cpu()); nb += 1

        # Eval
        tr = eval_slice(idx_tr)
        va = eval_slice(idx_va)
        te = eval_slice(idx_te)

        print(f"[Ep{ep:02d}] "
              f"Tr rmse={tr['rmse']:.4g} r={tr['r']:.3f} | "
              f"Va rmse={va['rmse']:.4g} r={va['r']:.3f} | "
              f"Te rmse={te['rmse']:.4g} r={te['r']:.3f} | "
              f"loss={total_loss/max(1,nb):.4f}")

        cur = score_from_metrics(va)
        improved = better(best_val_score, cur)

        if improved:
            best_val_score = cur; best_ep = ep; bad = 0
            torch.save({
                "state_dict": net.state_dict(),
                "arch": args.model, "model_cfg": model_cfg,
                "horizon_min": int(args.horizon_min),
                "variant": f"reg_{args.features}",
                "world": world.name,
                "feature_mode": args.features,
                "feature_norm": ({"mu": mu.astype(np.float32), "sd": sd.astype(np.float32)} if use_rich else {
                    "legacy_logp_mean": float(legacy_stats[0]), "legacy_logp_std": float(legacy_stats[1]),
                    "legacy_sigma_mean": float(legacy_stats[2]), "legacy_sigma_std": float(legacy_stats[3])
                }),
                "target": args.target,
                "std_y": bool(args.std_y),
                "y_mean": float(y_mean),
                "y_std": float(y_std),
                "train": tr, "val": va, "test": te,
                "context_len": (int(args.context_len) if use_rich else None)
            }, ckpt)
        else:
            bad += 1
            # durante warmup no paramos aunque no mejore
            if (ep > args.warmup_epochs) and (bad >= args.patience):
                print(f"Early stopping Ep{ep} (best Va {args.early_by}={best_val_score:.4g} @Ep{best_ep})")
                break

    print(f"\n✔ Checkpoint guardado: {ckpt}")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

