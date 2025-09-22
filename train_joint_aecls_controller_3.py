#!/usr/bin/env python
# scripts_nuevos/train_joint_aecls_controller_3.py
"""
Entrenamiento conjunto: AE(LSTM) + Clasificador (LSTM/Transformer) + Policy (MLP/Transformer)

Estabilidad y depuración:
- Dataset SIN imputar a 0 ni forward-fill plano: interpolación de huecos cortos en log-precio,
  segmentación, y ventanas solo dentro de segmentos válidos.
- Entrenamiento por fases:
  * Warmup: solo AE+CLS (policy congelada), sin trading.
  * Rampa: el peso del término de trading crece gradualmente hasta 1.0.
- LR warmup (LinearLR) durante warmup.
- Guardias anti-NaN/Inf en loss y gradientes. Conteo de batches saltados.
"""

import argparse, math, json
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DT = np.float32

# ───────── Utils ─────────

def set_seed(seed: int, deterministic: bool=False):
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
        fn = world/name
        if fn.exists(): return np.load(fn).astype(np.float64)
    raise FileNotFoundError(f"No se encontró serie en {world}")

def load_step_min(world: Path, default=5) -> int:
    for fn in ["SYN_META.json","meta.json"]:
        p = world/fn
        if p.exists():
            try:
                meta = json.loads(p.read_text())
                v = int(meta.get("step_min", meta.get("dt_min", default)))
                return v if v>0 else default
            except Exception:
                return default
    return default

def maybe_load_lambda_proxy(world: Path) -> Optional[np.ndarray]:
    for name in ["SYN_LAMBDA.npy","HAWKES_LAMBDA.npy","LAMBDA.npy"]:
        fn = world/name
        if fn.exists():
            try:
                return np.load(fn).astype(np.float64)
            except Exception:
                return None
    return None

def split_indices(N: int, tr=0.70, va=0.15):
    n_tr = int(tr*N); n_va = int(va*N)
    return slice(0,n_tr), slice(n_tr,n_tr+n_va), slice(n_tr+n_va,None)

# ───────── Interpolación y segmentación ─────────

def _nan_runs(mask_nan: np.ndarray) -> List[Tuple[int,int]]:
    runs = []
    n = mask_nan.size
    i = 0
    while i < n:
        if mask_nan[i]:
            j = i+1
            while j < n and mask_nan[j]: j += 1
            runs.append((i,j))
            i = j
        else:
            i += 1
    return runs

def interp_small_gaps_logprice(p: np.ndarray, max_gap_steps: int) -> np.ndarray:
    x = p.astype(np.float64).copy()
    x[x<=0] = np.nan
    logx = np.log(x)
    isnan = ~np.isfinite(logx)
    runs = _nan_runs(isnan)
    for i, j in runs:
        L = j - i
        if L <= max_gap_steps:
            left = i-1; right = j
            if left >= 0 and right < logx.size and np.isfinite(logx[left]) and np.isfinite(logx[right]):
                t = np.linspace(0.0, 1.0, L+2)[1:-1]
                logx[i:j] = (1.0 - t)*logx[left] + t*logx[right]
    out = np.exp(logx)
    return out

def segments_from_valid(valid: np.ndarray) -> List[Tuple[int,int]]:
    segs = []
    n = valid.size
    i = 0
    while i < n:
        if valid[i]:
            s = i
            i += 1
            while i < n and valid[i]: i += 1
            e = i-1
            segs.append((s,e))
        else:
            i += 1
    return segs

# ───────── Features (segmentadas) ─────────

def trailing_mean_seg(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1: return x.copy()
    c = np.cumsum(np.insert(x,0,0.0))
    m = (c[w:] - c[:-w]) / w
    out = np.empty_like(x, dtype=np.float64)
    out[:w-1] = x[:w-1]
    out[w-1:] = m
    return out

def rolling_std_seg(x: np.ndarray, w: int) -> np.ndarray:
    if w<=1: return np.zeros_like(x)
    c1 = np.cumsum(np.insert(x,0,0.0))
    c2 = np.cumsum(np.insert(x*x,0,0.0))
    s = (c1[w:] - c1[:-w]) / w
    s2 = (c2[w:] - c2[:-w]) / w
    var = np.maximum(0.0, s2 - s*s)
    out = np.zeros_like(x); out[w-1:] = np.sqrt(var)
    return out

def build_features_segmented(price_finite: np.ndarray, valid_mask: np.ndarray, step_min: int,
                             lambda_proxy: Optional[np.ndarray]=None) -> np.ndarray:
    N = price_finite.size
    F = 9 + (1 if (lambda_proxy is not None and lambda_proxy.shape[0]==N) else 0)
    X_all = np.full((N, F), np.nan, dtype=np.float64)

    s1h = max(1, 60 // max(1, step_min))
    s6h = max(1, 360 // max(1, step_min))
    day_period = float((24*60)//max(1,step_min))
    idx = np.arange(N, dtype=np.float64)

    segs = segments_from_valid(valid_mask)
    for (s,e) in segs:
        p = price_finite[s:e+1]
        logp = np.log(p)
        ret1 = np.diff(logp, prepend=logp[0])

        ma_fast = trailing_mean_seg(p, s1h) / p
        ma_slow = trailing_mean_seg(p, s6h) / p

        lag = np.concatenate([p[:s1h], p[:-s1h]])
        with np.errstate(divide='ignore', invalid='ignore'):
            mom_1h_seg = (p - lag) / lag
            mom_1h_seg[~np.isfinite(mom_1h_seg)] = 0.0

        vol_1h = rolling_std_seg(ret1, s1h)
        vol_6h = rolling_std_seg(ret1, s6h)
        sin_t = np.sin(2*np.pi * (idx[s:e+1] % day_period) / day_period)
        cos_t = np.cos(2*np.pi * (idx[s:e+1] % day_period) / day_period)

        cols = [logp, ret1, ma_fast, ma_slow, mom_1h_seg, vol_1h, vol_6h, sin_t, cos_t]
        if lambda_proxy is not None and lambda_proxy.shape[0]==N:
            lam_seg = lambda_proxy[s:e+1]
            cols.append(lam_seg)

        X_all[s:e+1, :] = np.stack(cols, axis=1)

    return X_all

# ───────── Dataset constructor ─────────

def build_dataset_no_impute(price_raw: np.ndarray,
                            step_min: int,
                            horizon_min: int,
                            window_len: int,
                            lambda_proxy: Optional[np.ndarray]=None,
                            max_gap_minutes: int = 60) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    N = int(price_raw.size)
    T = int(window_len)
    step = max(1, int(step_min))
    steps_H = max(1, int(horizon_min // step))
    max_gap_steps = max(1, int(round(max_gap_minutes / step)))

    p_interp = interp_small_gaps_logprice(price_raw, max_gap_steps=max_gap_steps)
    valid_price = np.isfinite(p_interp) & (p_interp > 0)

    lam_use = None
    if lambda_proxy is not None and lambda_proxy.shape[0] == N:
        lam_interp = lambda_proxy.astype(np.float64).copy()
        lam_isnan = ~np.isfinite(lam_interp)
        runs = _nan_runs(lam_isnan)
        for i,j in runs:
            L = j - i
            if L <= max_gap_steps:
                left = i-1; right = j
                if left>=0 and right<N and np.isfinite(lam_interp[left]) and np.isfinite(lam_interp[right]):
                    t = np.linspace(0.0, 1.0, L+2)[1:-1]
                    lam_interp[i:j] = (1.0 - t)*lam_interp[left] + t*lam_interp[right]
        lam_use = lam_interp

    segs = segments_from_valid(valid_price)
    X_all = build_features_segmented(p_interp, valid_price, step_min, lambda_proxy=lam_use)

    rH_full = np.full(N, np.nan, dtype=np.float64)
    for (s,e) in segs:
        last = e - steps_H
        if last < s: 
            continue
        p = p_interp[s:e+1]
        rH_full[s:last+1] = (p[steps_H:] - p[:-steps_H]) / p[:-steps_H]

    idx_t = []
    for (s,e) in segs:
        start_t = s + (T - 1)
        end_t   = e - steps_H
        if end_t < start_t:
            continue
        cand = np.arange(start_t, end_t+1, dtype=np.int64)
        for t in cand:
            win_slice = slice(t - (T - 1), t + 1)
            Xw = X_all[win_slice]
            if not np.isfinite(Xw).all():
                continue
            if lam_use is not None:
                if (not np.isfinite(lam_use[win_slice]).all()) or (not np.isfinite(lam_use[t + steps_H])):
                    continue
            if not np.isfinite(rH_full[t]):
                continue
            idx_t.append(int(t))

    if len(idx_t) == 0:
        raise RuntimeError("No hay ventanas válidas tras filtrar segmentos. Ajusta max_gap_minutes o T/H.")

    idx_t = np.asarray(idx_t, dtype=np.int64)
    Fdim = X_all.shape[1]
    M = idx_t.size
    Xw = np.zeros((M, T, Fdim), dtype=DT)
    for i, t in enumerate(idx_t):
        Xw[i] = X_all[t - (T - 1): t + 1, :].astype(DT)
    y  = rH_full[idx_t].astype(DT)

    idx_tr, idx_va, idx_te = split_indices(M, 0.70, 0.15)
    Xtr_flat = Xw[idx_tr].reshape(-1, Fdim).astype(np.float64)
    mu = np.mean(Xtr_flat, axis=0)
    sd = np.std(Xtr_flat, axis=0) + 1e-9
    Xn = (Xw - mu.astype(DT)) / sd.astype(DT)

    norm = {
        "mu": mu.astype(np.float32).tolist(),
        "sd": sd.astype(np.float32).tolist(),
        "features": ["logp","ret1","ma1h/p","ma6h/p","mom_1h","vol_1h","vol_6h","sin_t","cos_t"] + (["lambda_proxy"] if (lam_use is not None) else []),
        "max_gap_minutes": max_gap_minutes
    }

    lengths = np.full(M, T, dtype=np.int64)
    return Xn.astype(DT), y.astype(DT), lengths.astype(np.int64), norm

# ───────── Modelos ─────────

class LSTMAutoencoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int=128, layers: int=2,
                 latent_dim: int=64, dropout: float=0.1, aux_head: bool=False):
        super().__init__()
        self.encoder = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True,
                               dropout=(dropout if layers>1 else 0.0))
        self.to_latent = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, latent_dim), nn.Tanh())
        self.decoder = nn.LSTM(in_dim+latent_dim, hidden, num_layers=layers, batch_first=True,
                               dropout=(dropout if layers>1 else 0.0))
        self.dec_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, in_dim))
        self.aux = (nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, latent_dim),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(latent_dim, 1))
                    if aux_head else None)

    def encode(self, x):
        _, (hN, _) = self.encoder(x)
        h = hN[-1]
        return self.to_latent(h)

    def forward(self, x):
        z = self.encode(x)
        B,T,F = x.size()
        zt = z.unsqueeze(1).repeat(1,T,1)
        y,_ = self.decoder(torch.cat([x, zt], dim=-1))
        x_hat = self.dec_out(y)
        r_hat = self.aux(z).squeeze(-1) if self.aux is not None else None
        return z, x_hat, r_hat

class AttnPool(nn.Module):
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.w = nn.Linear(d, 1)
        self.drop = nn.Dropout(dropout)
    def forward(self, z, pad_mask):
        scores = self.w(self.drop(z)).squeeze(-1)
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask, -1e9)
        attn = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (z * attn).sum(dim=1)

class LSTMClassifier(nn.Module):
    def __init__(self, feat_dim=10, hidden=128, layers=2, dropout=0.1, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(feat_dim, hidden, num_layers=layers,
                            batch_first=True, dropout=(dropout if layers>1 else 0.0))
        self.attnp = AttnPool(hidden, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x, lengths, pad_mask):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        z, _ = self.lstm(packed)
        z, _ = nn.utils.rnn.pad_packed_sequence(z, batch_first=True, total_length=x.size(1))
        h = self.attnp(z, pad_mask)
        logits = self.head(h)
        return logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe)
    def forward(self, x): return x + self.pe[:x.size(1)].unsqueeze(0)

class TransformerClassifier(nn.Module):
    def __init__(self, feat_dim=10, d_model=128, nhead=4, ff=256, layers=3, dropout=0.1, num_classes=3):
        super().__init__()
        self.inp = nn.Linear(feat_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout,
                                               batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.attnp = AttnPool(d_model, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
    def forward(self, x, lengths, pad_mask):
        h = self.inp(x)
        h = self.pos(h)
        z = self.enc(h, src_key_padding_mask=pad_mask)
        pooled = self.attnp(z, pad_mask)
        logits = self.head(pooled)
        return logits

class PolicyMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int=128, dropout: float=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, x): return torch.tanh(self.net(x)).squeeze(-1)

class PolicyTransformer(nn.Module):
    def __init__(self, in_dim: int, d_model: int=192, nhead: int=4, ff: int=384,
                 layers: int=2, tok_dim: int=32, dropout: float=0.05):
        super().__init__()
        self.tok_dim = int(tok_dim)
        self.proj = nn.Linear(self.tok_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout,
                                               batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def _tokenize(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        T = int((D + self.tok_dim - 1) // self.tok_dim)
        pad = T*self.tok_dim - D
        if pad > 0:
            x = F.pad(x, (0, pad))
        return x.view(B, T, self.tok_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tok = self._tokenize(x)
        h = self.proj(tok)
        cls = self.cls.expand(h.size(0), -1, -1)
        z = torch.cat([cls, h], dim=1)
        z = self.enc(z)
        h_cls = z[:, 0, :]
        y = self.head(h_cls)
        return torch.tanh(y).squeeze(-1)

# ───────── Trading + Métricas ─────────

def trading_loss(positions, r_future, *,
                 loss_type="sharpe",
                 lambda_var=10.0,
                 tx_cost=0.0,
                 turnover_coef=0.0,
                 l2_pos=0.0,
                 vol_norm=False,
                 expo_target=None, expo_w=0.0,
                 mean_pos_target=0.0, mean_pos_w=0.0):
    a = positions
    if vol_norm:
        sd_r = r_future.std().detach().clamp_min(1e-6)
        a = a / sd_r
    pnl = a * r_future
    mu = pnl.mean()
    var = pnl.var(unbiased=False)
    sd = var.clamp_min(1e-12).sqrt()  # clamp antes de sqrt para Sharpe estable
    base_obj = (mu/sd) if loss_type=="sharpe" else (mu - lambda_var*var)

    if a.numel()>1:
        da = a[1:] - a[:-1]
        turnover = da.abs().mean()
        cost = tx_cost * da.abs().mean()
    else:
        turnover = a.new_tensor(0.0); cost = a.new_tensor(0.0)

    reg_l2 = l2_pos * (a.pow(2).mean())
    reg_expo = a.new_tensor(0.0)
    if (expo_target is not None) and (expo_w>0.0):
        reg_expo = expo_w * (a.abs().mean() - float(expo_target))**2
    reg_mean = a.new_tensor(0.0)
    if mean_pos_w>0.0:
        reg_mean = mean_pos_w * (a.mean() - float(mean_pos_target))**2

    objective = base_obj - cost - turnover_coef*turnover - reg_l2 - reg_expo - reg_mean
    loss = -objective
    return loss, {
        "mu": float(mu.detach().cpu()),
        "sd": float(sd.detach().cpu()),
        "sharpe_like": float((mu/sd).detach().cpu()),
        "turnover": float(turnover.detach().cpu()),
        "cost": float(cost.detach().cpu()),
        "a_mean": float(a.mean().detach().cpu()),
        "a_abs_mean": float(a.abs().mean().detach().cpu()),
    }

def metrics_cls_np(y_true, y_pred, C=3):
    y_true = np.asarray(y_true, np.int64); y_pred = np.asarray(y_pred, np.int64)
    cm = np.zeros((C,C), dtype=np.int64)
    for t,p in zip(y_true, y_pred):
        if 0<=t<C and 0<=p<C: cm[t,p]+=1
    acc = float(np.trace(cm)/max(1,cm.sum()))
    rec = []
    for c in range(C):
        tp = float(cm[c,c]); fn = float(cm[c,:].sum()-cm[c,c])
        r = tp/max(1.0,tp+fn); rec.append(r)
    bal_acc = float(np.mean(rec))
    return {"acc":acc, "bal_acc":bal_acc}

def labelize_vec(r_pct: np.ndarray, tau: float) -> np.ndarray:
    y = np.zeros_like(r_pct, dtype=np.int64)
    y[r_pct >  tau] = 2
    y[np.abs(r_pct) <= tau] = 1
    y[r_pct < -tau] = 0
    return y

# ───────── Helpers eval ─────────

def parse_vec(s: str) -> List[float]:
    toks = [t.strip() for t in s.split(",") if t.strip()!=""]
    return [float(t) for t in toks]

def eval_cls_split(model, X, y_c, lengths_t, pad_mask_t, device, sl: slice, BS: int, N: int):
    model.eval()
    start = sl.start or 0; stop = sl.stop if sl.stop is not None else N
    preds, ys = [], []
    with torch.no_grad():
        for i in range(start, stop, BS):
            end = min(i+BS, stop)
            xb = X[i:end].to(device, non_blocking=True)
            len_b = lengths_t[i:end].to(device, non_blocking=True)
            pad_b = pad_mask_t[i:end].to(device, non_blocking=True) if pad_mask_t is not None else None
            logits = model(xb, len_b, pad_b)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            ys.append(y_c[i:end].cpu().numpy())
    yhat = np.concatenate(preds) if preds else np.zeros(0, dtype=np.int64)
    ygt  = np.concatenate(ys) if ys else np.zeros(0, dtype=np.int64)
    return metrics_cls_np(ygt, yhat)

def eval_trading_split_np(positions_all: np.ndarray, y_all: np.ndarray) -> Dict[str,float]:
    n = min(positions_all.size, y_all.size)
    if n == 0:
        return {"mu":0.0, "sd":1.0, "sharpe_like":0.0, "hit_rate":0.0}
    a = positions_all[:n]; r = y_all[:n]
    pnl = a * r
    mu = pnl.mean(); sd = pnl.std() + 1e-12
    S = float(mu/sd)
    hit = float(np.mean(np.sign(a)==np.sign(r)))
    return {"mu": float(mu), "sd": float(sd), "sharpe_like": S, "hit_rate": hit}

# ───────── Main ─────────

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--world", required=True)
    pa.add_argument("--horizon-min", type=int, default=60)
    pa.add_argument("--window-len", type=int, default=256)
    pa.add_argument("--use-lambda-proxy", choices=["auto","off"], default="auto")
    pa.add_argument("--max-gap-minutes", type=int, default=60, help="Interpola huecos cortos (en log-precio) <= este valor; mayores cortan segmentos")

    # AE
    pa.add_argument("--ae-latent", type=int, default=64)
    pa.add_argument("--ae-hidden", type=int, default=128)
    pa.add_argument("--ae-layers", type=int, default=2)
    pa.add_argument("--ae-dropout", type=float, default=0.1)
    pa.add_argument("--ae-aux-weight", type=float, default=0.0)

    # CLS
    pa.add_argument("--cls-arch", choices=["lstm","transformer"], default="transformer")
    pa.add_argument("--cls-dropout", type=float, default=0.1)
    pa.add_argument("--cls-loss", choices=["ce","focal"], default="ce")
    pa.add_argument("--cls-gamma", type=float, default=1.0)
    pa.add_argument("--cls-focal-alpha", type=str, default="0.3,0.5,0.3")
    pa.add_argument("--q-tau", type=float, default=0.5)
    pa.add_argument("--neutral-weight-mult", type=float, default=1.0)
    pa.add_argument("--class-weights", type=str, default="auto")

    # Policy
    pa.add_argument("--policy", choices=["mlp","transformer"], default="transformer")
    pa.add_argument("--pol-hidden", type=int, default=192)
    pa.add_argument("--pol-dropout", type=float, default=0.05)
    pa.add_argument("--pol-d-model", type=int, default=192)
    pa.add_argument("--pol-nhead", type=int, default=4)
    pa.add_argument("--pol-ff", type=int, default=384)
    pa.add_argument("--pol-layers", type=int, default=2)
    pa.add_argument("--pol-tok-dim", type=int, default=32)

    # Objetivo trading
    pa.add_argument("--loss-type", choices=["meanvar","sharpe"], default="sharpe")
    pa.add_argument("--lambda-var", type=float, default=10.0)
    pa.add_argument("--tx-cost", type=float, default=1e-4)
    pa.add_argument("--turnover-coef", type=float, default=0.05)
    pa.add_argument("--l2-pos", type=float, default=0.0)
    pa.add_argument("--vol-norm", action="store_true")
    pa.add_argument("--expo-target", type=float, default=None)
    pa.add_argument("--expo-w", type=float, default=0.0)
    pa.add_argument("--mean-pos-target", type=float, default=0.0)
    pa.add_argument("--mean-pos-w", type=float, default=0.0)

    # Entrenamiento
    pa.add_argument("--epochs", type=int, default=40)
    pa.add_argument("--batch-size", type=int, default=512)
    pa.add_argument("--lr", type=float, default=3e-4)
    pa.add_argument("--weight-decay", type=float, default=1e-4)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--device", default="cuda:0")
    pa.add_argument("--deterministic", action="store_true")
    pa.add_argument("--patience", type=int, default=8)
    pa.add_argument("--min-delta", type=float, default=1e-4)
    pa.add_argument("--warmup-epochs", type=int, default=3, help="N épocas entrenando solo AE+CLS (policy congelada, trading weight=0)")
    pa.add_argument("--tr-ramp-epochs", type=int, default=5, help="Épocas para ramp-up lineal del peso de trading hasta 1.0")

    # Pesos de pérdidas
    pa.add_argument("--rec-weight", type=float, default=1.0, help="Peso reconstrucción AE")
    pa.add_argument("--cls-weight", type=float, default=0.5, help="Peso clasificación CLS")

    args = pa.parse_args()

    set_seed(args.seed, deterministic=args.deterministic)
    device = torch.device(args.device)
    world = Path(args.world).expanduser().resolve()

    STEP_MIN = load_step_min(world, default=5)
    price_raw = load_price_series(world)
    lam = maybe_load_lambda_proxy(world) if args.use_lambda_proxy=="auto" else None

    # Dataset
    X_np, rH_np, lengths_np, norm = build_dataset_no_impute(
        price_raw, step_min=STEP_MIN,
        horizon_min=args.horizon_min,
        window_len=args.window_len,
        lambda_proxy=lam,
        max_gap_minutes=args.max_gap_minutes
    )

    N, T, Fdim = X_np.shape
    idx_tr, idx_va, idx_te = split_indices(N, 0.70, 0.15)

    X = torch.from_numpy(X_np.astype(DT))
    y_r = torch.from_numpy(rH_np.astype(DT))
    if device.type=="cuda":
        X = X.pin_memory(); y_r = y_r.pin_memory()

    # Etiquetas 3 clases (tau en TRAIN)
    tau = float(np.quantile(np.abs(rH_np[idx_tr]), args.q_tau)) if idx_tr.stop>0 else float(np.quantile(np.abs(rH_np), args.q_tau))
    y_cls_np = labelize_vec(rH_np, tau)
    y_c = torch.from_numpy(y_cls_np.astype(np.int64))
    if device.type=="cuda":
        y_c = y_c.pin_memory()

    # Sin padding real
    pad_mask_np = np.zeros((N, T), dtype=bool)
    lengths_t = torch.from_numpy(lengths_np.astype(np.int64))
    pad_mask_t = torch.from_numpy(pad_mask_np)

    # Modelos
    ae = LSTMAutoencoder(in_dim=Fdim, hidden=args.ae_hidden, layers=args.ae_layers,
                         latent_dim=args.ae_latent, dropout=args.ae_dropout,
                         aux_head=(args.ae_aux_weight>0.0)).to(device)

    if args.cls_arch=="lstm":
        cls = LSTMClassifier(feat_dim=Fdim, hidden=128, layers=2,
                             dropout=args.cls_dropout, num_classes=3).to(device)
    else:
        cls = TransformerClassifier(feat_dim=Fdim, d_model=128, nhead=4, ff=256, layers=3,
                                    dropout=args.cls_dropout, num_classes=3).to(device)

    pol_in = int(args.ae_latent + 3)
    if args.policy == "transformer":
        nhead = int(args.pol_nhead)
        if args.pol_d_model % nhead != 0:
            print(f"[WARN] pol-d-model={args.pol_d_model} no divisible por nhead={nhead}. Ajusto nhead=1.")
            nhead = 1
        pol = PolicyTransformer(
            in_dim=pol_in,
            d_model=int(args.pol_d_model),
            nhead=nhead,
            ff=int(args.pol_ff),
            layers=int(args.pol_layers),
            tok_dim=int(args.pol_tok_dim),
            dropout=float(args.pol_dropout) if hasattr(args, "pol_dropout") else float(args.pol_dropout) if False else float(args.pol_dropout) if False else float(args.pol_dropout)  # safe fallback
        ).to(device)
    else:
        pol = PolicyMLP(in_dim=pol_in, hidden=args.pol_hidden, dropout=args.pol_dropout if hasattr(args,"pol_dropout") else args.pol_dropout if False else args.pol_dropout if False else args.pol_dropout).to(device)  # same fallback


    del pol  # reconstruimos limpio con los nombres estándar
    if args.policy == "transformer":
        nhead = int(args.pol_nhead)
        if args.pol_d_model % nhead != 0:
            print(f"[WARN] pol-d-model={args.pol_d_model} no divisible por nhead={nhead}. Ajusto nhead=1.")
            nhead = 1
        pol = PolicyTransformer(
            in_dim=pol_in,
            d_model=int(args.pol_d_model),
            nhead=nhead,
            ff=int(args.pol_ff),
            layers=int(args.pol_layers),
            tok_dim=int(args.pol_tok_dim),
            dropout=float(args.pol_dropout) if hasattr(args,"pol_dropout") else float(args.pol_dropout)
        ).to(device)
    else:
        pol = PolicyMLP(in_dim=pol_in, hidden=args.pol_hidden,
                        dropout=args.pol_dropout if hasattr(args,"pol_dropout") else args.pol_dropout).to(device)

    # Optimizador y scheduler (LR warmup)
    params = list(ae.parameters()) + list(cls.parameters()) + list(pol.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = None
    if args.warmup_epochs > 0:
        # escala lineal desde 20% del LR hasta 100% durante warmup_epochs
        sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.2, total_iters=max(1, args.warmup_epochs))

    # Pesos de clase
    cw_arg = args.class_weights.strip().lower()
    if cw_arg=="auto":
        freqs = np.bincount(y_cls_np[idx_tr], minlength=3).astype(np.float64)
        inv = 1.0 / np.clip(freqs, 1.0, None)
        inv = inv / inv.mean()
        inv[1] *= float(args.neutral_weight_mult)
        class_weights = torch.tensor(inv, dtype=torch.float32, device=device)
    else:
        wlist = parse_vec(args.class_weights); assert len(wlist)==3
        class_weights = torch.tensor(wlist, dtype=torch.float32, device=device)

    alpha_vec = None
    if args.cls_loss=="focal" and args.cls_focal_alpha.strip():
        al = parse_vec(args.cls_focal_alpha); assert len(al)==3
        alpha_vec = torch.tensor(al, dtype=torch.float32, device=device)

    def cls_loss_fn(logits, yb):
        if args.cls_loss=="focal":
            logp = F.log_softmax(logits, dim=1)
            p = torch.exp(logp)
            pt = p.gather(1, yb.view(-1,1)).squeeze(1)
            logpt = logp.gather(1, yb.view(-1,1)).squeeze(1)
            w = class_weights.gather(0, yb)
            if alpha_vec is not None: w = w * alpha_vec.gather(0, yb)
            return (- w * ((1-pt)**args.cls_gamma) * logpt).mean()
        else:
            return F.cross_entropy(logits, yb, weight=class_weights)

    def set_policy_requires_grad(enable: bool):
        for p in pol.parameters():
            p.requires_grad = enable

    def run_epoch(mode: str, sl: slice, ep: int, tr_weight: float):
        # política congelada durante warmup
        if ep <= args.warmup_epochs:
            set_policy_requires_grad(False)
        else:
            set_policy_requires_grad(True)

        ae.train(mode=="train"); cls.train(mode=="train"); pol.train(mode=="train")
        start = sl.start or 0; stop = sl.stop if sl.stop is not None else N

        BS = args.batch_size
        nb = 0
        skipped_loss = 0
        skipped_badgrad = 0
        agg = {"loss":0.0,"tr_mu":0.0,"tr_sd":0.0,"tr_S":0.0,"cls_acc":0.0,"cls_bal":0.0,
               "ae_rec":0.0,"ae_aux":0.0,"turnover":0.0,"cost":0.0}
        pos_all = []

        with torch.set_grad_enabled(mode=="train"):
            for i in range(start, stop, BS):
                end = min(i+BS, stop)
                xb = X[i:end].to(device, non_blocking=True)
                yb_r = y_r[i:end].to(device, non_blocking=True)
                yb_c = y_c[i:end].to(device, non_blocking=True)
                len_b = lengths_t[i:end].to(device, non_blocking=True)
                pad_b = pad_mask_t[i:end].to(device, non_blocking=True)

                z_ae, x_hat, r_hat = ae(xb)
                logits = cls(xb, len_b, pad_b)
                feat = torch.cat([z_ae, logits], dim=-1)
                a = pol(feat)

                loss_tr, terms = trading_loss(
                    a, yb_r,
                    loss_type=args.loss_type, lambda_var=args.lambda_var,
                    tx_cost=args.tx_cost, turnover_coef=args.turnover_coef, l2_pos=args.l2_pos,
                    vol_norm=args.vol_norm, expo_target=args.expo_target, expo_w=args.expo_w,
                    mean_pos_target=args.mean_pos_target, mean_pos_w=args.mean_pos_w
                )
                loss_rec = F.smooth_l1_loss(x_hat, xb, reduction="mean")
                loss_cls = cls_loss_fn(logits, yb_c)
                loss_aux = xb.new_tensor(0.0)
                if hasattr(args, "ae_aux_weight") and (args.ae_aux_weight>0.0) and (r_hat is not None):
                    loss_aux = F.mse_loss(r_hat, yb_r, reduction="mean")

                loss = (tr_weight * loss_tr) + args.rec_weight*loss_rec + args.cls_weight*loss_cls + getattr(args, "ae_aux_weight", 0.0)*loss_aux

                if not torch.isfinite(loss):
                    skipped_loss += 1
                    continue

                if mode=="train":
                    opt.zero_grad(set_to_none=True)
                    loss.backward()

                    # Chequeo de gradientes antes del step
                    bad_grad = False
                    for p in ae.parameters():
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            bad_grad = True; break
                    if not bad_grad:
                        for p in cls.parameters():
                            if p.grad is not None and not torch.isfinite(p.grad).all():
                                bad_grad = True; break
                    if not bad_grad:
                        for p in pol.parameters():
                            if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all():
                                bad_grad = True; break

                    if bad_grad:
                        skipped_badgrad += 1
                        # reducimos LR para intentar estabilizar
                        for g in opt.param_groups:
                            g["lr"] = max(g["lr"] * 0.5, 1e-6)
                        opt.zero_grad(set_to_none=True)
                        continue

                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()

                agg["loss"] += float(loss.detach().cpu())
                agg["ae_rec"] += float(loss_rec.detach().cpu())
                agg["ae_aux"] += float(loss_aux.detach().cpu())
                agg["turnover"] += float(terms.get("turnover", 0.0))
                agg["cost"]     += float(terms.get("cost", 0.0))
                pos_all.append(a.detach().cpu().numpy()); nb += 1

        pos_all = np.concatenate(pos_all) if pos_all else np.zeros(0, dtype=np.float32)
        y_np = y_r[(sl.start or 0):(sl.stop if sl.stop is not None else N)].cpu().numpy()
        trm = eval_trading_split_np(pos_all, y_np)
        clm = eval_cls_split(cls, X, y_c, lengths_t, pad_mask_t, device, sl, args.batch_size, N)

        for k in ["loss","ae_rec","ae_aux","turnover","cost"]:
            agg[k] /= max(1, nb)
        agg.update({"tr_mu":trm["mu"], "tr_sd":trm["sd"], "tr_S":trm["sharpe_like"],
                    "cls_acc":clm["acc"], "cls_bal":clm["bal_acc"],
                    "skipped_loss": skipped_loss, "skipped_badgrad": skipped_badgrad, "batches": nb})
        return agg, pos_all

    print(f"\n=== Train JOINT AE+CLS+Policy world={world.name} | Policy={args.policy.upper()} ===")

    best_va = -1e9; best_ep = 0; bad = 0
    for ep in range(1, args.epochs+1):
        # Peso de trading con rampa tras warmup
        if ep <= args.warmup_epochs:
            tr_weight = 0.0
        else:
            k = max(1, int(args.tr-ramp-epochs) if hasattr(args, "tr-ramp-epochs") else int(args.tr_ramp_epochs))
            tr_weight = min(1.0, (ep - args.warmup_epochs) / k)

        tr, _ = run_epoch("train", idx_tr, ep, tr_weight)
        va, _ = run_epoch("eval",  idx_va, ep, tr_weight)
        te, _ = run_epoch("eval",  idx_te, ep, tr_weight)

        # LR actual (primer param group)
        cur_lr = opt.param_groups[0]["lr"]

        print(f"[Ep{ep:02d}] "
              f"Tr S~={tr['tr_S']:.3f} | Va S~={va['tr_S']:.3f} | Te S~={te['tr_S']:.3f} | "
              f"loss={tr['loss']:.4f} | AE(rec)={tr['ae_rec']:.4f} aux={tr['ae_aux']:.4f} | "
              f"CLS(acc/bal)={tr['cls_acc']:.3f}/{tr['cls_bal']:.3f} | "
              f"skipTr={tr['skipped_loss']+tr['skipped_badgrad']}/{tr['batches']} "
              f"skipVa={va['skipped_loss']+va['skipped_badgrad']}/{va['batches']} | "
              f"lr={cur_lr:.2e} | tr_w={tr_weight:.2f}")

        # scheduler step tras train epoch (solo durante warmup)
        if sched is not None and ep <= args.warmup_epochs:
            sched.step()

        score = va["tr_S"]
        improved = (score - best_va) > args.min_delta

        outdir = (world/"trained"); outdir.mkdir(parents=True, exist_ok=True)
        ckpt = outdir / f"JOINT_AECLS_CTRL_H{args.horizon_min}m_T{args.window_len}_Z{args.ae_latent}_{world.name}.pt"

        if ep <= args.warmup_epochs:
            if score > best_va and np.isfinite(score):
                best_va = score; best_ep = ep; bad = 0
                torch.save({
                    "ae": ae.state_dict(), "cls": cls.state_dict(), "pol": pol.state_dict(),
                    "norm": norm, "cfg": vars(args),
                    "train": tr, "val": va, "test": te,
                    "tau": float(tau),
                    "arch": {"cls": args.cls_arch, "policy": args.policy}
                }, ckpt)
        else:
            if improved and np.isfinite(score):
                best_va = score; best_ep = ep; bad = 0
                torch.save({
                    "ae": ae.state_dict(), "cls": cls.state_dict(), "pol": pol.state_dict(),
                    "norm": norm, "cfg": vars(args),
                    "train": tr, "val": va, "test": te,
                    "tau": float(tau),
                    "arch": {"cls": args.cls_arch, "policy": args.policy}
                }, ckpt)
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"Early stopping Ep{ep} (best Va S~={best_va:.4f} @Ep{best_ep})")
                    break

    print("\n✔ Entrenamiento conjunto AE+CLS+Policy finalizado.")
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

