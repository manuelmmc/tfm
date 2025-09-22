#!/usr/bin/env python
# scripts_nuevos/eval_cls_as_policy_2.py
"""
Evalúa Sharpe usando directamente las predicciones del clasificador (acciones = {-1,0,+1}).

- Carga un checkpoint de train_memory_cls_weighted.py (modo rich).
- Reconstruye features y ventanas exactamente igual.
- Usa argmax de la probabilidad para mapear clases -> acciones: {DOWN: -1, NEUTRAL: 0, UP: +1}.
- Auto-detecta si el checkpoint tiene atención (claves 'attnp.*') y arma el modelo acorde.
- Soporta mismatch 9↔10 features (lambda_proxy) usando las mu/sd de ckpt.

Uso:
  python scripts_nuevos/eval_cls_as_policy.py \
    --world prepared/SPX \
    --ckpt prepared/SPX/trained/CLS_TRANSFORMER_richT96_zH_lossCES0.02_autoNM1.2_H60m_SPX.pt \
    --device cuda:0
"""

import argparse, json, math
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

DT = np.float32

# ---------- utilidades ----------
def load_price_series(world: Path) -> np.ndarray:
    for name in ["SYN_PRICE.npy","SYN_SERIE_5m.npy","PRICE_5m.npy","SERIE_5m.npy"]:
        fn = world / name
        if fn.exists(): return np.load(fn).astype(np.float64)
    raise FileNotFoundError(f"No price series in {world}")

def load_step_min(world: Path, default=5) -> int:
    for fn in ["SYN_META.json", "meta.json"]:
        p = world / fn
        if p.exists():
            try:
                meta = json.loads(p.read_text())
                return int(meta.get("dt_min", meta.get("step_min", default)))
            except Exception:
                pass
    return default

def maybe_load_lambda_proxy(world: Path) -> Optional[np.ndarray]:
    for name in ["SYN_LAMBDA.npy", "HAWKES_LAMBDA.npy", "LAMBDA.npy"]:
        fn = world / name
        if fn.exists():
            try:
                return np.load(fn).astype(np.float64)
            except Exception:
                return None
    return None

def split_indices(N: int, tr=0.70, va=0.15):
    n_tr = int(tr * N); n_va = int(va * N)
    return slice(0, n_tr), slice(n_tr, n_tr+n_va), slice(n_tr+n_va, N)

def compute_rH(price: np.ndarray, Hmin: int, step_min: int) -> np.ndarray:
    steps = max(1, Hmin // max(1, step_min))
    p = np.clip(price.astype(np.float64), 1e-9, None)
    r = (p[steps:] - p[:-steps]) / p[:-steps]
    return r.astype(DT)

# ---------- features (idénticas al training "rich") ----------
def trailing_mean(x, w):
    if w <= 1: return x.copy()
    c = np.cumsum(np.insert(x, 0, 0.0))
    m = (c[w:] - c[:-w]) / w
    out = np.empty_like(x, dtype=np.float64)
    out[:w-1] = x[:w-1]; out[w-1:] = m
    return out

def rolling_std(x, w):
    if w <= 1: return np.zeros_like(x)
    c1 = np.cumsum(np.insert(x, 0, 0.0))
    c2 = np.cumsum(np.insert(x*x, 0, 0.0))
    s = (c1[w:] - c1[:-w]) / w
    s2 = (c2[w:] - c2[:-w]) / w
    var = np.maximum(0.0, s2 - s*s)
    out = np.zeros_like(x); out[w-1:] = np.sqrt(var)
    return out

def build_raw_features(price: np.ndarray, step_min: int, lambda_proxy=None) -> np.ndarray:
    p = np.clip(price.astype(np.float64), 1e-9, None)
    logp = np.log(p)
    ret1 = np.diff(logp, prepend=logp[0])
    s1h = max(1, 60 // max(1, step_min))
    s6h = max(1, 360 // max(1, step_min))
    ma_fast = trailing_mean(p, s1h) / p
    ma_slow = trailing_mean(p, s6h) / p
    p_lag_1h = np.concatenate([p[:s1h], p[:-s1h]])
    mom_1h = (p - p_lag_1h) / p_lag_1h
    vol_1h = rolling_std(ret1, s1h)
    vol_6h = rolling_std(ret1, s6h)
    N = p.size
    day_period = float((24*60) // max(1, step_min))
    idx = np.arange(N, dtype=np.float64)
    sin_t = np.sin(2*np.pi * (idx % day_period) / day_period)
    cos_t = np.cos(2*np.pi * (idx % day_period) / day_period)
    feats = [logp, ret1, ma_fast, ma_slow, mom_1h, vol_1h, vol_6h, sin_t, cos_t]
    if lambda_proxy is not None and lambda_proxy.shape[0] == N:
        feats.append(lambda_proxy.astype(np.float64))
    X_all = np.stack(feats, axis=1)
    return np.nan_to_num(X_all, 0.0, 0.0, 0.0)

def make_windows_from_matrix(X_all: np.ndarray, T: int):
    N, F = X_all.shape
    X = np.zeros((N, T, F), dtype=DT)
    lengths = np.zeros((N,), dtype=np.int64)
    for i in range(N):
        s = i - T + 1
        if s < 0:
            X[i, -i-1:, :] = X_all[:i+1, :]
            lengths[i] = i + 1
        else:
            X[i, :, :] = X_all[s:i+1, :]
            lengths[i] = T
    return X, lengths

# ---------- modelos ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe)
    def forward(self, x): return x + self.pe[:x.size(1)].unsqueeze(0)

class AttnPool(nn.Module):
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.w = nn.Linear(d, 1)
        self.drop = nn.Dropout(dropout)
    def forward(self, z, pad_mask):
        scores = self.w(self.drop(z)).squeeze(-1)     # (B,T)
        scores = scores.masked_fill(pad_mask, -1e9)
        attn = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B,T,1)
        return (z * attn).sum(dim=1)                  # (B,D)

class TransformerClassifierSimple(nn.Module):
    """Pooling por media (sin atención) – retrocompatible."""
    def __init__(self, feat_dim=2, d_model=128, nhead=4, ff=256, layers=3, dropout=0.1, num_classes=3):
        super().__init__()
        self.inp = nn.Linear(feat_dim, d_model); self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout,
                                               batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model),
                                  nn.Linear(d_model, d_model), nn.ReLU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(d_model, num_classes))
    def forward(self, x, lengths, pad_mask):
        h = self.inp(x); h = self.pos(h)
        z = self.enc(h, src_key_padding_mask=pad_mask)
        valid = (~pad_mask).unsqueeze(-1); z = z * valid
        denom = lengths.clamp_min(1).unsqueeze(1).float()
        pooled = z.sum(dim=1) / denom
        return self.head(pooled)

class TransformerClassifierAttn(nn.Module):
    """Con atención de pooling (attnp.*), compatible con tus ckpts nuevos."""
    def __init__(self, feat_dim=2, d_model=128, nhead=4, ff=256, layers=3, dropout=0.1, num_classes=3):
        super().__init__()
        self.inp = nn.Linear(feat_dim, d_model); self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout,
                                               batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.attnp = AttnPool(d_model, dropout=dropout)
        self.head = nn.Sequential(nn.LayerNorm(d_model),
                                  nn.Linear(d_model, d_model), nn.ReLU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(d_model, num_classes))
    def forward(self, x, lengths, pad_mask):
        h = self.inp(x); h = self.pos(h)
        z = self.enc(h, src_key_padding_mask=pad_mask)
        pooled = self.attnp(z, pad_mask)
        return self.head(pooled)

# ---------- métricas ----------
def eval_sharpe(actions: np.ndarray, r_future: np.ndarray):
    n = min(len(actions), len(r_future))
    a = actions[:n]; r = r_future[:n]
    pnl = a * r
    mu = float(pnl.mean()); sd = float(pnl.std() + 1e-12)
    sharpe = mu / sd
    hit = float(np.mean(np.sign(a) == np.sign(r)))
    return {"mu": mu, "sd": sd, "sharpe": float(sharpe), "hit": hit}

# ---------- main ----------
def main():
    import math
    pa = argparse.ArgumentParser()
    pa.add_argument("--world", required=True)
    pa.add_argument("--ckpt", required=True, help="Checkpoint del clasificador")
    pa.add_argument("--device", default="cuda:0")
    args = pa.parse_args()

    world = Path(args.world).expanduser().resolve()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    assert ckpt.get("feature_mode","rich") == "rich", "Este script asume modo 'rich'."

    device = torch.device(args.device)
    H = int(ckpt.get("horizon_min", 60))
    T = int(ckpt.get("context_len", 256))
    model_cfg = ckpt.get("model_cfg", {})
    mu = ckpt["feature_norm"]["mu"].astype(np.float32)
    sd = ckpt["feature_norm"]["sd"].astype(np.float32)

    step_min = load_step_min(world, default=5)
    price = load_price_series(world)
    lam = maybe_load_lambda_proxy(world)

    rH_full = compute_rH(price, H, step_min)
    X_all = build_raw_features(price, step_min, lambda_proxy=lam)

    # Alinear por horizonte y hacer ventanas
    steps_H = max(1, H // max(1, step_min))
    N_eff = min(X_all.shape[0]-steps_H, rH_full.shape[0])
    X_all = X_all[:N_eff, :]
    rH = rH_full[:N_eff]
    X_raw, lengths = make_windows_from_matrix(X_all, T=T)

    # ---- Normalizar con stats del checkpoint manejando 9↔10 features
    def norm_with_stats(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
        d = mu.shape[0]; f = X.shape[2]
        if d == f:
            return (X - mu.astype(np.float32)) / (sd.astype(np.float32) + 1e-9)
        elif d == f - 1:
            # El ckpt esperaba 9 (sin lambda) pero X tiene 10 → recortamos última col (lambda)
            Xs = X[..., :d]
            return (Xs - mu.astype(np.float32)) / (sd.astype(np.float32) + 1e-9)
        elif d == f + 1:
            # El ckpt esperaba 10 (con lambda) pero X trae 9 → no tenemos lambda_proxy
            raise SystemExit("El ckpt esperaba 10 features (incl. lambda_proxy) pero el mundo no la tiene.")
        else:
            raise SystemExit(f"Dim mismatch: ckpt mu/sd={d}, X features={f}")

    X = norm_with_stats(X_raw, mu, sd)

    # Máscaras
    Tctx = X.shape[1]
    pad_mask = np.zeros((X.shape[0], Tctx), dtype=bool)
    for i, L in enumerate(lengths):
        if L < Tctx: pad_mask[i, :Tctx-L] = True

    # Tensores
    X_t = torch.from_numpy(X.astype(np.float32))
    pad_t = torch.from_numpy(pad_mask)
    len_t = torch.from_numpy(lengths.astype(np.int64))

    # Slices
    idx_tr, idx_va, idx_te = split_indices(X.shape[0], 0.70, 0.15)

    # ¿El ckpt tiene atención?
    state_dict = ckpt["state_dict"]
    has_attn = any(k.startswith("attnp.") for k in state_dict.keys())

    # Modelo
    common_kwargs = dict(
        feat_dim=X.shape[2],
        d_model=model_cfg.get("d_model", 128),
        nhead=model_cfg.get("nhead", 4),
        ff=model_cfg.get("dim_feedforward", 256),
        layers=model_cfg.get("num_layers", 3),
        dropout=model_cfg.get("dropout", 0.1),
        num_classes=3
    )
    if has_attn:
        net = TransformerClassifierAttn(**common_kwargs).to(device)
    else:
        net = TransformerClassifierSimple(**common_kwargs).to(device)

    # Cargar pesos (strict=True si coincide, sino probar strict=False por si hay buffers no críticos)
    try:
        net.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"[WARN] load_state_dict strict=True falló: {e}\nReintentando con strict=False…")
        net.load_state_dict(state_dict, strict=False)

    net.eval()

    @torch.no_grad()
    def predict_actions(sl: slice):
        start = sl.start or 0
        stop = sl.stop if sl.stop is not None else X_t.size(0)
        BS = 1024
        acts = []
        for i in range(start, stop, BS):
            end = min(i+BS, stop)
            xb = X_t[i:end].to(device, non_blocking=True)
            pb = pad_t[i:end].to(device, non_blocking=True)
            lb = len_t[i:end].to(device, non_blocking=True)
            logits = net(xb, lb, pb)
            yhat = torch.argmax(logits, dim=1).cpu().numpy()
            # clase -> acción
            a = np.where(yhat==2, 1.0, np.where(yhat==1, 0.0, -1.0)).astype(np.float32)
            acts.append(a)
        return np.concatenate(acts) if acts else np.zeros(0, dtype=np.float32)

    # Eval
    a_tr = predict_actions(idx_tr)
    a_va = predict_actions(idx_va)
    a_te = predict_actions(idx_te)

    r_tr = rH[idx_tr]
    r_va = rH[idx_va]
    r_te = rH[idx_te]

    def eval_sharpe(actions: np.ndarray, r_future: np.ndarray):
        n = min(len(actions), len(r_future))
        a = actions[:n]; r = r_future[:n]
        pnl = a * r
        mu = float(pnl.mean()); sd = float(pnl.std() + 1e-12)
        sharpe = mu / sd
        hit = float(np.mean(np.sign(a) == np.sign(r)))
        return {"mu": mu, "sd": sd, "sharpe": float(sharpe), "hit": hit}

    m_tr = eval_sharpe(a_tr, r_tr)
    m_va = eval_sharpe(a_va, r_va)
    m_te = eval_sharpe(a_te, r_te)

    def fmt(m): return f"Sharpe={m['sharpe']:.4f} | mu={m['mu']:.6g} sd={m['sd']:.6g} | hit={100*m['hit']:.2f}%"
    print(f"World={world.name} | H={H}m | T={T}")
    print(f"TRAIN: {fmt(m_tr)}")
    print(f"VAL  : {fmt(m_va)}")
    print(f"TEST : {fmt(m_te)}")

if __name__ == "__main__":
    main()

