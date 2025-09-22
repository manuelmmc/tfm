#!/usr/bin/env python
# scripts_nuevos/eval_ensemble_ctrl_plus_cls.py
"""
Evalúa un ensemble: acciones = clip( a_controlador + a_clasificador , -1, 1 )

- Controlador: checkpoint de train_trading_controller_direct.py (TinyTransformerPolicy continuo).
- Clasificador: checkpoint de train_memory_cls_weighted.py
  * Soporta ambas variantes:
      (a) con attention pooling (state_dict incluye 'attnp.w.*')
      (b) con pooling por media enmascarada (sin 'attnp.*')
- Features "rich" (sin fuga) y ventanas T; normalización tomada de cada checkpoint.

Uso (SPX):
  python scripts_nuevos/eval_ensemble_ctrl_plus_cls.py \
    --world prepared/SPX \
    --ckpt-controller prepared/SPX/trained/CTRL_DIRECT_TRANSFORMER_H60m_SPX.pt \
    --ckpt-cls prepared/SPX/trained/CLS_TRANSFORMER_richT96_zH_lossCES0.02_autoNM1.2_H60m_SPX.pt \
    --device cuda:0
"""

import argparse, json, math
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DT = np.float32

# ---------- Utilidades ----------
def load_price_series(world: Path) -> np.ndarray:
    for name in ["SYN_PRICE.npy","SYN_SERIE_5m.npy","PRICE_5m.npy","SERIE_5m.npy"]:
        fn = world / name
        if fn.exists(): return np.load(fn).astype(np.float64)
    raise FileNotFoundError(f"No price series in {world}")

def load_step_min(world: Path, default=5) -> int:
    for fn in ["SYN_META.json","meta.json"]:
        p = world / fn
        if p.exists():
            try:
                meta = json.loads(p.read_text())
                return int(meta.get("dt_min", meta.get("step_min", default)))
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

def split_indices(N: int, tr=0.70, va=0.15):
    n_tr = int(tr*N); n_va = int(va*N)
    return slice(0,n_tr), slice(n_tr,n_tr+n_va), slice(n_tr+n_va,N)

def compute_rH(price: np.ndarray, Hmin: int, step_min: int) -> np.ndarray:
    steps = max(1, Hmin // max(1, step_min))
    p = np.clip(price.astype(np.float64), 1e-9, None)
    r = (p[steps:] - p[:-steps]) / p[:-steps]
    return r.astype(DT)

# ---------- Features ricas ----------
def trailing_mean(x, w):
    if w<=1: return x.copy()
    c = np.cumsum(np.insert(x,0,0.0))
    m = (c[w:]-c[:-w])/w
    out = np.empty_like(x, dtype=np.float64)
    out[:w-1] = x[:w-1]; out[w-1:] = m
    return out

def rolling_std(x, w):
    if w<=1: return np.zeros_like(x)
    c1 = np.cumsum(np.insert(x,0,0.0))
    c2 = np.cumsum(np.insert(x*x,0,0.0))
    s = (c1[w:]-c1[:-w])/w
    s2 = (c2[w:]-c2[:-w])/w
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
    if lambda_proxy is not None and lambda_proxy.shape[0]==N:
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

def normalize_with_stats(X_raw: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Robusto a casos 9↔10 features (lambda_proxy). Si mu tiene d y X_raw F:
      - d==F: normaliza tal cual
      - d==F-1: descarta la última columna de X_raw (asumimos lambda en la última)
      - d==F+1: error (faltan features en X_raw)
    """
    d = int(mu.shape[0]); F = int(X_raw.shape[2])
    if d == F:
        return (X_raw - mu) / (sd + 1e-9)
    elif d == F-1:
        Xs = X_raw[..., :d]
        return (Xs - mu) / (sd + 1e-9)
    else:
        raise AssertionError(f"Dim mismatch: mu/sd({d}) vs F({F}). ¿faltan/sobran features?")

# ---------- Modelos ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe)
    def forward(self, x): return x + self.pe[:x.size(1)].unsqueeze(0)

class TinyTransformerPolicy(nn.Module):
    """Misma arquitectura que train_trading_controller_direct.py (Transformer)."""
    def __init__(self, in_dim, d_model=128, nhead=4, ff=256, layers=2, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(in_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff,
            batch_first=True, dropout=dropout, norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
    def forward(self, x):
        h = self.inp(x); h = self.pos(h)
        z = self.enc(h)            # sin máscara (como en training)
        pooled = z[:, -1, :]       # último token (como en training)
        return torch.tanh(self.head(pooled)).squeeze(-1)

class AttnPool(nn.Module):
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.w = nn.Linear(d, 1)
        self.drop = nn.Dropout(dropout)
    def forward(self, z, pad_mask):
        # z: (B,T,D), pad_mask: (B,T) True=pad
        scores = self.w(self.drop(z)).squeeze(-1)      # (B,T)
        scores = scores.masked_fill(pad_mask, -1e9)
        attn = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B,T,1)
        return (z * attn).sum(dim=1)                   # (B,D)

class TransformerClassifier(nn.Module):
    """Clasificador 3 clases con opción de atención o pooling por media."""
    def __init__(self, feat_dim=10, d_model=128, nhead=4, ff=256, layers=3, dropout=0.1,
                 num_classes=3, use_attn: bool = False):
        super().__init__()
        self.use_attn = use_attn
        self.inp = nn.Linear(feat_dim, d_model); self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout,
                                               batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        if use_attn:
            self.attnp = AttnPool(d_model, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
    def forward(self, x, lengths, pad_mask):
        h = self.inp(x); h = self.pos(h)
        z = self.enc(h, src_key_padding_mask=pad_mask)
        if self.use_attn:
            pooled = self.attnp(z, pad_mask)
        else:
            valid = (~pad_mask).unsqueeze(-1); z = z * valid
            denom = lengths.clamp_min(1).unsqueeze(1).float()
            pooled = z.sum(dim=1) / denom
        return self.head(pooled)

# ---------- Métricas ----------
def eval_sharpe(actions: np.ndarray, r_future: np.ndarray):
    n = min(len(actions), len(r_future))
    a = actions[:n]; r = r_future[:n]
    pnl = a * r
    mu = float(pnl.mean()); sd = float(pnl.std() + 1e-12)
    sharpe = mu / sd
    hit = float(np.mean(np.sign(a) == np.sign(r)))
    return {"mu": mu, "sd": sd, "sharpe": float(sharpe), "hit": hit}

# ---------- Main ----------
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--world", required=True)
    pa.add_argument("--ckpt-controller", required=True)
    pa.add_argument("--ckpt-cls", required=True)
    pa.add_argument("--device", default="cuda:0")
    args = pa.parse_args()

    device = torch.device(args.device)
    world = Path(args.world).expanduser().resolve()
    step_min = load_step_min(world, default=5)
    price = load_price_series(world)
    lam = maybe_load_lambda_proxy(world)

    # Cargar ckpts
    ckpt_ctrl = torch.load(args.ckpt_controller, map_location="cpu")
    ckpt_cls  = torch.load(args.ckpt_cls, map_location="cpu")
    sd_cls    = ckpt_cls.get("state_dict", ckpt_cls)

    H_ctrl = int(ckpt_ctrl.get("horizon_min", ckpt_ctrl.get("cfg", {}).get("horizon_min", 60)))
    H_cls  = int(ckpt_cls.get("horizon_min", 60))
    if H_ctrl != H_cls:
        print(f"[Aviso] H diferentes (ctrl={H_ctrl}, cls={H_cls}). Se usará H={H_cls}.")
    H = H_cls

    T_ctrl = int(ckpt_ctrl.get("window_len", ckpt_ctrl.get("cfg", {}).get("window_len", 256)))
    T_cls  = int(ckpt_cls.get("context_len", ckpt_cls.get("window_len", 256)))
    if T_ctrl != T_cls:
        print(f"[Aviso] T diferentes (ctrl={T_ctrl}, cls={T_cls}). Usaré T={T_cls} al construir ventanas.")
    T = T_cls

    # Datos y ventanas
    rH_full = compute_rH(price, H, step_min)
    X_all = build_raw_features(price, step_min, lambda_proxy=lam)
    N_eff = min(X_all.shape[0] - max(1, H // step_min), rH_full.shape[0])
    X_all = X_all[:N_eff]; rH = rH_full[:N_eff]
    X_raw, lengths = make_windows_from_matrix(X_all, T=T)

    idx_tr, idx_va, idx_te = split_indices(X_raw.shape[0], 0.70, 0.15)
    pad_mask = np.zeros((X_raw.shape[0], T), dtype=bool)
    for i, L in enumerate(lengths):
        if L < T: pad_mask[i, :T-L] = True
    pad_t = torch.from_numpy(pad_mask)
    len_t = torch.from_numpy(lengths.astype(np.int64))

    # Normalizaciones
    # Clasificador: stats del ckpt (obligatorio, robusto a 9/10 features)
    mu_cls = ckpt_cls["feature_norm"]["mu"].astype(np.float32)
    sd_cls = ckpt_cls["feature_norm"]["sd"].astype(np.float32)
    X_cls = normalize_with_stats(X_raw, mu_cls, sd_cls)

    # Controlador: usa feature_norm si está en ckpt; si no, calcula en TRAIN
    if "feature_norm" in ckpt_ctrl and isinstance(ckpt_ctrl["feature_norm"], dict):
        mu_ctrl = np.asarray(ckpt_ctrl["feature_norm"]["mu"], dtype=np.float32)
        sd_ctrl = np.asarray(ckpt_ctrl["feature_norm"]["sd"], dtype=np.float32)
        X_ctrl = normalize_with_stats(X_raw, mu_ctrl, sd_ctrl)
    else:
        Xtr_flat = X_raw[idx_tr].reshape(-1, X_raw.shape[2]).astype(np.float64)
        mu_ctrl = np.mean(Xtr_flat, axis=0).astype(np.float32)
        sd_ctrl = (np.std(Xtr_flat, axis=0) + 1e-9).astype(np.float32)
        X_ctrl = (X_raw - mu_ctrl) / (sd_ctrl + 1e-9)

    # Tensores
    X_cls_t  = torch.from_numpy(X_cls.astype(np.float32))
    X_ctrl_t = torch.from_numpy(X_ctrl.astype(np.float32))

    # Clasificador (detectar si el ckpt tiene atención)
    mc = ckpt_cls.get("model_cfg", {})
    use_attn = any(k.startswith("attnp.") for k in sd_cls.keys())
    net_cls = TransformerClassifier(
        feat_dim=X_cls.shape[2],
        d_model=mc.get("d_model", 128),
        nhead=mc.get("nhead", 4),
        ff=mc.get("dim_feedforward", 256),
        layers=mc.get("num_layers", 3),
        dropout=mc.get("dropout", 0.1),
        num_classes=3,
        use_attn=use_attn
    ).to(device)
    # Carga estricta (debe cuadrar ahora que la arquitectura tiene attn condicional)
    net_cls.load_state_dict(sd_cls, strict=True)
    net_cls.eval()

    # Controlador: misma arquitectura que en training (FF=256, layers=2)
    net_ctrl = TinyTransformerPolicy(
        in_dim=X_ctrl.shape[2],
        d_model=128, nhead=4, ff=256, layers=2, dropout=0.1
    ).to(device)

    # Extraer state_dict del controlador
    state_dict_ctrl = ckpt_ctrl.get("state_dict", None)
    if state_dict_ctrl is None and any(isinstance(v, torch.Tensor) for v in ckpt_ctrl.values()):
        state_dict_ctrl = ckpt_ctrl
    if state_dict_ctrl is None:
        raise SystemExit("No se pudo localizar 'state_dict' del controlador en el checkpoint.")
    missing, unexpected = net_ctrl.load_state_dict(state_dict_ctrl, strict=False)
    if missing or unexpected:
        print(f"[Aviso] load_state_dict controlador: missing={len(missing)} unexpected={len(unexpected)}")
    net_ctrl.eval()

    # Predictores
    @torch.no_grad()
    def predict_actions_cls(sl: slice):
        start = sl.start or 0; stop = sl.stop if sl.stop is not None else X_cls_t.size(0)
        BS = 1024; acts = []
        for i in range(start, stop, BS):
            end = min(i+BS, stop)
            xb = X_cls_t[i:end].to(device, non_blocking=True)
            pb = pad_t[i:end].to(device, non_blocking=True)
            lb = len_t[i:end].to(device, non_blocking=True)
            logits = net_cls(xb, lb, pb)
            yhat = torch.argmax(logits, dim=1).cpu().numpy()
            a = np.where(yhat==2, 1.0, np.where(yhat==1, 0.0, -1.0)).astype(np.float32)
            acts.append(a)
        return np.concatenate(acts) if acts else np.zeros(0, dtype=np.float32)

    @torch.no_grad()
    def predict_actions_ctrl(sl: slice):
        start = sl.start or 0; stop = sl.stop if sl.stop is not None else X_ctrl_t.size(0)
        BS = 1024; acts = []
        for i in range(start, stop, BS):
            end = min(i+BS, stop)
            xb = X_ctrl_t[i:end].to(device, non_blocking=True)
            a = net_ctrl(xb).cpu().numpy().astype(np.float32)
            acts.append(a)
        return np.concatenate(acts) if acts else np.zeros(0, dtype=np.float32)

    # Evaluación
    def fmt(m): return f"Sharpe={m['sharpe']:.4f} | mu={m['mu']:.6g} sd={m['sd']:.6g} | hit={100*m['hit']:.2f}%"
    splits = {"TRAIN": split_indices(X_raw.shape[0])[0],
              "VAL":   split_indices(X_raw.shape[0])[1],
              "TEST":  split_indices(X_raw.shape[0])[2]}

    print(f"World={world.name} | H={H}m | T={T} | step_min={step_min} | use_attn_cls={use_attn}")
    for name, sl in splits.items():
        a_cls  = predict_actions_cls(sl)
        a_ctrl = predict_actions_ctrl(sl)
        a_ens  = np.clip(a_cls + a_ctrl, -1.0, 1.0)
        r = rH[sl]
        m_cls  = eval_sharpe(a_cls,  r)
        m_ctrl = eval_sharpe(a_ctrl, r)
        m_ens  = eval_sharpe(a_ens,  r)
        print(f"\n[{name}]")
        print(f"  CLS : {fmt(m_cls)}")
        print(f"  CTRL: {fmt(m_ctrl)}")
        print(f"  ENS : {fmt(m_ens)}")

if __name__ == "__main__":
    main()

