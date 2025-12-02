#!/usr/bin/env python
# src/models/controller/train_ctrl_v3.py
"""
Controller con contexto (AE + CLS) robusto para series reales:

- Interpolación sólo en huecos cortos (en log-precio) + segmentación por tramos válidos.
- Ventanas completas sólo dentro de segmentos (sin padding).
- Normalización con mu/sd guardadas en ckpts (AE y CLS) con control de mismatch (9 vs 10 features).
- AE y CLS congelados: se extrae z_AE + (pooled_CLS, logits_CLS) como contexto.
- Policy (MLP o Transformer) entrenada con Sharpe o μ-λ·Var.
- LR warmup y guardias anti-NaN/Inf.

Salida:
  prepared/<world>/trained/CTRL_CTX_AECLS_T{T}_H{H}m_Z{Z}_IN{D}_{world}.pt
"""

import argparse, json, math
from pathlib import Path
from typing import Optional, List, Tuple, Dict
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
    raise FileNotFoundError(f"No hay serie de precio en {world}")

def load_step_min(world: Path, default=5) -> int:
    for fn in ["SYN_META.json","meta.json"]:
        p = world / fn
        if p.exists():
            try:
                meta = json.loads(p.read_text())
                v = int(meta.get("step_min", meta.get("dt_min", default)))
                return v if v>0 else default
            except Exception: pass
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
    return slice(0,n_tr), slice(n_tr,n_tr+n_va), slice(n_tr+n_va,None)

# ───────── Segmentación y features (robusto) ─────────

def _nan_runs(mask_nan: np.ndarray) -> List[Tuple[int,int]]:
    runs = []
    n = mask_nan.size; i = 0
    while i < n:
        if mask_nan[i]:
            j = i+1
            while j < n and mask_nan[j]: j += 1
            runs.append((i,j)); i = j
        else:
            i += 1
    return runs

def interp_small_gaps_logprice(p: np.ndarray, max_gap_steps: int) -> np.ndarray:
    x = p.astype(np.float64).copy()
    x[x<=0] = np.nan
    logx = np.log(x)
    isnan = ~np.isfinite(logx)
    for i,j in _nan_runs(isnan):
        L = j - i
        if L <= max_gap_steps:
            left = i-1; right = j
            if left >= 0 and right < logx.size and np.isfinite(logx[left]) and np.isfinite(logx[right]):
                t = np.linspace(0.0, 1.0, L+2)[1:-1]
                logx[i:j] = (1.0 - t)*logx[left] + t*logx[right]
    return np.exp(logx)

def segments_from_valid(valid: np.ndarray) -> List[Tuple[int,int]]:
    segs = []
    n = valid.size; i = 0
    while i < n:
        if valid[i]:
            s = i; i += 1
            while i < n and valid[i]: i += 1
            e = i-1; segs.append((s,e))
        else:
            i += 1
    return segs

def trailing_mean_seg(x: np.ndarray, w: int) -> np.ndarray:
    if w<=1: return x.copy()
    c = np.cumsum(np.insert(x,0,0.0))
    m = (c[w:] - c[:-w]) / w
    out = np.empty_like(x, dtype=np.float64)
    out[:w-1] = x[:w-1]; out[w-1:] = m
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

    for (s,e) in segments_from_valid(valid_mask):
        p = price_finite[s:e+1]
        logp = np.log(p)
        ret1 = np.diff(logp, prepend=logp[0])
        ma_fast = trailing_mean_seg(p, s1h) / p
        ma_slow = trailing_mean_seg(p, s6h) / p
        lag = np.concatenate([p[:s1h], p[:-s1h]])
        with np.errstate(divide='ignore', invalid='ignore'):
            mom_1h = (p - lag) / lag
            mom_1h[~np.isfinite(mom_1h)] = 0.0
        vol_1h = rolling_std_seg(ret1, s1h)
        vol_6h = rolling_std_seg(ret1, s6h)
        sin_t = np.sin(2*np.pi * (idx[s:e+1] % day_period) / day_period)
        cos_t = np.cos(2*np.pi * (idx[s:e+1] % day_period) / day_period)

        cols = [logp, ret1, ma_fast, ma_slow, mom_1h, vol_1h, vol_6h, sin_t, cos_t]
        if lambda_proxy is not None and lambda_proxy.shape[0]==N:
            cols.append(lambda_proxy[s:e+1])
        X_all[s:e+1, :] = np.stack(cols, axis=1)

    return X_all

def build_dataset_no_impute(price_raw: np.ndarray,
                            step_min: int,
                            horizon_min: int,
                            window_len: int,
                            lambda_proxy: Optional[np.ndarray]=None,
                            max_gap_minutes: int = 60):
    """
    Devuelve:
      X_win: [M, T, F]  ventanas completas dentro de segmentos, sin NaNs
      rH   : [M]
      lengths: [M] (=T)
      pad_mask: [M, T] (todo False)
      info : dict auxiliar (features, mu/sd se aplican más tarde por ckpts)
    """
    N = int(price_raw.size)
    T = int(window_len)
    step = max(1, int(step_min))
    Hs  = max(1, int(horizon_min // step))
    max_gap_steps = max(1, int(round(max_gap_minutes / step)))

    # Interp huecos cortos en log-precio
    p_interp = interp_small_gaps_logprice(price_raw, max_gap_steps=max_gap_steps)
    valid_price = np.isfinite(p_interp) & (p_interp > 0)

    # lambda proxy (interp lineal en huecos cortos)
    lam_use = None
    if lambda_proxy is not None and lambda_proxy.shape[0]==N:
        lam_use = lambda_proxy.astype(np.float64).copy()
        isnan = ~np.isfinite(lam_use)
        for i,j in _nan_runs(isnan):
            L = j - i
            if L <= max_gap_steps:
                left = i-1; right = j
                if left>=0 and right<N and np.isfinite(lam_use[left]) and np.isfinite(lam_use[right]):
                    t = np.linspace(0.0,1.0,L+2)[1:-1]
                    lam_use[i:j] = (1.0-t)*lam_use[left] + t*lam_use[right]

    # Features segmentadas
    X_all = build_features_segmented(p_interp, valid_price, step_min, lambda_proxy=lam_use)

    # Target rH dentro de segmento
    rH_full = np.full(N, np.nan, dtype=np.float64)
    for (s,e) in segments_from_valid(valid_price):
        last = e - Hs
        if last < s: continue
        p = p_interp[s:e+1]
        rH_full[s:last+1] = (p[Hs:] - p[:-Hs]) / p[:-Hs]

    # Recolecta ventanas completas sin NaNs
    idx_t: List[int] = []
    for (s,e) in segments_from_valid(valid_price):
        start_t = s + (T - 1)
        end_t   = e - Hs
        if end_t < start_t: continue
        cand = np.arange(start_t, end_t+1, dtype=np.int64)
        for t in cand:
            sl = slice(t - (T - 1), t + 1)
            Xw = X_all[sl]
            if not np.isfinite(Xw).all(): continue
            if lam_use is not None:
                if (not np.isfinite(lam_use[sl]).all()) or (not np.isfinite(lam_use[t+Hs])): continue
            if not np.isfinite(rH_full[t]): continue
            idx_t.append(int(t))

    if len(idx_t)==0:
        raise RuntimeError("No hay ventanas válidas. Ajusta --max-gap-minutes o T/H.")

    idx_t = np.asarray(idx_t, dtype=np.int64)
    Fdim = X_all.shape[1]
    M = idx_t.size
    Xw = np.zeros((M, T, Fdim), dtype=DT)
    for i,t in enumerate(idx_t):
        Xw[i] = X_all[t-(T-1):t+1,:].astype(DT)
    y  = rH_full[idx_t].astype(DT)

    lengths = np.full(M, T, dtype=np.int64)
    pad_mask = np.zeros((M, T), dtype=bool)
    info = {
        "features": ["logp","ret1","ma1h/p","ma6h/p","mom_1h","vol_1h","vol_6h","sin_t","cos_t"] + (["lambda_proxy"] if (lam_use is not None) else []),
        "has_lambda": bool(lam_use is not None)
    }
    return Xw, y, lengths, pad_mask, info

# ───────── Modelos CLS (con atención) ─────────

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
    def __init__(self, feat_dim=2, hidden=128, layers=2, dropout=0.1, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(feat_dim, hidden, num_layers=layers,
                            dropout=(dropout if layers>1 else 0.0), batch_first=True)
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
        return logits, h

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe)
    def forward(self, x): return x + self.pe[:x.size(1)].unsqueeze(0)

class TransformerClassifier(nn.Module):
    def __init__(self, feat_dim=2, d_model=128, nhead=4, ff=256, layers=3, dropout=0.1, num_classes=3):
        super().__init__()
        self.inp = nn.Linear(feat_dim, d_model); self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
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
        pooled = self.attnp(z, pad_mask)
        logits = self.head(pooled)
        return logits, pooled

# ───────── AE encoder compatible ─────────

class AE_EncCompat(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, latent_dim: int, dropout: float):
        super().__init__()
        self.encoder = nn.LSTM(input_size=in_dim, hidden_size=hidden, num_layers=layers,
                               batch_first=True, dropout=(dropout if layers>1 else 0.0))
        self.to_latent = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, latent_dim), nn.Tanh())
    def forward(self, x):
        _, (hN, _) = self.encoder(x)
        return self.to_latent(hN[-1])

# ───────── Policies (trading) ─────────

class PolicyMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, x):
        return torch.tanh(self.net(x)).squeeze(-1)

class PolicyTransformer(nn.Module):
    """Atiende sobre 3 'tokens': z_AE, pooled_CLS, logits_CLS."""
    def __init__(self, z_dim:int, h_dim:int, logit_dim:int=3,
                 d_model:int=192, nhead:int=4, ff:int=384, layers:int=2, dropout:float=0.05):
        super().__init__()
        self.z_proj  = nn.Sequential(nn.Linear(z_dim,   d_model), nn.LayerNorm(d_model))
        self.h_proj  = nn.Sequential(nn.Linear(h_dim,   d_model), nn.LayerNorm(d_model))
        self.l_proj  = nn.Sequential(nn.Linear(logit_dim, d_model), nn.LayerNorm(d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout,
                                               batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.attnp = AttnPool(d_model, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
    def forward(self, z, h, logits):
        t1 = self.z_proj(z).unsqueeze(1)
        t2 = self.h_proj(h).unsqueeze(1)
        t3 = self.l_proj(logits).unsqueeze(1)
        x = torch.cat([t1, t2, t3], dim=1)    # [B,3,D]
        y = self.enc(x)
        pad_mask = torch.zeros(y.size(0), y.size(1), dtype=torch.bool, device=y.device)
        pooled = self.attnp(y, pad_mask)
        out = self.head(pooled).squeeze(-1)
        return torch.tanh(out)

def trading_loss(positions, r_future, *,
                 loss_type="meanvar",
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
    sd = var.clamp_min(1e-12).sqrt()
    base_obj = (mu/sd) if loss_type=="sharpe" else (mu - lambda_var*var)
    if a.numel() > 1:
        da = a[1:] - a[:-1]
        turnover = da.abs().mean()
        cost = tx_cost * da.abs().mean()
    else:
        turnover = a.new_tensor(0.0); cost = a.new_tensor(0.0)
    reg_l2 = l2_pos * (a.pow(2).mean())
    reg_expo = a.new_tensor(0.0)
    if (expo_target is not None) and (expo_w > 0.0):
        reg_expo = expo_w * (a.abs().mean() - float(expo_target))**2
    reg_mean = a.new_tensor(0.0)
    if mean_pos_w > 0.0:
        reg_mean = mean_pos_w * (a.mean() - float(mean_pos_target))**2
    objective = base_obj - cost - turnover_coef*turnover - reg_l2 - reg_expo - reg_mean
    loss = -objective
    terms = {
        "mu": float(mu.detach().cpu()),
        "sd": float(sd.detach().cpu()),
        "sharpe_like": float((mu/sd).detach().cpu()),
        "turnover": float(turnover.detach().cpu()),
        "cost": float(cost.detach().cpu()),
        "a_mean": float(a.mean().detach().cpu()),
        "a_abs_mean": float(a.abs().mean().detach().cpu()),
    }
    return loss, terms

# ───────── Carga de ckpts (AE/CLS) ─────────

def _as_numpy_vec(x):
    if x is None: return None
    if isinstance(x, (list, tuple)): return np.asarray(x, dtype=np.float32)
    if isinstance(x, np.ndarray): return x.astype(np.float32)
    if torch.is_tensor(x): return x.detach().cpu().numpy().astype(np.float32)
    return None

def _get_feat_norm_dict(ck):
    feat_norm = ck.get("feature_norm", None) or ck.get("norm", None) or {}
    if isinstance(feat_norm, dict):
        return {"mu": _as_numpy_vec(feat_norm.get("mu")), "sd": _as_numpy_vec(feat_norm.get("sd"))}
    return None

def load_ae_encoder(ckpt_path: Path, device):
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck.get("state_dict", ck)
    mc = ck.get("model_cfg", {}) or {}
    in_dim = mc.get("in_dim", None)
    hidden = mc.get("hidden", 128)
    layers = mc.get("layers", 2)
    latent_dim = mc.get("latent_dim", 64)
    dropout = mc.get("dropout", 0.1)

    # Si no viene in_dim en el ckpt, infiérelo del primer weight del encoder
    if in_dim is None:
        fn = _get_feat_norm_dict(ck) or {}
        mu = fn.get("mu", None)
        if mu is not None:
            in_dim = int(mu.shape[0])
        else:
            for k, v in sd.items():
                if k.endswith("encoder.weight_ih_l0"):
                    in_dim = int(v.shape[1]); hidden = int(v.shape[0] // 4)
                    break
    assert in_dim is not None, "No se pudo inferir in_dim del AE."

    enc = AE_EncCompat(in_dim=in_dim, hidden=hidden, layers=layers,
                       latent_dim=latent_dim, dropout=dropout).to(device)
    # Filtra sólo capas del encoder y proyección a latente
    allowed = {k: v for k, v in sd.items()
               if k.startswith("encoder.") or k.startswith("to_latent.")}
    enc.load_state_dict(allowed, strict=False)

    T = int(ck.get("window_len", 256))
    Haux = int(ck.get("aux_horizon_min", ck.get("horizon_min", 60)))
    feat_norm = _get_feat_norm_dict(ck)
    return enc, latent_dim, T, Haux, feat_norm


def load_classifier_from_ckpt(ckpt_path: Path, device):
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck.get("state_dict", ck)

    arch = str(ck.get("arch", "")).lower()
    if arch in ["transformer", "transf", "tfm"]:
        model_name = "transformer"
    elif arch in ["lstm", "rnn", "gru"]:
        model_name = "lstm"
    else:
        model_name = "transformer" if "trans" in arch else "lstm"

    mc = ck.get("model_cfg", {}) or {}
    feat_dim = int(mc.get("input_dim", mc.get("feat_dim", 10)))

    if model_name == "lstm":
        hidden = int(mc.get("hidden_dim", mc.get("hidden", 128)))
        layers = int(mc.get("num_layers", mc.get("layers", 2)))
        dropout = float(mc.get("dropout", 0.1))
        net = LSTMClassifier(feat_dim=feat_dim, hidden=hidden,
                             layers=layers, dropout=dropout, num_classes=3).to(device)
    else:
        d_model = int(mc.get("d_model", 128))
        nhead = int(mc.get("nhead", 4))
        ff = int(mc.get("dim_feedforward", 256))
        layers = int(mc.get("num_layers", 3))
        dropout = float(mc.get("dropout", 0.1))
        net = TransformerClassifier(feat_dim=feat_dim, d_model=d_model, nhead=nhead,
                                    ff=ff, layers=layers, dropout=dropout, num_classes=3).to(device)

    # Carga flexible de pesos
    model_state = net.state_dict()
    sd_filtered = {k: v for k, v in sd.items() if k in model_state and v.shape == model_state[k].shape}
    missing = [k for k in model_state.keys() if k not in sd_filtered]
    unexpected = [k for k in sd.keys() if k not in model_state]
    print(f"[load CLS] Cargando {len(sd_filtered)} pesos | faltan={len(missing)} | extra={len(unexpected)}")
    if unexpected:
        print(f"[load CLS] Ignorando claves extra (ej.): {', '.join(unexpected[:5])}{' ...' if len(unexpected)>5 else ''}")
    net.load_state_dict(sd_filtered, strict=False)

    T = int(ck.get("context_len", None) or ck.get("window_len", 256))
    H = int(ck.get("horizon_min", 60))
    feat_mode = ck.get("feature_mode", "rich")
    if feat_mode != "rich":
        raise SystemExit("Este controller requiere un CLS entrenado con features 'rich'.")
    feat_norm = _get_feat_norm_dict(ck)
    tau = float(ck.get("tau", 0.0))
    return net, model_name, T, H, feat_norm, tau


# ───────── Main robusto (sin NaNs / sin aprender 'Flat') ─────────

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--world", required=True)
    pa.add_argument("--ae-ckpt", required=True)
    pa.add_argument("--cls-ckpt", required=True)
    pa.add_argument("--horizon-min", type=int, default=None)

    # Datos reales
    pa.add_argument("--max-gap-minutes", type=int, default=60, help="Interpola huecos <= max_gap en log-precio; mayores → corta segmento")

    # Policy
    pa.add_argument("--policy", choices=["mlp", "transformer", "transf", "tfm"], default="mlp")
    pa.add_argument("--device", default="cuda:0")
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--deterministic", action="store_true")

    # MLP policy
    pa.add_argument("--hidden", type=int, default=256)
    pa.add_argument("--dropout", type=float, default=0.05)

    # Transformer policy
    pa.add_argument("--pol-d-model", type=int, default=192)
    pa.add_argument("--pol-nhead", type=int, default=4)
    pa.add_argument("--pol-ff", type=int, default=384)
    pa.add_argument("--pol-layers", type=int, default=2)

    # Trading objective
    pa.add_argument("--loss-type", choices=["meanvar", "sharpe"], default="sharpe")
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
    pa.add_argument("--epochs", type=int, default=25)
    pa.add_argument("--batch-size", type=int, default=512)
    pa.add_argument("--lr", type=float, default=3e-4)
    pa.add_argument("--weight-decay", type=float, default=1e-4)
    pa.add_argument("--patience", type=int, default=8)
    pa.add_argument("--min-delta", type=float, default=1e-4)

    args = pa.parse_args()
    if args.policy in ["transf", "tfm"]:
        args.policy = "transformer"

    set_seed(args.seed, deterministic=args.deterministic)
    device = torch.device(args.device)

    world = Path(args.world).expanduser().resolve()
    step_min = load_step_min(world, default=5)
    price = load_price_series(world)

    # Cargar AE / CLS congelados
    ae_enc, z_dim, T_ae, H_aux, ae_norm = load_ae_encoder(Path(args.ae_ckpt), device)
    cls_net, cls_arch, T_cls, H_cls, cls_norm, tau = load_classifier_from_ckpt(Path(args.cls_ckpt), device)

    H_target = int(args.horizon_min or H_cls)
    T_run = int(max(T_ae, T_cls))

    # Comprobar si los ckpts esperaban lambda_proxy
    ae_F = int(ae_norm["mu"].shape[0]) if (ae_norm and ae_norm.get("mu") is not None) else 9
    cls_F = int(cls_norm["mu"].shape[0]) if (cls_norm and cls_norm.get("mu") is not None) else 9
    need_lambda_any = (ae_F == 10) or (cls_F == 10)
    lam = maybe_load_lambda_proxy(world) if need_lambda_any else None
    if need_lambda_any and lam is None:
        raise SystemExit("Los ckpts esperan 'lambda_proxy' (10 features), pero no existe fichero de lambda en el mundo.")

    # Dataset robusto (segmentado, sin NaN, ventanas completas)
    X_win, rH, lengths, pad_mask, info = build_dataset_no_impute(
        price_raw=price,
        step_min=step_min,
        horizon_min=H_target,
        window_len=T_run,
        lambda_proxy=lam if need_lambda_any else None,
        max_gap_minutes=args.max_gap_minutes
    )
    N, T, Fbase = X_win.shape
    print(f"\n=== Controller con contexto (AE+CLS) world={world.name} ===")
    print(f"Device={device} | step_min={step_min} | N={N} T_run={T} Fbase={Fbase} | T_AE={T_ae} T_CLS={T_cls} | H_target={H_target}")
    print(f"AE expects F={ae_F} | CLS expects F={cls_F} | has_lambda_in_data={info['has_lambda']}")

    # Normalización con mu/sd de cada ckpt (manejo 9/10 features)
    def norm_with(stats, X):  # X: [N,T,Fbase]
        if not stats or stats.get("mu") is None or stats.get("sd") is None:
            raise AssertionError("El ckpt no contiene 'feature_norm' (mu/sd).")
        mu = stats["mu"].astype(DT); sd = stats["sd"].astype(DT)
        d = int(mu.shape[0]); f = X.shape[2]
        sd_safe = sd.copy()
        sd_safe[~np.isfinite(sd_safe)] = 1.0
        sd_safe[sd_safe < 1e-8] = 1.0

        if d == f:
            Z = (X - mu) / (sd_safe + 1e-9)
            return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
        elif d == f - 1:
            # El ckpt no usó lambda_proxy → usa primeras d columnas
            Xsub = X[..., :d]
            Z = (Xsub - mu) / (sd_safe + 1e-9)
            return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            raise AssertionError(f"Dim mismatch: mu/sd({d}) vs Fbase({f}). "
                                 f"{'Falta lambda_proxy' if d==f+1 else 'Revisar features.'}")

    # Construye vistas específicas para AE/CLS por longitud de contexto
    def slice_last_T(X, Twant):
        if X.shape[1] == Twant: return X
        return X[:, -Twant:, :]

    X_for_ae  = slice_last_T(X_win, T_ae)
    X_for_cls = slice_last_T(X_win, T_cls)

    Xae_n = norm_with(ae_norm,  X_for_ae)
    Xcl_n = norm_with(cls_norm, X_for_cls)

    # Tensores
    Xae = torch.from_numpy(Xae_n.astype(DT))
    Xcl = torch.from_numpy(Xcl_n.astype(DT))
    y   = torch.from_numpy(rH.astype(DT))
    if torch.cuda.is_available() and device.type == "cuda":
        Xae = Xae.pin_memory(); Xcl = Xcl.pin_memory(); y = y.pin_memory()

    # Máscaras/longitudes (no hay padding real: ventanas completas)
    N = Xae.size(0)
    lengths_ae = torch.full((N,), T_ae, dtype=torch.long)
    lengths_cl = torch.full((N,), T_cls, dtype=torch.long)
    pad_mask_ae = torch.zeros((N, T_ae), dtype=torch.bool)
    pad_mask_cl = torch.zeros((N, T_cls), dtype=torch.bool)

    # Congela AE/CLS
    ae_enc.eval(); cls_net.eval()
    for p in ae_enc.parameters(): p.requires_grad_(False)
    for p in cls_net.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def encode_batch(i, end):
        xb_ae = Xae[i:end].to(device, non_blocking=True)
        xb_cl = Xcl[i:end].to(device, non_blocking=True)
        len_ae = lengths_ae[i:end].to(device, non_blocking=True)
        len_cl = lengths_cl[i:end].to(device, non_blocking=True)
        pad_ae = pad_mask_ae[i:end].to(device, non_blocking=True)
        pad_cl = pad_mask_cl[i:end].to(device, non_blocking=True)

        z = ae_enc(xb_ae)  # [B, z_dim]
        logits, pooled = cls_net(xb_cl, len_cl, pad_cl)  # [B,3], [B,H]

        # Guardias anti-NaN/Inf en embeddings
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
        return z, logits, pooled

    # Dry run para resolver dims del policy
    z0, log0, h0 = encode_batch(0, min(64, N))
    pooled_dim = int(h0.shape[1])
    in_dim = int(z_dim + 3 + pooled_dim)

    # Policy
    if args.policy == "transformer":
        pol = PolicyTransformer(z_dim=z_dim, h_dim=pooled_dim, logit_dim=3,
                                d_model=args.pol_d_model, nhead=args.pol_nhead,
                                ff=args.pol_ff, layers=args.pol_layers,
                                dropout=args.dropout).to(device)
    else:
        pol = PolicyMLP(in_dim=in_dim, hidden=args.hidden, dropout=args.dropout).to(device)

    opt = torch.optim.AdamW(pol.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Splits
    idx_tr, idx_va, idx_te = split_indices(N, 0.70, 0.15)
    BS = args.batch_size
    best_va = -1e9; best_ep = 0; bad = 0

    def run_epoch(mode: str, sl: slice):
        pol.train(mode == "train")
        start = sl.start or 0
        stop  = sl.stop if sl.stop is not None else N
        total = 0.0; nb = 0
        pos_all = []
        with torch.set_grad_enabled(mode == "train"):
            for i in range(start, stop, BS):
                end = min(i+BS, stop)
                z, logits, pooled = encode_batch(i, end)
                if args.policy == "transformer":
                    a = pol(z, pooled, logits)
                else:
                    feat = torch.cat([z, pooled, logits], dim=-1)
                    feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
                    a = pol(feat)

                yb = y[i:end].to(device, non_blocking=True)
                yb = torch.nan_to_num(yb, nan=0.0, posinf=0.0, neginf=0.0)

                loss, _ = trading_loss(
                    a, yb,
                    loss_type=args.loss_type,
                    lambda_var=args.lambda_var,
                    tx_cost=args.tx_cost,
                    turnover_coef=args.turnover_coef,
                    l2_pos=args.l2_pos,
                    vol_norm=args.vol_norm,
                    expo_target=args.expo_target, expo_w=args.expo_w,
                    mean_pos_target=args.mean_pos_target, mean_pos_w=args.mean_pos_w
                )

                if mode == "train":
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
                    opt.step()

                total += float(loss.detach().cpu()); nb += 1
                pos_all.append(a.detach().cpu().numpy())

        # Métricas Sharpe-like offline
        pos_all = np.concatenate(pos_all) if pos_all else np.zeros(0, dtype=np.float32)
        y_np = y[start:stop].cpu().numpy()
        n = min(pos_all.size, y_np.size)
        pnl = pos_all[:n] * y_np[:n]
        mu = float(np.mean(pnl)); sd = float(np.std(pnl) + 1e-12); S = (mu / sd) if sd > 0 else 0.0
        return {"loss": total / max(1, nb), "sharpe_like": S}

    for ep in range(1, args.epochs + 1):
        tr = run_epoch("train", idx_tr)
        va = run_epoch("eval",  idx_va)
        te = run_epoch("eval",  idx_te)
        print(f"[Ep{ep:02d}] Tr S~={tr['sharpe_like']:.3f} | Va S~={va['sharpe_like']:.3f} | Te S~={te['sharpe_like']:.3f} | loss={tr['loss']:.4f}")

        improved = (va["sharpe_like"] - best_va) > args.min_delta
        if ep == 1 or improved:
            best_va = va["sharpe_like"]; best_ep = ep; bad = 0
            outdir = (world / "trained").resolve(); outdir.mkdir(parents=True, exist_ok=True)
            ckpt = outdir / f"CTRL_CTX_AECLS_T{T_run}_H{H_target}m_Z{z_dim}_IN{in_dim}_{world.name}.pt"
            torch.save({
                "pol_state": pol.state_dict(),
                "cfg": vars(args),
                "world": world.name,
                "dims": {"z_dim": int(z_dim), "pooled_dim": int(pooled_dim), "in_dim": int(in_dim), "T": int(T_run)},
                "metrics": {"train": tr, "val": va, "test": te},
                "ae": {"ckpt": str(args.ae_ckpt), "window_len": int(T_ae)},
                "cls": {"ckpt": str(args.cls_ckpt), "arch": cls_arch, "context_len": int(T_cls)},
                "horizon_min": int(H_target),
                "policy": args.policy,
                "feature_mode": "rich"
            }, ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping Ep{ep} (best Va S~={best_va:.4f} @Ep{best_ep})")
                break

    print("✔ Entrenamiento controller con contexto finalizado.")


if __name__ == "__main__":
    main()


