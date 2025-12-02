#!/usr/bin/env python
# src/models/memory/train_cls_unweighted.py
"""
Clasificación 3 clases (DOWN=0, NEUTRAL=1, UP=2) SIN ponderación de clases (unweighted).

Se mantiene:
  - Attention Pooling en LSTM/Transformer.
  - Normalización rich SIN padding (máscara por longitudes).
  - Embargo temporal >= H entre TRAIN/VAL/TEST.
  - Etiquetado VOL-AWARE (z_H = r_H / sigma_1h) o |r_H|.
  - CE con label-smoothing opcional, o Focal sin alphas ni weights.
  - Early stopping por balanced accuracy de validación.
  - Métricas: acc, balanced acc, F1-macro, MCC y matriz de confusión.
"""

import argparse, math, json
from pathlib import Path
from typing import Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

DT = np.float32

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
    for name in ["SYN_LAMBDA.npy", "HAWKES_LAMBDA.npy", "LAMBDA.npy"]:
        fn = world / name
        if fn.exists():
            try:
                arr = np.load(fn).astype(np.float64)
                return arr
            except Exception:
                return None
    return None

def load_windows(world: Path, H: int) -> np.ndarray:
    fn = world / f"SYN_WINDOWS_{H}m_X.npy"
    if not fn.exists():
        raise FileNotFoundError(f"Missing {fn}. Genera ventanas primero o usa --features rich.")
    return np.load(fn, mmap_mode="r").astype(DT)

def split_indices_purged(N: int, steps_H: int, tr=0.70, va=0.15):
    n_tr = int(tr * N); n_va = int(va * N)
    i_tr_end = n_tr
    i_va_start = min(N, i_tr_end + steps_H)
    i_va_end = min(N, i_va_start + n_va)
    i_te_start = min(N, i_va_end + steps_H)
    return slice(0, i_tr_end), slice(i_va_start, i_va_end), slice(i_te_start, N)

def compute_pct_target(price: np.ndarray, horizon_min: int, step_min: int) -> np.ndarray:
    steps = max(1, horizon_min // max(1, step_min))
    p = np.clip(price.astype(np.float64), 1e-9, None)
    rH = (p[steps:] - p[:-steps]) / p[:-steps]
    return rH.astype(np.float32)

# ───────── Features RICAS ─────────

def trailing_mean(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1: return x.copy()
    c = np.cumsum(np.insert(x, 0, 0.0))
    m = (c[w:] - c[:-w]) / w
    out = np.empty_like(x, dtype=np.float64)
    out[:w-1] = x[:w-1]
    out[w-1:] = m
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
    steps_1h = max(1, 60 // max(1, step_min))
    steps_6h = max(1, (6*60) // max(1, step_min))
    ma_fast = trailing_mean(p, steps_1h) / p
    ma_slow = trailing_mean(p, steps_6h) / p
    mom_1h  = (p - np.concatenate([p[:steps_1h], p[:-steps_1h]])) / np.concatenate([p[:steps_1h], p[:-steps_1h]])
    vol_1h  = rolling_std(ret1, steps_1h)
    vol_6h  = rolling_std(ret1, steps_6h)

    N = p.size
    period = float((24 * 60) // max(1, step_min))
    idx = np.arange(N, dtype=np.float64)
    sin_t = np.sin(2*np.pi * (idx % period) / period)
    cos_t = np.cos(2*np.pi * (idx % period) / period)

    feats = [logp, ret1, ma_fast, ma_slow, mom_1h, vol_1h, vol_6h, sin_t, cos_t]
    if lambda_proxy is not None and lambda_proxy.shape[0] == N:
        feats.append(lambda_proxy.astype(np.float64))
    X_all = np.stack(feats, axis=1)
    return np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

def build_windows_from_matrix(X_all: np.ndarray, T: int) -> Tuple[np.ndarray, np.ndarray]:
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
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.w = nn.Linear(d, 1)
        self.drop = nn.Dropout(dropout)
    def forward(self, z, pad_mask):
        scores = self.w(self.drop(z)).squeeze(-1)     # (B,T)
        scores = scores.masked_fill(pad_mask, -1e9)
        attn = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B,T,1)
        return (z * attn).sum(dim=1)                  # (B,D)

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
        z, _ = nn.utils.rnn.pad_packed_sequence(z, batch_first=True)
        h = self.attnp(z, pad_mask)
        return self.head(h)

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
        return self.head(pooled)

# ───────── Pérdidas / Métricas ─────────

def focal_loss(logits, targets, gamma=1.0, reduction="mean"):
    logp = F.log_softmax(logits, dim=1)
    p = torch.exp(logp)
    pt = p.gather(1, targets.view(-1,1)).squeeze(1)
    logpt = logp.gather(1, targets.view(-1,1)).squeeze(1)
    loss = - ((1-pt)**gamma) * logpt   # ← sin weights ni alpha
    if reduction=="mean": return loss.mean()
    if reduction=="sum":  return loss.sum()
    return loss

def confusion_matrix_np(y_true, y_pred, C=3):
    cm = np.zeros((C, C), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0<=t<C and 0<=p<C: cm[t,p]+=1
    return cm

def metrics_cls(y_true, y_pred, C=3):
    y_true = np.asarray(y_true, np.int64); y_pred = np.asarray(y_pred, np.int64)
    cm_int = confusion_matrix_np(y_true, y_pred, C)
    acc = float(np.trace(cm_int)/max(1, cm_int.sum()))
    rec, f1s = [], []
    for c in range(C):
        tp = float(cm_int[c,c]); fn = float(cm_int[c,:].sum()-cm_int[c,c]); fp = float(cm_int[:,c].sum()-cm_int[c,c])
        r = tp/max(1.0, tp+fn); p = tp/max(1.0, tp+fp); f1 = 0.0 if (p+r)==0 else (2*p*r/(p+r))
        rec.append(r); f1s.append(f1)
    bal_acc = float(np.mean(rec)); f1_macro = float(np.mean(f1s))
    cm = cm_int.astype(np.float64); t_sum = cm.sum(axis=1); p_sum = cm.sum(axis=0)
    n = cm.sum(); c = np.trace(cm); s = float(np.dot(p_sum, t_sum))
    a = float(n**2 - np.sum(p_sum**2)); b = float(n**2 - np.sum(t_sum**2))
    denom_sq = max(0.0, a*b); mcc = 0.0 if denom_sq==0 else float((c*n - s) / math.sqrt(denom_sq))
    return {"acc":acc, "bal_acc":bal_acc, "f1_macro":f1_macro, "mcc":mcc, "cm":cm_int.tolist()}

# ───────── Main ─────────

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--world", required=True)
    pa.add_argument("--horizon-min", type=int, required=True, help="H para etiquetar r_H%")
    pa.add_argument("--features", choices=["rich","windows"], default="rich",
                    help="rich = features on-the-fly; windows = usa SYN_WINDOWS_{H}m_X.npy")
    pa.add_argument("--context-len", type=int, default=256, help="T (solo en modo rich)")
    pa.add_argument("--use-lambda-proxy", choices=["auto","off"], default="auto",
                    help="Añade lambda proxy si hay fichero (auto) o lo desactiva (off)")
    pa.add_argument("--q-tau", type=float, default=0.6, help="Cuantil para tau (TRAIN)")
    pa.add_argument("--labeling", choices=["vol_aware","abs_pct"], default="vol_aware",
                    help="vol_aware usa z_H=r_H/sigma_1h(t); abs_pct usa |r_H|")

    pa.add_argument("--model", choices=["lstm","transformer"], default="lstm")
    pa.add_argument("--epochs", type=int, default=40)
    pa.add_argument("--batch-size", type=int, default=512)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--device", default="cuda:0")
    pa.add_argument("--dropout", type=float, default=0.1)
    pa.add_argument("--lr", type=float, default=3e-4)
    pa.add_argument("--weight-decay", type=float, default=1e-4)
    pa.add_argument("--label-smoothing", type=float, default=0.0)

    # pérdidas (SIN ponderación)
    pa.add_argument("--loss", choices=["ce","focal"], default="ce")
    pa.add_argument("--gamma", type=float, default=1.0)  # focal gamma

    # early stopping
    pa.add_argument("--patience", type=int, default=12)
    pa.add_argument("--min-delta", type=float, default=1e-5)
    pa.add_argument("--warmup-epochs", type=int, default=5)

    # embargo
    pa.add_argument("--embargo-steps", type=str, default="auto",
                    help="'auto' = steps de H; o número entero de pasos para el hueco entre splits")
    pa.add_argument("--deterministic", action="store_true")
    args = pa.parse_args()

    set_seed(args.seed, deterministic=args.deterministic)
    device = torch.device(args.device)
    world = Path(args.world).expanduser().resolve()
    STEP_MIN = load_step_min(world, default=5)

    price = load_price_series(world)
    r_pct_full = compute_pct_target(price, args.horizon_min, STEP_MIN)

    logp = np.log(np.clip(price.astype(np.float64), 1e-9, None))
    ret1 = np.diff(logp, prepend=logp[0])
    steps_1h = max(1, 60 // max(1, STEP_MIN))
    sig1h = rolling_std(ret1, steps_1h)
    sig1h = sig1h[:r_pct_full.shape[0]]
    sig1h = np.clip(sig1h, 1e-8, None)
    z_H_full = (r_pct_full / sig1h).astype(np.float32)

    # ---------- Features & ventanas ----------
    use_rich = (args.features == "rich")
    if not use_rich:
        H = int(args.horizon_min)
        steps_H = max(1, H // STEP_MIN)
        X_np = load_windows(world, H).astype(DT)              # (N, T, 2)
        N_eff = min(X_np.shape[0] - steps_H, r_pct_full.shape[0])
        if N_eff <= 0:
            raise SystemExit("No hay muestras tras alinear horizonte.")
        X_np = X_np[:N_eff]
        r_pct = r_pct_full[:N_eff]
        z_H = z_H_full[:N_eff]

        if args.embargo_steps.strip().lower() == "auto":
            embargo = steps_H
        else:
            embargo = max(0, int(args.embargo_steps))
        n_tr = int(0.70 * N_eff); n_va = int(0.15 * N_eff)
        i_tr_end = n_tr
        i_va_start = min(N_eff, i_tr_end + embargo)
        i_va_end = min(N_eff, i_va_start + n_va)
        i_te_start = min(N_eff, i_va_end + embargo)
        idx_tr, idx_va, idx_te = slice(0, i_tr_end), slice(i_va_start, i_va_end), slice(i_te_start, N_eff)

        Xtr = X_np[idx_tr]
        price_tr = Xtr[..., 0]; sigma_tr = Xtr[..., 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            logp_tr = np.log(np.clip(price_tr, 1e-6, None))
        m0 = float(np.nanmean(logp_tr)); s0 = float(np.nanstd(logp_tr) + 1e-9)
        m1 = float(np.nanmean(sigma_tr)); s1 = float(np.nanstd(sigma_tr) + 1e-9)
        legacy_stats = (m0, s0, m1, s1)

        Fdim = X_np.shape[2]
        pad_mask_full = None
        lengths_full = None
        rich_stats = None

    else:
        lam = maybe_load_lambda_proxy(world) if args.use_lambda_proxy == "auto" else None
        X_all = build_raw_features(price, STEP_MIN, lambda_proxy=lam)
        H = int(args.horizon_min)
        steps_H = max(1, H // STEP_MIN)
        N_eff = min(X_all.shape[0] - steps_H, r_pct_full.shape[0])
        if N_eff <= 0:
            raise SystemExit("No hay muestras tras alinear horizonte.")
        X_all = X_all[:N_eff, :]
        r_pct = r_pct_full[:N_eff]
        z_H = z_H_full[:N_eff]

        X_raw, lengths = build_windows_from_matrix(X_all, T=int(args.context_len))
        Fdim = X_raw.shape[2]

        if args.embargo_steps.strip().lower() == "auto":
            embargo = steps_H
        else:
            embargo = max(0, int(args.embargo_steps))
        n_tr = int(0.70 * X_raw.shape[0]); n_va = int(0.15 * X_raw.shape[0])
        i_tr_end = n_tr
        i_va_start = min(X_raw.shape[0], i_tr_end + embargo)
        i_va_end = min(X_raw.shape[0], i_va_start + n_va)
        i_te_start = min(X_raw.shape[0], i_va_end + embargo)
        idx_tr, idx_va, idx_te = slice(0, i_tr_end), slice(i_va_start, i_va_end), slice(i_te_start, X_raw.shape[0])

        Tctx = X_raw.shape[1]
        valid = np.zeros((X_raw.shape[0], Tctx), dtype=bool)
        for i, L in enumerate(lengths):
            if L > 0:
                valid[i, Tctx - int(L):] = True
        valid_tr = valid[idx_tr]
        mask3 = np.repeat(valid_tr[:, :, None], Fdim, axis=2)
        Xtr_valid = X_raw[idx_tr][mask3].reshape(-1, Fdim).astype(np.float64)
        mu = np.mean(Xtr_valid, axis=0)
        sd = np.std(Xtr_valid, axis=0) + 1e-9
        X_np = (X_raw - mu.astype(DT)) / sd.astype(DT)

        pad_mask_full = ~valid
        lengths_full = lengths
        rich_stats = {"mu": mu.astype(np.float32), "sd": sd.astype(np.float32)}

    # ---------- Etiquetado ----------
    base_for_tau = z_H if args.labeling == "vol_aware" else r_pct
    tau = float(np.quantile(np.abs(base_for_tau[idx_tr]), args.q_tau))
    def labelize(v, t):
        return 0 if v < -t else (2 if v > t else 1)
    y_cls = np.array([labelize(v, tau) for v in (z_H if args.labeling == "vol_aware" else r_pct)],
                     dtype=np.int64)

    # Distribución por split (info)
    def dist(slice_):
        sl = range(slice_.start or 0, slice_.stop if slice_.stop is not None else len(y_cls))
        cnt = Counter(y_cls[sl]); total = max(1, len(list(sl)))
        return {int(k): float(cnt.get(k,0)/total) for k in [0,1,2]}

    # Tensores base
    X_t = torch.from_numpy(np.asarray(X_np, np.float32))
    y_t = torch.from_numpy(y_cls.astype(np.int64))
    device_is_cuda = (torch.device(args.device).type == "cuda")
    if device_is_cuda:
        X_t = X_t.pin_memory(); y_t = y_t.pin_memory()

    if use_rich:
        pad_mask_full_t = torch.from_numpy(pad_mask_full)          # (N,T)
        lengths_full_t = torch.from_numpy(lengths_full.astype(np.int64))

    # ---------- Modelo ----------
    if args.model == "lstm":
        net = LSTMClassifier(feat_dim=Fdim, hidden=128, layers=2,
                             dropout=args.dropout, num_classes=3).to(args.device)
        model_cfg = {"input_dim": int(Fdim), "hidden_dim": 128, "num_layers": 2, "dropout": args.dropout}
        model_name = "LSTM"
    else:
        net = TransformerClassifier(feat_dim=Fdim, d_model=128, nhead=4, ff=256, layers=3,
                                    dropout=args.dropout, num_classes=3).to(args.device)
        model_cfg = {"input_dim": int(Fdim), "d_model": 128, "nhead": 4,
                     "dim_feedforward": 256, "num_layers": 3, "dropout": args.dropout}
        model_name = "TRANSFORMER"

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---------- Paths / logging ----------
    outdir = (world / "trained").resolve(); outdir.mkdir(parents=True, exist_ok=True)
    feat_tag = "richT{}".format(int(args.context_len)) if use_rich else "windows"
    lab_tag = "zH" if args.labeling == "vol_aware" else "pct"
    loss_tag = ("lossCE" if args.loss=="ce" else f"lossFocalG{int(args.gamma)}")
    if args.loss == "ce" and args.label_smoothing > 0.0:
        loss_tag += f"S{args.label_smoothing:g}"
    ckpt = outdir / f"CLS_{model_name}_{feat_tag}_{lab_tag}_{loss_tag}_UNWEIGHTED_H{int(args.horizon_min)}m_{world.name}.pt"

    print(f"\n=== Train CLS-unweighted ({args.model}) world={world.name} H={int(args.horizon_min)}m ===")
    print(f"Device={args.device} | step_min={STEP_MIN} | features={args.features} | F={Fdim}")
    if use_rich and rich_stats is not None:
        has_lambda = (maybe_load_lambda_proxy(world) is not None) and (args.use_lambda_proxy == 'auto')
        print(f"Rich feats: T={int(args.context_len)} | lambda_proxy={'ON' if has_lambda else 'OFF'}")
    print(f"Labeling={args.labeling} | Tau(q={args.q_tau:g}) ≈ {tau:.6g} | "
          f"Train dist {dist(idx_tr)} | Val {dist(idx_va)} | Test {dist(idx_te)}")
    print(f"Loss={args.loss} (sin ponderación) | label_smoothing={args.label_smoothing:g}")

    BS = args.batch_size

    @torch.no_grad()
    def eval_split_fn(idx_slice):
        net.eval()
        start = idx_slice.start or 0
        stop = idx_slice.stop if idx_slice.stop is not None else X_t.size(0)
        preds, ys = [], []
        for i in range(start, stop, BS):
            end = min(i + BS, stop)
            xb = X_t[i:end]
            yb = y_t[i:end].to(args.device, non_blocking=True)

            if use_rich:
                pad_b = pad_mask_full_t[i:end]
                len_b = lengths_full_t[i:end]
                x = xb.to(args.device, non_blocking=True)
                pad_mask = pad_b.to(args.device, non_blocking=True)
                lengths = len_b.to(args.device, non_blocking=True)
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
                x = torch.nan_to_num(x, nan=0.0).to(args.device, non_blocking=True)
                lengths = (~pad_mask).sum(dim=1).to(args.device)

            logits = net(x, lengths, pad_mask)
            yhat = torch.argmax(logits, dim=1)
            preds.append(yhat.cpu().numpy()); ys.append(yb.cpu().numpy())

        if not preds:
            return {"acc":0.0, "bal_acc":0.0, "f1_macro":0.0, "mcc":0.0, "cm":[[0]*3]*3}
        y_pred = np.concatenate(preds); y_true = np.concatenate(ys)
        return metrics_cls(y_true, y_pred, C=3)

    # ---- Entrenamiento ----
    best_va = -1e9; best_ep = 0; bad = 0
    for ep in range(1, args.epochs + 1):
        net.train()
        perm = torch.randperm(idx_tr.stop - (idx_tr.start or 0)) + (idx_tr.start or 0)
        total_loss = 0.0; nb = 0; pred_hist = np.zeros(3, dtype=np.int64)

        for i in range(idx_tr.start or 0, idx_tr.stop, BS):
            b_idx = perm[i - (idx_tr.start or 0): i - (idx_tr.start or 0) + BS]
            xb = X_t[b_idx]
            yb = y_t[b_idx].to(args.device, non_blocking=True)

            if use_rich:
                pad_b = pad_mask_full_t[b_idx]
                len_b = lengths_full_t[b_idx]
                x = xb.to(args.device, non_blocking=True)
                pad_mask = pad_b.to(args.device, non_blocking=True)
                lengths = len_b.to(args.device, non_blocking=True)
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
                x = torch.nan_to_num(x, nan=0.0).to(args.device, non_blocking=True)
                lengths = (~pad_mask).sum(dim=1).to(args.device)

            logits = net(x, lengths, pad_mask)

            if args.loss == "focal":
                loss = focal_loss(logits, yb, gamma=args.gamma, reduction="mean")
            else:
                loss = F.cross_entropy(logits, yb, weight=None,
                                       label_smoothing=float(args.label_smoothing))

            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); total_loss += float(loss.detach().cpu()); nb += 1

            with torch.no_grad():
                yhat = torch.argmax(logits, dim=1).cpu().numpy()
                for k in yhat: pred_hist[int(k)] += 1

        va = eval_split_fn(idx_va)
        te = eval_split_fn(idx_te)
        tr = eval_split_fn(idx_tr)

        ph = pred_hist.sum()
        ph_str = f"pred%: [{(100*pred_hist[0]/max(1,ph)):.1f}, {(100*pred_hist[1]/max(1,ph)):.1f}, {(100*pred_hist[2]/max(1,ph)):.1f}]"
        print(f"[Ep{ep:02d}] Tr acc={tr['acc']:.3f} bal={tr['bal_acc']:.3f} f1M={tr['f1_macro']:.3f} | "
              f"Va acc={va['acc']:.3f} bal={va['bal_acc']:.3f} f1M={va['f1_macro']:.3f} | "
              f"Te acc={te['acc']:.3f} bal={te['bal_acc']:.3f} f1M={te['f1_macro']:.3f} | "
              f"loss={total_loss/max(1,nb):.4f} | {ph_str}")

        improved = (va["bal_acc"] - best_va) > args.min_delta
        if ep <= args.warmup_epochs:
            if va["bal_acc"] > best_va:
                best_va = va["bal_acc"]; best_ep = ep; bad = 0
                torch.save({
                    "state_dict": net.state_dict(),
                    "arch": args.model, "model_cfg": model_cfg,
                    "horizon_min": int(args.horizon_min), "variant": f"cls3_unweighted_{args.features}",
                    "world": world.name, "tau": float(tau),
                    "class_map": {"DOWN":0,"NEUTRAL":1,"UP":2},
                    "feature_mode": args.features,
                    "feature_norm": (rich_stats if use_rich else {
                        "legacy_logp_mean": float(legacy_stats[0]), "legacy_logp_std": float(legacy_stats[1]),
                        "legacy_sigma_mean": float(legacy_stats[2]), "legacy_sigma_std": float(legacy_stats[3])
                    }),
                    "train_dist": dist(idx_tr), "val": va, "test": te,
                    "loss_cfg": {"loss":args.loss, "gamma":args.gamma,
                                 "label_smoothing": float(args.label_smoothing)},
                    "context_len": (int(args.context_len) if use_rich else None),
                    "labeling": args.labeling
                }, ckpt)
        else:
            if improved:
                best_va = va["bal_acc"]; best_ep = ep; bad = 0
                torch.save({
                    "state_dict": net.state_dict(),
                    "arch": args.model, "model_cfg": model_cfg,
                    "horizon_min": int(args.horizon_min), "variant": f"cls3_unweighted_{args.features}",
                    "world": world.name, "tau": float(tau),
                    "class_map": {"DOWN":0,"NEUTRAL":1,"UP":2},
                    "feature_mode": args.features,
                    "feature_norm": (rich_stats if use_rich else {
                        "legacy_logp_mean": float(legacy_stats[0]), "legacy_logp_std": float(legacy_stats[1]),
                        "legacy_sigma_mean": float(legacy_stats[2]), "legacy_sigma_std": float(legacy_stats[3])
                    }),
                    "train_dist": dist(idx_tr), "val": va, "test": te,
                    "loss_cfg": {"loss":args.loss, "gamma":args.gamma,
                                 "label_smoothing": float(args.label_smoothing)},
                    "context_len": (int(args.context_len) if use_rich else None),
                    "labeling": args.labeling
                }, ckpt)
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"Early stopping Ep{ep} (best Va bal_acc={best_va:.4f} @Ep{best_ep})")
                    break

    print(f"\n✔ Checkpoint guardado: {ckpt}")
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

