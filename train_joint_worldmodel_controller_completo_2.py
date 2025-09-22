#!/usr/bin/env python
# scripts_nuevos/train_joint_worldmodel_controller_completo.py
"""
Entrenamiento conjunto Controller + World Model + (opcional) Clasificador para series sintéticas 24/7.

- WM: LSTM-AE | MDN-RNN (con imaginación) | RSSM (con imaginación)
- CLS (opcional con --with-cls): LSTM/Transformer con Attention Pooling
- Controller: MLP o Transformer (tokeniza vector de entrada en chunks + [CLS])
- Features 100% retrospectivas (sin fuga) con normalización de TRAIN.
  Opcional 'lambda_proxy' si hay fichero compatible.

Checkpoint:
  trained/JOINT_<WM>_CTRL_H{H}m_T{T}_Z{Z}_{world}.pt
"""

import argparse, math, json
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DT = np.float32
EPS = 1e-9
LOG2PI = math.log(2.0 * math.pi)

# ───────── Utilidades ─────────

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
    for name in ["SYN_PRICE.npy", "SYN_SERIE_5m.npy", "PRICE_5m.npy", "SERIE_5m.npy"]:
        fn = world / name
        if fn.exists():
            return np.load(fn).astype(np.float64)
    raise FileNotFoundError(f"No se encontró serie de precio en {world}")

def load_step_min(world: Path, default=5) -> int:
    for fn in ["SYN_META.json", "meta.json"]:
        p = world / fn
        if p.exists():
            try:
                meta = json.loads(p.read_text())
                v = int(meta.get("step_min", meta.get("dt_min", default)))
                return v if v > 0 else default
            except Exception:
                return default
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

# ───────── Dataset con ventanas + longitudes ─────────

def trailing_mean(x, w):
    if w <= 1: return x.copy()
    c = np.cumsum(np.insert(x, 0, 0.0))
    m = (c[w:] - c[:-w]) / w
    out = np.empty_like(x, dtype=np.float64)
    out[:w-1] = x[:w-1]
    out[w-1:] = m
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

def build_raw_features(price: np.ndarray, step_min: int, lambda_proxy: Optional[np.ndarray]=None) -> np.ndarray:
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
    day_period = float((24 * 60) // max(1, step_min))
    idx = np.arange(N, dtype=np.float64)
    sin_t = np.sin(2*np.pi * (idx % day_period) / day_period)
    cos_t = np.cos(2*np.pi * (idx % day_period) / day_period)

    feats = [logp, ret1, ma_fast, ma_slow, mom_1h, vol_1h, vol_6h, sin_t, cos_t]
    if lambda_proxy is not None and lambda_proxy.shape[0] == N:
        feats.append(lambda_proxy.astype(np.float64))

    X_all = np.stack(feats, axis=1)
    return np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

def make_windows_from_matrix(X_all: np.ndarray, T: int):
    N, F = X_all.shape
    X = np.zeros((N, T, F), dtype=DT)
    lengths = np.zeros((N,), dtype=np.int32)
    pad_mask = np.zeros((N, T), dtype=bool)
    for i in range(N):
        s = i - T + 1
        if s < 0:
            L = i + 1
            X[i, -L:, :] = X_all[:i+1, :]
            pad_mask[i, :T-L] = True
            lengths[i] = L
        else:
            X[i, :, :] = X_all[s:i+1, :]
            lengths[i] = T
    return X, lengths, pad_mask

def build_dataset(price: np.ndarray, step_min: int, horizon_min: int, window_len: int,
                  lambda_proxy: Optional[np.ndarray]=None):
    """
    Devuelve:
      X_win: [N_eff, T, F] (z-score con stats de TRAIN)
      rH   : [N_eff]
      lengths: [N_eff] longitudes reales de ventana
      pad_mask: [N_eff, T] True en padding
      norm : estadísticas de normalización
    """
    p = np.clip(price.astype(np.float64), 1e-9, None)
    steps_H = max(1, horizon_min // max(1, step_min))
    rH = (p[steps_H:] - p[:-steps_H]) / p[:-steps_H]
    N_eff = rH.size

    X_raw_all = build_raw_features(price, step_min=step_min, lambda_proxy=lambda_proxy)
    X_raw = X_raw_all[:N_eff]
    X_win_raw, lengths, pad_mask = make_windows_from_matrix(X_raw, T=int(window_len))

    idx_tr, _, _ = split_indices(N_eff, 0.70, 0.15)
    mu = np.mean(X_raw[idx_tr], axis=0)
    sd = np.std(X_raw[idx_tr], axis=0) + 1e-9
    Xn = (X_win_raw - mu) / sd

    feat_names = ["logp","ret1","ma1h/p","ma6h/p","mom_1h","vol_1h","vol_6h","sin_t","cos_t"]
    if lambda_proxy is not None and lambda_proxy.shape[0] >= N_eff:
        feat_names.append("lambda_proxy")

    norm = {
        "mu": mu.astype(np.float32).tolist(),
        "sd": sd.astype(np.float32).tolist(),
        "features": feat_names
    }
    return Xn.astype(DT), rH.astype(DT), lengths.astype(np.int64), pad_mask, norm

# ───────── Policies ─────────

class PolicyMLP(nn.Module):
    def __init__(self, z_dim, hidden=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Linear(z_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, z): return torch.tanh(self.net(z)).squeeze(-1)

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

# ───────── Trading / métricas ─────────

def trading_loss(positions, r_future, *,
                 loss_type="meanvar",
                 lambda_var=3.0,
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
    mu = pnl.mean(); var = pnl.var(unbiased=False); sd = var.sqrt() + 1e-12
    base_obj = (mu/sd) if loss_type == "sharpe" else (mu - lambda_var * var)
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
        "var": float(var.detach().cpu()),
        "sd": float(sd.detach().cpu()),
        "sharpe_like": float((mu/sd).detach().cpu()),
        "turnover": float(turnover.detach().cpu()),
        "cost": float(cost.detach().cpu()),
        "a_mean": float(a.mean().detach().cpu()),
        "a_abs_mean": float(a.abs().mean().detach().cpu()),
    }
    return loss, terms

def eval_metrics(positions: np.ndarray, r_future: np.ndarray) -> Dict[str,float]:
    n = min(len(positions), len(r_future))
    positions = positions[:n]; r_future = r_future[:n]
    pnl = positions * r_future
    mu = pnl.mean(); sd = pnl.std() + 1e-12
    sharpe = mu / sd
    hit = np.mean(np.sign(positions) == np.sign(r_future))
    return {"mu": float(mu), "sd": float(sd), "sharpe_like": float(sharpe), "hit_rate": float(hit)}

# ───────── World Models ─────────
class BaseWM(nn.Module):
    def __init__(self): super().__init__()
    def encode_window(self, x): raise NotImplementedError
    def recon_loss(self, x): return x.new_tensor(0.0)
    def aux_reward(self, z): return None
    def supports_imagination(self): return False
    def imagine_step(self, state): raise NotImplementedError

class LSTMMemory(BaseWM):
    def __init__(self, in_dim, hidden=128, layers=2, z_dim=64, dropout=0.1, aux_head=True):
        super().__init__()
        self.enc = nn.LSTM(in_dim, hidden, layers, batch_first=True,
                           dropout=(dropout if layers>1 else 0.0))
        self.to_z = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, z_dim), nn.Tanh())
        self.dec = nn.LSTM(in_dim+z_dim, hidden, layers, batch_first=True,
                           dropout=(dropout if layers>1 else 0.0))
        self.dec_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, in_dim))
        self.aux = (nn.Sequential(
            nn.LayerNorm(z_dim), nn.Linear(z_dim, z_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(z_dim, 1)
        ) if aux_head else None)

    def encode_window(self, x):
        _, (hN, _) = self.enc(x); h = hN[-1]; z = self.to_z(h)
        return z

    def forward(self, x):
        z = self.encode_window(x)
        B,T,Fdim = x.size()
        zt = z.unsqueeze(1).repeat(1,T,1)
        y,_ = self.dec(torch.cat([x, zt], dim=-1))
        x_hat = self.dec_out(y)
        r_hat = self.aux(z).squeeze(-1) if self.aux is not None else None
        return z, x_hat, r_hat

    def recon_loss(self, x):
        _, x_hat, _ = self.forward(x)
        return F.smooth_l1_loss(x_hat, x, reduction="mean")

    def aux_reward(self, z):
        return self.aux(z).squeeze(-1) if self.aux is not None else None

class MDNRNN(BaseWM):
    def __init__(self, in_dim, hidden=128, layers=1, z_dim=64, dropout=0.1, n_mix=5, aux_head=True):
        super().__init__()
        self.in_dim = in_dim; self.hidden = hidden; self.layers = max(1, layers); self.n_mix = n_mix
        self.rnn = nn.LSTM(in_dim, hidden, num_layers=self.layers, batch_first=True,
                           dropout=(dropout if self.layers>1 else 0.0))
        self.to_z = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, z_dim), nn.Tanh())
        out_dim = in_dim * n_mix * 3  # (logits, mu, log_sigma)
        self.mdn_head = nn.Linear(hidden, out_dim)
        self.aux = (nn.Sequential(nn.LayerNorm(z_dim), nn.Linear(z_dim, z_dim), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(z_dim, 1))
                    if aux_head else None)

    def _params_from_h_seq(self, h_seq):
        B,T,H = h_seq.size()
        raw = self.mdn_head(h_seq).view(B, T, self.in_dim, self.n_mix, 3)
        logit = raw[..., 0]; mu = raw[..., 1]; log_sigma = raw[..., 2]
        pi = torch.softmax(logit, dim=-1)
        return pi, mu, log_sigma

    def _params_from_h_step(self, h_step):
        raw = self.mdn_head(h_step).view(h_step.size(0), self.in_dim, self.n_mix, 3)
        logit = raw[..., 0]; mu = raw[..., 1]; log_sigma = raw[..., 2]
        pi = torch.softmax(logit, dim=-1)
        return pi, mu, log_sigma

    def _nll_next(self, x):
        h_seq, _ = self.rnn(x)
        if x.size(1) < 2:
            return x.new_tensor(0.0)
        pi, mu, log_sigma = self._params_from_h_seq(h_seq[:, :-1, :])
        target = x[:, 1:, :].unsqueeze(-1)
        var = torch.exp(2.0 * log_sigma)
        log_comp = -0.5 * ((target - mu)**2 / (var + 1e-6)) - log_sigma - 0.5 * LOG2PI
        log_mix = torch.log(pi + 1e-8) + log_comp
        log_prob = torch.logsumexp(log_mix, dim=-1)
        nll = -(log_prob.sum(dim=-1)).mean()
        return nll

    def encode_window(self, x):
        _, (hN, _) = self.rnn(x)
        return self.to_z(hN[-1])

    def recon_loss(self, x): return self._nll_next(x)

    def aux_reward(self, z):
        return self.aux(z).squeeze(-1) if self.aux is not None else None

    def supports_imagination(self): return True

    @torch.no_grad()
    def imagine_step(self, state):
        hN, cN, x_prev = state["hN"], state["cN"], state["x_prev"]
        y_step, (hN1, cN1) = self.rnn(x_prev.unsqueeze(1), (hN, cN))
        h_step = y_step[:, -1, :]
        pi, mu, log_sigma = self._params_from_h_step(h_step)
        B, Fdim, K = pi.size()
        cat = torch.distributions.Categorical(probs=pi.view(B*Fdim, K))
        idx = cat.sample().view(B, Fdim, 1)
        mu_sel = mu.gather(-1, idx).squeeze(-1)
        std_sel = torch.exp(log_sigma.gather(-1, idx).squeeze(-1))
        eps = torch.randn_like(std_sel)
        x_next = mu_sel + std_sel * eps
        y_step2, (hN2, cN2) = self.rnn(x_next.unsqueeze(1), (hN1, cN1))
        z_ctrl = self.to_z(hN2[-1])
        new_state = {"hN": hN2, "cN": cN2, "x_prev": x_next}
        return new_state, z_ctrl

    def mdn_roll(self, x):
        h_seq, (hN, cN) = self.rnn(x)
        nll = self._nll_next(x)
        z_ctrl = self.to_z(hN[-1])
        state = {"hN": hN.detach(), "cN": cN.detach(), "x_prev": x[:, -1, :].detach()}
        return nll, z_ctrl, state

class RSSM(BaseWM):
    def __init__(self, obs_dim: int, deter: int = 128, stoch: int = 32,
                 min_std: float = 0.1, dropout: float = 0.0,
                 kl_scale: float = 1.0, aux_head: bool = False, z_out: int = None):
        super().__init__()
        self.obs_dim = obs_dim; self.deter = deter; self.stoch = stoch
        self.min_std = float(min_std); self.kl_scale = float(kl_scale)

        self.obs_enc = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(128, 128), nn.ReLU())
        self.obs_dec = nn.Sequential(nn.Linear(deter + stoch, 128), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(128, obs_dim))
        self.gru = nn.GRUCell(input_size=stoch, hidden_size=deter)
        self.prior_net = nn.Sequential(nn.Linear(deter, 128), nn.ReLU())
        self.prior_head = nn.Linear(128, 2 * stoch)
        self.post_net = nn.Sequential(nn.Linear(deter + 128, 128), nn.ReLU())
        self.post_head = nn.Linear(128, 2 * stoch)

        self.to_z = nn.Identity() if (z_out is None or z_out == deter + stoch) \
                    else nn.Sequential(nn.Linear(deter + stoch, z_out), nn.Tanh())
        self.aux = (nn.Sequential(nn.LayerNorm(z_out or (deter + stoch)),
                                  nn.Linear(z_out or (deter + stoch), 128), nn.ReLU(),
                                  nn.Linear(128, 1))
                    if aux_head else None)

    @staticmethod
    def _split_stats(stats):
        mu, pre_std = stats.chunk(2, dim=-1)
        std = F.softplus(pre_std) + 1e-3
        return mu, std

    def prior(self, h_prev, z_prev):
        h_t = self.gru(z_prev, h_prev)
        mu_p, std_p = self._split_stats(self.prior_head(self.prior_net(h_t)))
        return mu_p, std_p, h_t

    def posterior(self, h_t, x_t):
        emb_x = self.obs_enc(x_t)
        mu_q, std_q = self._split_stats(self.post_head(self.post_net(torch.cat([h_t, emb_x], dim=-1))))
        return mu_q, std_q

    @staticmethod
    def kl_normal(mu_q, std_q, mu_p, std_p):
        eps = 1e-8
        var_q = std_q.pow(2) + eps; var_p = std_p.pow(2) + eps
        term = (var_q / var_p) + ((mu_p - mu_q).pow(2) / var_p) - 1.0 + (2.0 * (std_p.log() - std_q.log()))
        kl = 0.5 * term.sum(dim=-1)
        return kl

    def rssm_roll(self, x):
        B, T, Fdim = x.size()
        device = x.device
        h = torch.zeros(B, self.deter, device=device)
        z = torch.zeros(B, self.stoch, device=device)

        rec_sum = x.new_tensor(0.0); kl_sum = x.new_tensor(0.0)

        for t in range(T):
            mu_p, std_p, h = self.prior(h, z)
            emb_x = x[:, t, :]
            mu_q, std_q = self.posterior(h, emb_x)
            eps = torch.randn_like(mu_q)
            z = mu_q + std_q * eps
            x_mu = self.obs_dec(torch.cat([h, z], dim=-1))
            rec_sum = rec_sum + F.mse_loss(x_mu, x[:, t, :], reduction="mean")
            kl_sum = kl_sum + self.kl_scale * self.kl_normal(mu_q, std_q, mu_p, std_p).mean()

        rec_mean = rec_sum / max(1, T); kl_mean = kl_sum / max(1, T)
        z_ctrl = self.to_z(torch.cat([h, z], dim=-1))
        state = {"h": h, "z": z}
        return rec_mean, kl_mean, z_ctrl, state

    def encode_window(self, x):
        with torch.no_grad():
            _, _, z_ctrl, _ = self.rssm_roll(x)
        return z_ctrl

    def recon_loss(self, x):
        rec, kl, _, _ = self.rssm_roll(x)
        return rec + kl

    def aux_reward(self, z_ctrl):
        return self.aux(z_ctrl).squeeze(-1) if self.aux is not None else None

    def supports_imagination(self): return True

    @torch.no_grad()
    def imagine_step(self, state):
        h, z = state["h"], state["z"]
        mu_p, std_p, h_next = self.prior(h, z)
        eps = torch.randn_like(std_p)
        z_next = mu_p + eps * std_p
        z_ctrl = self.to_z(torch.cat([h_next, z_next], dim=-1))
        new_state = {"h": h_next, "z": z_next}
        return new_state, z_ctrl

# ───────── Classifier opcional ─────────

class AttnPool(nn.Module):
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.w = nn.Linear(d, 1)
        self.drop = nn.Dropout(dropout)
    def forward(self, z, pad_mask):
        scores = self.w(self.drop(z)).squeeze(-1)
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
        z, _ = nn.utils.rnn.pad_packed_sequence(z, batch_first=True)
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

def labelize_vec(r_pct: np.ndarray, tau: float) -> np.ndarray:
    y = np.zeros_like(r_pct, dtype=np.int64)
    y[r_pct >  tau] = 2
    y[np.abs(r_pct) <= tau] = 1
    y[r_pct < -tau] = 0
    return y

# ───────── Main ─────────

def parse_vec(s: str) -> List[float]:
    toks = [t.strip() for t in s.split(",") if t.strip()!=""]
    return [float(t) for t in toks]

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--world", required=True)
    pa.add_argument("--wm-type", choices=["lstm_ae","mdn_rnn","rssm"], default="lstm_ae")
    # Datos
    pa.add_argument("--horizon-min", type=int, default=60)
    pa.add_argument("--step-min", type=int, default=None)
    pa.add_argument("--window-len", type=int, default=256)
    pa.add_argument("--use-lambda-proxy", choices=["auto","off"], default="auto")

    # WM dims
    pa.add_argument("--z-dim", type=int, default=64)
    pa.add_argument("--wm-hidden", type=int, default=128)
    pa.add_argument("--wm-layers", type=int, default=2)
    pa.add_argument("--wm-dropout", type=float, default=0.1)
    pa.add_argument("--mdn-mixtures", type=int, default=5)
    pa.add_argument("--rssm-deter", type=int, default=128)
    pa.add_argument("--rssm-stoch", type=int, default=32)
    pa.add_argument("--kl-scale", type=float, default=1.0)

    # CLS conjunto (opcional)
    pa.add_argument("--with-cls", action="store_true")
    pa.add_argument("--cls-arch", choices=["lstm","transformer"], default="transformer")
    pa.add_argument("--cls-dropout", type=float, default=0.1)
    pa.add_argument("--cls-loss", choices=["ce","focal"], default="ce")
    pa.add_argument("--cls-gamma", type=float, default=1.0)
    pa.add_argument("--cls-focal-alpha", type=str, default="0.3,0.5,0.3")
    pa.add_argument("--q-tau", type=float, default=0.5)
    pa.add_argument("--neutral-weight-mult", type=float, default=1.0)
    pa.add_argument("--class-weights", type=str, default="auto")
    pa.add_argument("--cls-weight", type=float, default=0.5)

    # Pérdidas WM/aux
    pa.add_argument("--aux-weight", type=float, default=0.5)
    pa.add_argument("--rec-weight", type=float, default=1.0)

    # Policy
    pa.add_argument("--policy", choices=["mlp","transformer"], default="transformer")
    pa.add_argument("--pol-hidden", type=int, default=192)
    pa.add_argument("--pol-dropout", type=float, default=0.05)
    pa.add_argument("--pol-d-model", type=int, default=192)
    pa.add_argument("--pol-nhead", type=int, default=4)
    pa.add_argument("--pol-ff", type=int, default=384)
    pa.add_argument("--pol-layers", type=int, default=2)
    pa.add_argument("--pol-tok-dim", type=int, default=32)

    # Trading obj
    pa.add_argument("--loss-type", choices=["meanvar","sharpe"], default="meanvar")
    pa.add_argument("--lambda-var", type=float, default=3.0)
    pa.add_argument("--tx-cost", type=float, default=0.0001)
    pa.add_argument("--turnover-coef", type=float, default=0.2)
    pa.add_argument("--l2-pos", type=float, default=0.0)
    pa.add_argument("--vol-norm", action="store_true")
    pa.add_argument("--expo-target", type=float, default=None)
    pa.add_argument("--expo-w", type=float, default=0.0)
    pa.add_argument("--mean-pos-target", type=float, default=0.0)
    pa.add_argument("--mean-pos-w", type=float, default=0.0)

    # Imagination
    pa.add_argument("--imagine-steps", type=int, default=0)
    pa.add_argument("--imagine-weight", type=float, default=0.5)

    # Train
    pa.add_argument("--epochs", type=int, default=40)
    pa.add_argument("--batch-size", type=int, default=512)
    pa.add_argument("--lr", type=float, default=3e-4)
    pa.add_argument("--weight-decay", type=float, default=1e-4)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--device", default="cuda:0")
    pa.add_argument("--deterministic", action="store_true")

    # Early stopping
    pa.add_argument("--patience", type=int, default=8)
    pa.add_argument("--min-delta", type=float, default=1e-4)
    pa.add_argument("--warmup-epochs", type=int, default=3)

    args = pa.parse_args()

    set_seed(args.seed, deterministic=args.deterministic)
    device = torch.device(args.device)

    world = Path(args.world).expanduser().resolve()
    step_min = load_step_min(world, default=5) if args.step_min is None else int(args.step_min)

    price = load_price_series(world)
    lam = maybe_load_lambda_proxy(world) if args.use_lambda_proxy == "auto" else None
    X_np, rH_np, lengths_np, pad_mask_np, norm = build_dataset(price, step_min, args.horizon_min, args.window_len,
                                                               lambda_proxy=lam)
    N,T,Fdim = X_np.shape
    idx_tr, idx_va, idx_te = split_indices(N, 0.70, 0.15)

    X = torch.from_numpy(X_np.astype(DT))
    y = torch.from_numpy(rH_np.astype(DT))
    lengths_t = torch.from_numpy(lengths_np.astype(np.int64))
    pad_mask_t = torch.from_numpy(pad_mask_np)
    if device.type == "cuda":
        X = X.pin_memory(); y = y.pin_memory()

    # ─ WM
    if args.wm_type == "lstm_ae":
        wm = LSTMMemory(Fdim, args.wm_hidden, args.wm_layers, args.z_dim, args.wm_dropout,
                        aux_head=(args.aux_weight > 0.0)).to(device)
        z_dim_for_policy = args.z_dim
    elif args.wm_type == "mdn_rnn":
        wm = MDNRNN(Fdim, hidden=args.wm_hidden, layers=max(1, args.wm_layers),
                    z_dim=args.z_dim, dropout=args.wm_dropout, n_mix=args.mdn_mixtures,
                    aux_head=(args.aux_weight > 0.0)).to(device)
        z_dim_for_policy = args.z_dim
    else:  # rssm
        wm = RSSM(Fdim, deter=args.rssm_deter, stoch=args.rssm_stoch,
                  dropout=args.wm_dropout, kl_scale=args.kl_scale,
                  aux_head=(args.aux_weight > 0.0), z_out=args.z_dim).to(device)
        z_dim_for_policy = args.z_dim

    # ─ Clasificador opcional
    use_cls = bool(args.with_cls)
    if use_cls:
        if args.cls_arch == "lstm":
            cls = LSTMClassifier(feat_dim=Fdim, hidden=128, layers=2,
                                 dropout=args.cls_dropout, num_classes=3).to(device)
        else:
            cls = TransformerClassifier(feat_dim=Fdim, d_model=128, nhead=4, ff=256, layers=3,
                                        dropout=args.cls_dropout, num_classes=3).to(device)
        tau = float(np.quantile(np.abs(rH_np[idx_tr]), args.q_tau))
        y_cls_np = labelize_vec(rH_np, tau)
        y_c = torch.from_numpy(y_cls_np.astype(np.int64))
        if device.type == "cuda": y_c = y_c.pin_memory()

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
    else:
        cls = None
        y_c = None

    # ─ Policy
    pol_in = z_dim_for_policy + (3 if use_cls else 0)
    if args.policy == "transformer":
        pol = PolicyTransformer(
            in_dim=pol_in,
            d_model=args.pol_d_model, nhead=args.pol_nhead,
            ff=args.pol_ff, layers=args.pol_layers,
            tok_dim=args.pol_tok_dim, dropout=args.pol_dropout
        ).to(device)
    else:
        pol = PolicyMLP(pol_in, hidden=args.pol_hidden, dropout=args.pol_dropout).to(device)

    # Optim
    params = list(wm.parameters()) + list(pol.parameters())
    if use_cls: params += list(cls.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    outdir = (world / "trained").resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt = outdir / f"JOINT_{args.wm_type.upper()}_CTRL_H{args.horizon_min}m_T{args.window_len}_Z{args.z_dim}_{world.name}.pt"

    print(f"\n=== Train Joint WM({args.wm_type}) + {'CLS+' if use_cls else ''}Controller={args.policy.upper()} world={world.name} H={args.horizon_min}m ===")
    print(f"Device: {device} | step_min={step_min} | N={N} T={T} F={Fdim}")
    print(f"lambda_proxy={'ON' if lam is not None else 'OFF'} | imagine_steps={args.imagine_steps}")

    BS = args.batch_size
    best_va = -1e9; best_ep = 0; bad = 0

    def run_epoch(mode: str, sl: slice):
        if mode=="train":
            wm.train(); pol.train(); 
            if use_cls: cls.train()
        else:
            wm.eval(); pol.eval(); 
            if use_cls: cls.eval()

        start = sl.start or 0
        stop = sl.stop if sl.stop is not None else X.size(0)
        total_loss = 0.0; nb = 0
        positions_all = []
        wm_rec_sum = 0.0; wm_aux_sum = 0.0; wm_kl_sum = 0.0; dream_sum = 0.0
        cls_sum = 0.0

        with torch.set_grad_enabled(mode=="train"):
            for i in range(start, stop, BS):
                end = min(i+BS, stop)
                xb = X[i:end].to(device, non_blocking=True)
                yb = y[i:end].to(device, non_blocking=True)
                len_b = lengths_t[i:end].to(device, non_blocking=True)
                pad_b = pad_mask_t[i:end].to(device, non_blocking=True)

                # --- World Model ---
                if args.wm_type == "rssm":
                    rec, kl, z_ctrl, state = wm.rssm_roll(xb)
                    wm_rec_loss = rec; wm_kl_loss = kl; z = z_ctrl
                elif args.wm_type == "mdn_rnn":
                    nll, z_ctrl, state = wm.mdn_roll(xb)
                    wm_rec_loss = nll; wm_kl_loss = xb.new_tensor(0.0); z = z_ctrl
                else:
                    z, x_hat, _ = wm(xb)
                    wm_rec_loss = F.smooth_l1_loss(x_hat, xb, reduction="mean")
                    wm_kl_loss = xb.new_tensor(0.0); state = None

                wm_aux_loss = xb.new_tensor(0.0)
                if args.aux_weight > 0.0:
                    r_hat = wm.aux_reward(z)
                    if r_hat is not None:
                        wm_aux_loss = F.mse_loss(r_hat, yb, reduction="mean")

                # --- Clasificador opcional ---
                logits = None
                if use_cls:
                    logits = cls(xb, len_b, pad_b)
                    loss_cls = cls_loss_fn(logits, y_c[i:end].to(device, non_blocking=True))
                else:
                    loss_cls = xb.new_tensor(0.0)

                # --- Controller ---
                pol_in_vec = torch.cat([z, logits], dim=-1) if use_cls else z
                pos = pol(pol_in_vec)
                tr_loss, terms = trading_loss(
                    pos, yb,
                    loss_type=args.loss_type,
                    lambda_var=args.lambda_var,
                    tx_cost=args.tx_cost,
                    turnover_coef=args.turnover_coef,
                    l2_pos=args.l2_pos,
                    vol_norm=args.vol_norm,
                    expo_target=args.expo_target, expo_w=args.expo_w,
                    mean_pos_target=args.mean_pos_target, mean_pos_w=args.mean_pos_w
                )

                # --- Imagination (si procede) ---
                dream_loss = xb.new_tensor(0.0)
                if (mode == "train") and (args.imagine_steps > 0) and state is not None and wm.supports_imagination():
                    dream_state = state
                    dream_positions = []; dream_returns = []
                    for _ in range(args.imagine_steps):
                        dream_state, z_im = wm.imagine_step(dream_state)
                        a_im = pol(z_im)
                        rhat_im = wm.aux_reward(z_im)
                        if rhat_im is None: rhat_im = z_im.new_zeros(z_im.size(0))
                        dream_positions.append(a_im); dream_returns.append(rhat_im.detach())
                    dream_positions = torch.stack(dream_positions, dim=0).mean(dim=0)
                    dream_returns  = torch.stack(dream_returns,  dim=0).mean(dim=0)
                    dream_loss, _ = trading_loss(
                        dream_positions, dream_returns,
                        loss_type=args.loss_type, lambda_var=args.lambda_var,
                        tx_cost=0.0, turnover_coef=0.0,
                        l2_pos=args.l2_pos, vol_norm=args.vol_norm,
                        expo_target=args.expo_target, expo_w=args.expo_w,
                        mean_pos_target=args.mean_pos_target, mean_pos_w=args.mean_pos_w
                    )
                    dream_loss = args.imagine_weight * dream_loss

                loss = tr_loss + args.rec_weight*(wm_rec_loss + wm_kl_loss) + args.aux_weight*wm_aux_loss + dream_loss + (args.cls_weight*loss_cls if use_cls else 0.0)

                if mode=="train":
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()

                total_loss += float(loss.detach().cpu()); nb += 1
                positions_all.append(pos.detach().cpu().numpy())
                wm_rec_sum += float(wm_rec_loss.detach().cpu())
                wm_aux_sum += float(wm_aux_loss.detach().cpu())
                wm_kl_sum  += float(wm_kl_loss.detach().cpu())
                dream_sum  += float(dream_loss.detach().cpu())
                if use_cls: cls_sum += float(loss_cls.detach().cpu())

        positions_all = np.concatenate(positions_all) if positions_all else np.zeros(0, dtype=np.float32)
        y_np = y[start:stop].cpu().numpy()
        metrics = eval_metrics(positions_all, y_np)

        return {
            "loss": total_loss/max(1,nb),
            **metrics,
            "wm_rec": wm_rec_sum / max(1, nb),
            "wm_aux": wm_aux_sum / max(1, nb),
            "wm_kl":  wm_kl_sum  / max(1, nb),
            "dream_loss": dream_sum / max(1, nb),
            "cls_loss": (cls_sum / max(1, nb)) if use_cls else 0.0
        }

    for ep in range(1, args.epochs+1):
        tr = run_epoch("train", idx_tr)
        va = run_epoch("eval", idx_va)
        te = run_epoch("eval", idx_te)

        print(f"[Ep{ep:02d}] "
              f"Tr S~={tr['sharpe_like']:.3f} | Va S~={va['sharpe_like']:.3f} | Te S~={te['sharpe_like']:.3f} | "
              f"loss={tr['loss']:.4f} | WM(rec)={tr['wm_rec']:.4f} aux={tr['wm_aux']:.4f} kl={tr['wm_kl']:.4f} Dream={tr['dream_loss']:.4f} | CLS={tr['cls_loss']:.4f}")

        score = va["sharpe_like"]
        improved = (score - best_va) > args.min_delta
        if ep <= args.warmup_epochs:
            if score > best_va:
                best_va = score; best_ep = ep; bad = 0
                torch.save({"wm": wm.state_dict(), "pol": pol.state_dict(),
                            "cls": (cls.state_dict() if use_cls else None),
                            "cfg": vars(args), "norm": norm,
                            "train": tr, "val": va, "test": te,
                            "arch": {"wm": args.wm_type, "pol": args.policy, "with_cls": use_cls}}, ckpt)
        else:
            if improved:
                best_va = score; best_ep = ep; bad = 0
                torch.save({"wm": wm.state_dict(), "pol": pol.state_dict(),
                            "cls": (cls.state_dict() if use_cls else None),
                            "cfg": vars(args), "norm": norm,
                            "train": tr, "val": va, "test": te,
                            "arch": {"wm": args.wm_type, "pol": args.policy, "with_cls": use_cls}}, ckpt)
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"Early stopping Ep{ep} (best Va S~={best_va:.4f} @Ep{best_ep})")
                    break

    print(f"\n✔ Checkpoint guardado en {ckpt}")

if __name__ == "__main__":
    main()

