#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baselines canónicos sobre mundos sintéticos, ahora con evaluación alineada a los trainers:
  - Evaluación por defecto: MISMO esquema que los controladores (cada paso del split: stride=1)
  - Opción --nonoverlap para recuperar la evaluación anterior (submuestreo no solapado stride=steps_H)

Baselines:
  - FLAT (pos=0)
  - LONG (constante) y LONG vol-target (cap a [-1,1])
  - MOMENTUM / CONTRARIAN (1 variable con umbral relativo a τ)
  - EMA/SMA crossover + filtro de volatilidad

Definiciones:
  - r_H(t) = (P_{t+H}-P_t)/P_t (simple return)
  - Splits 70/15/15 sobre índice de r_H (N_eff)
  - Métrica coherente con trainers: μ, σ, Sharpe_like = μ/σ, hit-rate
  - Se reporta turnover = mean(|Δpos|) (para referencia), pero **no** se restan costes al PnL
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

# ------------------- Utilidades de datos -------------------
def load_price_series(world: Path) -> np.ndarray:
    for name in ["SYN_PRICE.npy", "SYN_SERIE_5m.npy", "PRICE_5m.npy", "SERIE_5m.npy"]:
        p = world / name
        if p.exists():
            arr = np.load(p).astype(np.float64)
            if arr.ndim > 1:
                arr = arr.squeeze()
            return arr
    raise FileNotFoundError(f"No se encontró serie de precio en {world}")

def load_step_min(world: Path, default=5) -> int:
    for name in ["SYN_META.json", "meta.json"]:
        p = world / name
        if p.exists():
            try:
                meta = json.loads(p.read_text())
                v = int(meta.get("step_min", meta.get("dt_min", default)))
                return v if v > 0 else default
            except Exception:
                return default
    return default

def split_indices(N: int, tr=0.70, va=0.15):
    n_tr = int(tr * N); n_va = int(va * N)
    return slice(0, n_tr), slice(n_tr, n_tr + n_va), slice(n_tr + n_va, N)

def forward_return(price: np.ndarray, steps_H: int) -> np.ndarray:
    price = np.clip(price.astype(np.float64), 1e-12, None)
    return (price[steps_H:] - price[:-steps_H]) / price[:-steps_H]

def quantile_tau(abs_rH_train: np.ndarray, q: float) -> float:
    return float(np.quantile(abs_rH_train, q))

def slice_apply(x: np.ndarray, sl: slice) -> np.ndarray:
    return x[sl.start:sl.stop] if (sl.start is not None and sl.stop is not None) else x

# ------------------- Señales baseline -------------------
def pos_flat(N: int) -> np.ndarray:
    return np.zeros(N, dtype=np.float64)

def pos_long_const(N: int, scale: float = 1.0) -> np.ndarray:
    return np.full(N, float(np.clip(scale, -1, 1)), dtype=np.float64)

def rolling_std(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return np.full_like(x, np.std(x))
    out = np.full_like(x, np.nan)
    c = np.cumsum(np.r_[0.0, x])
    c2 = np.cumsum(np.r_[0.0, x * x])
    for i in range(win, x.size):
        s = c[i] - c[i - win]
        s2 = c2[i] - c2[i - win]
        var = (s2 - (s * s) / win) / max(1, (win - 1))
        out[i] = np.sqrt(max(var, 1e-18))
    first = np.nanmin(np.where(~np.isnan(out), out, np.inf))
    out[: win] = first if np.isfinite(first) else np.nan_to_num(out, nan=np.std(x))
    return out

def ema(x: np.ndarray, win: int) -> np.ndarray:
    a = 2.0 / (win + 1.0)
    y = np.zeros_like(x)
    y[0] = x[0]
    for t in range(1, x.size):
        y[t] = a * x[t] + (1 - a) * y[t - 1]
    return y

def pos_long_vol_target(r1: np.ndarray,
                        steps_H: int,
                        target_daily_sigma: float = 0.01,
                        step_min: int = 5,
                        cap: float = 1.0) -> np.ndarray:
    """
    Vol-target sobre LONG. Estimamos σ diaria con r1 y ventana de ~1 día.
    Escalamos a horizonte H asumiendo random walk: σ_H ≈ σ_1 * sqrt(steps_H).
    """
    per_day = int(round(24 * 60 / max(1, step_min)))
    sig1 = rolling_std(r1, max(per_day, 32))
    sigH = sig1 * np.sqrt(max(1, steps_H))
    scale = np.clip(target_daily_sigma / (sigH + 1e-12), -cap, cap)
    return np.clip(scale, -cap, cap)

def pos_momentum(price: np.ndarray,
                 k_steps: int,
                 theta: float = 0.0,
                 contrarian: bool = False) -> np.ndarray:
    """
    Señal con log-precio: sign( logp_t - logp_{t-k} ), con banda neutra |Δ|<theta -> 0.
    """
    logp = np.log(np.clip(price, 1e-12, None))
    delta = np.zeros_like(logp)
    if k_steps > 0:
        delta[k_steps:] = logp[k_steps:] - logp[:-k_steps]
    s = np.sign(delta)
    if theta > 0:
        s[np.abs(delta) < theta] = 0.0
    if contrarian:
        s = -s
    return s.astype(np.float64)

def pos_crossover(price: np.ndarray,
                  fast: int, slow: int,
                  theta_vol: float = 0.0,
                  use_ema: bool = True,
                  r1_for_vol: np.ndarray | None = None) -> np.ndarray:
    """
    Crossover EMA/SMA. Señal = sign(spread), con filtro de volatilidad:
    neutral si |spread| < theta_vol * σ (σ de r1).
    """
    logp = np.log(np.clip(price, 1e-12, None))
    if use_ema:
        ma_f = ema(logp, fast)
        ma_s = ema(logp, slow)
    else:
        kf = max(1, fast); ks = max(1, slow)
        ma_f = np.convolve(logp, np.ones(kf)/kf, mode="same")
        ma_s = np.convolve(logp, np.ones(ks)/ks, mode="same")

    spread = ma_f - ma_s
    s = np.sign(spread)

    if theta_vol > 0:
        if r1_for_vol is None:
            r1_for_vol = np.diff(price, prepend=price[0]) / np.maximum(price, 1e-12)
        sig = rolling_std(r1_for_vol, max(slow, 32))
        s[np.abs(spread) < theta_vol * sig] = 0.0

    return s.astype(np.float64)

# ------------------- Evaluación (trainer-like por defecto) -------------------
def eval_positions_trainer_like(rH: np.ndarray,
                                pos: np.ndarray) -> dict:
    """
    Evalúa como en los trainers: cada paso (stride=1), sin restar costes al PnL.
    Reporta μ, σ, sharpe_like=μ/σ, hit, turnover, n.
    """
    N = rH.size
    pos = np.clip(pos[:N], -1.0, 1.0).astype(np.float64)
    pnl = pos * rH
    mu = float(np.mean(pnl))
    sd = float(np.std(pnl) + 1e-12)
    sharpe = float(mu / sd)
    hit = float(np.mean(np.sign(pos) == np.sign(rH)))
    dpos = np.diff(pos, prepend=pos[0] if N > 0 else 0.0)
    turn = float(np.mean(np.abs(dpos)))
    long_share = float(np.mean(pos > 0))
    flat_share = float(np.mean(np.isclose(pos, 0.0, atol=1e-3)))
    short_share = float(np.mean(pos < 0))
    return dict(mu=mu, sigma=sd, sharpe_like=sharpe, hit=hit, turnover=turn,
                long=long_share, flat=flat_share, short=short_share, n=N)

def eval_positions_nonoverlap(rH: np.ndarray,
                              pos: np.ndarray,
                              stride: int,
                              tx_cost: float = 1e-4,
                              turnover_pen: float = 0.5,
                              lambda_var: float = 12.0) -> dict:
    """
    Versión anterior (no solapada) mantenida para comparación (--nonoverlap).
    """
    N = rH.size
    idx = np.arange(0, N, stride, dtype=int)
    pos_eval = np.clip(pos[:N], -1.0, 1.0)[idx]
    r_eval = rH[idx]

    dpos = np.diff(pos_eval, prepend=0.0)
    costs = tx_cost * np.abs(dpos) + turnover_pen * np.abs(dpos)

    pnl = pos_eval * r_eval  # ojo: no restamos costes para métrica comparable
    mu = float(np.mean(pnl))
    sd = float(np.std(pnl) + 1e-12)
    sharpe = float(mu / sd)

    # Para referencia dejamos también el objetivo anterior μ - λ Var y 'costes' aparte
    var = float(np.var(pos_eval * r_eval))
    s_tilde_old = mu - lambda_var * var

    hit = float(np.mean((pos_eval * r_eval) > 0.0))
    turn = float(np.mean(np.abs(dpos)))
    return dict(mu=mu, sigma=sd, sharpe_like=sharpe, hit=hit, turnover=turn,
                n=int(idx.size), legacy_S_tilde=s_tilde_old,
                legacy_cost=float(np.mean(costs)))

# ------------------- Pipeline por mundo -------------------
def run_world(world_dir: Path,
              horizon_min: int,
              q_tau: float,
              lambda_var: float,
              tx_cost: float,
              turnover_pen: float,
              daily_sigma_target: float,
              grid_mom: list[tuple[int,float,bool]],
              grid_xover: list[tuple[int,int,float,bool]],
              nonoverlap: bool = False):
    world = Path(world_dir).expanduser().resolve()
    price = load_price_series(world)
    step_min = load_step_min(world, 5)
    steps_H = max(1, horizon_min // max(1, step_min))
    rH = forward_return(price, steps_H)
    N_eff = rH.size
    sl_tr, sl_va, sl_te = split_indices(N_eff, 0.70, 0.15)

    # r1 para vol-target/filtros
    r1 = (price[1:] - price[:-1]) / price[:-1]
    r1 = np.r_[0.0, r1]  # misma longitud que price

    # τ desde TRAIN (para umbrales de señales discretas)
    tau = quantile_tau(np.abs(rH[sl_tr]), q_tau)

    # Selector de evaluación
    def EVAL(rH_split, pos_split, label_prefix=""):
        if nonoverlap:
            stride = steps_H
            m = eval_positions_nonoverlap(rH_split, pos_split, stride,
                                          tx_cost=tx_cost, turnover_pen=turnover_pen, lambda_var=lambda_var)
        else:
            m = eval_positions_trainer_like(rH_split, pos_split)
        return m

    def _print_block(name: str, tau_val: float, metrics: dict):
        print(f"\n{name} | τ(TRAIN)≈{tau_val:.5f} | H={horizon_min}m | step={step_min}m | "
              f"mode={'NON-OVERLAP' if nonoverlap else 'TRAINER-LIKE'}")
        for split in ["train","val","test"]:
            m = metrics[split]
            print(f"  {split:5s}: μ={m['mu']:+.6e}  σ={m['sigma']:.6e}  "
                  f"S~={m['sharpe_like']:+.6e}  hit={m['hit']:.3f}  "
                  f"turn={m['turnover']:.4f}  n={m['n']}")

    results = {}

    # -------- FLAT / LONG ----------
    pos0 = pos_flat(N_eff)
    flat_res = {
        "train": EVAL(rH[sl_tr], pos0[sl_tr]),
        "val":   EVAL(rH[sl_va], pos0[sl_va]),
        "test":  EVAL(rH[sl_te], pos0[sl_te]),
    }
    _print_block("FLAT (pos=0)", tau, flat_res)
    results["flat"] = flat_res

    pos1 = pos_long_const(N_eff, 1.0)
    long_res = {
        "train": EVAL(rH[sl_tr], pos1[sl_tr]),
        "val":   EVAL(rH[sl_va], pos1[sl_va]),
        "test":  EVAL(rH[sl_te], pos1[sl_te]),
    }
    _print_block("LONG (pos=+1)", tau, long_res)
    results["long_1"] = long_res

    pos1v = pos_long_vol_target(r1, steps_H, target_daily_sigma=daily_sigma_target,
                                step_min=step_min, cap=1.0)[:N_eff]
    long_vt_res = {
        "train": EVAL(rH[sl_tr], pos1v[sl_tr]),
        "val":   EVAL(rH[sl_va], pos1v[sl_va]),
        "test":  EVAL(rH[sl_te], pos1v[sl_te]),
    }
    _print_block(f"LONG vol-target (σ*={daily_sigma_target:.3f}/día)", tau, long_vt_res)
    results["long_vol_target"] = long_vt_res

    # -------- MOMENTUM / CONTRARIAN (selección por VAL Sharpe_like) ----------
    best_val, best_conf, best_pos = (-1e100, None, None)
    for k_steps, theta_rel, contr in grid_mom:
        theta = theta_rel * tau
        pos_mom = pos_momentum(price[:N_eff + steps_H], k_steps=k_steps, theta=theta, contrarian=contr)
        pos_mom = pos_mom[:N_eff]  # alinear a rH
        m_val = EVAL(rH[sl_va], pos_mom[sl_va])
        if m_val["sharpe_like"] > best_val:
            best_val, best_conf, best_pos = m_val["sharpe_like"], (k_steps, theta_rel, contr), pos_mom.copy()
    mom_res = {
        "train": EVAL(rH[sl_tr], best_pos[sl_tr]),
        "val":   EVAL(rH[sl_va], best_pos[sl_va]),
        "test":  EVAL(rH[sl_te], best_pos[sl_te]),
    }
    _print_block(f"MOM/CONTR (best VAL) k={best_conf[0]} θ={best_conf[1]}·τ contr={best_conf[2]}", tau, mom_res)
    results["momentum/contrarian_best"] = mom_res

    # -------- EMA/SMA crossover (selección por VAL Sharpe_like) ----------
    best_val, best_conf, best_pos = (-1e100, None, None)
    for fast, slow, theta_rel, use_ema in grid_xover:
        if fast >= slow:
            continue
        pos_x = pos_crossover(price[:N_eff + steps_H], fast=fast, slow=slow,
                              theta_vol=theta_rel * tau, use_ema=use_ema, r1_for_vol=r1)
        pos_x = pos_x[:N_eff]
        m_val = EVAL(rH[sl_va], pos_x[sl_va])
        if m_val["sharpe_like"] > best_val:
            best_val, best_conf, best_pos = m_val["sharpe_like"], (fast, slow, theta_rel, use_ema), pos_x.copy()
    lab = "EMA" if best_conf[3] else "SMA"
    xover_res = {
        "train": EVAL(rH[sl_tr], best_pos[sl_tr]),
        "val":   EVAL(rH[sl_va], best_pos[sl_va]),
        "test":  EVAL(rH[sl_te], best_pos[sl_te]),
    }
    _print_block(f"{lab} crossover (best VAL) fast={best_conf[0]} slow={best_conf[1]} θ={best_conf[2]}·τ", tau, xover_res)
    results["xover_best"] = xover_res

    return {
        "world": world.name, "step_min": step_min, "H_min": horizon_min, "steps_H": steps_H,
        "tau_train": tau, "daily_sigma_target": daily_sigma_target,
        "mode": ("nonoverlap" if nonoverlap else "trainer_like"),
        "results": results
    }

# ------------------- CLI -------------------
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--world", required=True, help="Ruta mundo: prepared/synth/msar5m_easy_seed123")
    pa.add_argument("--horizon-min", type=int, default=60)
    pa.add_argument("--q-tau", type=float, default=0.60)
    # costes conservados para la ruta legacy (non-overlap)
    pa.add_argument("--lambda-var", type=float, default=12.0)
    pa.add_argument("--tx-cost", type=float, default=1e-4)
    pa.add_argument("--turnover-pen", type=float, default=0.5)
    pa.add_argument("--daily-sigma-target", type=float, default=0.010)
    pa.add_argument("--out", type=str, default=None)
    pa.add_argument("--nonoverlap", action="store_true",
                    help="Usa evaluación antigua (stride=H, μ-λVar legacy guardado como 'legacy_S_tilde').")

    # Grids por defecto
    pa.add_argument("--grid-mom", type=str,
        default="H2,0.0,False;H,0.0,False;2H,0.0,False;H2,0.5,False;H,0.5,False;2H,0.5,False;H,1.0,False;H,0.5,True",
        help="Items 'k,theta_rel,contrarian' separados por ';'. k en {H2,H,2H} para H/2, H, 2H.")
    pa.add_argument("--grid-xover", type=str,
        default="12,48,0.0,ema;24,96,0.0,ema;12,48,0.25,ema;24,96,0.25,ema;12,48,0.5,ema;24,96,0.5,ema",
        help="Items 'fast,slow,theta_rel,ema|sma' separados por ';'.")

    args = pa.parse_args()

    # Parse grids
    def parse_mom(s, steps_H):
        out=[]
        for item in s.split(";"):
            k,th,co = item.split(",")
            if k.lower()=="h2": k_steps=max(1, steps_H//2)
            elif k.lower()=="h": k_steps=steps_H
            elif k.lower()=="2h": k_steps=steps_H*2
            else: k_steps=int(float(k))
            out.append((k_steps, float(th), co.lower()=="true"))
        return out
    def parse_xover(s):
        out=[]
        for item in s.split(";"):
            f,sl,th,mode = item.split(",")
            out.append((int(f), int(sl), float(th), mode.lower()=="ema"))
        return out

    wpath = Path(args.world).expanduser().resolve()
    step_min = load_step_min(wpath, 5)
    steps_H = max(1, args.horizon_min // max(1, step_min))
    grid_m = parse_mom(args.grid_mom, steps_H)
    grid_x = parse_xover(args.grid_xover)

    res = run_world(
        wpath, args.horizon_min, args.q_tau,
        args.lambda_var, args.tx_cost, args.turnover_pen, args.daily_sigma_target,
        grid_m, grid_x,
        nonoverlap=args.nonoverlap
    )
    # Guardado
    out = Path(args.out) if args.out else (wpath / f"BASELINES_H{args.horizon_min}m_results.json")
    out.write_text(json.dumps(res, indent=2))
    print(f"\n✔ Resultados guardados en: {out}")

if __name__ == "__main__":
    main()

