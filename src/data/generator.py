#!/usr/bin/env python3
# src/data/generator.py
"""
Serie sintética: MS-AR + GARCH + Hawkes (saltos) + **motivos repetidos** (patrones cortos).
- Conserva tu pipeline base (regímenes, estacionalidad, garch, hawkes).
- Añade patrones cortos y repetidos que se inyectan como offsets sobre los log-retornos.
- Reduce por defecto la tasa/intensidad de saltos Hawkes para facilitar métricas mejores.

Salidas (compat):
  SYN_PRICE.npy, PRICE_5m.npy, RETURNS.npy, REGIMES.npy, JUMPS.npy, HAWKES_LAMBDA.npy, MOTIFS.npy, meta.json
"""

import argparse, json, math, random
from pathlib import Path
import numpy as np

DT  = np.float64
EPS = 1e-12

# ───────────────── util ─────────────────

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)

def sample_markov_chain(T: int, P: np.ndarray, s0: int) -> np.ndarray:
    S = P.shape[0]
    states = np.zeros(T, dtype=np.int64)
    states[0] = s0
    for t in range(1, T):
        states[t] = np.random.choice(S, p=P[states[t-1]])
    return states

def garch11(T: int, omega: float, alpha: float, beta: float, shock: np.ndarray,
            sigma_floor: float, sigma_cap: float) -> np.ndarray:
    sig2 = np.zeros(T, dtype=DT)
    sig2[0] = max(omega / max(1e-6, (1.0 - alpha - beta)), (sigma_floor**2))
    for t in range(1, T):
        sig2[t] = omega + alpha * (shock[t-1] ** 2) + beta * sig2[t-1]
        if not np.isfinite(sig2[t]) or sig2[t] <= 0:
            sig2[t] = sig2[t-1]
    sigma = np.sqrt(np.maximum(sig2, 1e-18))
    return np.clip(sigma, sigma_floor, sigma_cap)

def make_seasonality(T: int, period: int, a: float, b: float) -> np.ndarray:
    t = np.arange(T, dtype=DT)
    ang = 2.0 * math.pi * t / max(1, period)
    return a * np.sin(ang) + b * np.cos(ang)

def simulate_hawkes_bernoulli(T: int, lam0: float, alpha: float, beta: float):
    """ λ_t = lam0 + e^{-beta}(λ_{t-1}-lam0) + alpha n_{t-1},  n_t ~ Bernoulli(1-exp(-λ_t)) """
    n = np.zeros(T, dtype=np.int8)
    lam = np.zeros(T, dtype=DT)
    decay = math.exp(-beta)
    U = np.random.rand(T)
    lam[0] = lam0
    p0 = 1.0 - math.exp(-max(lam[0], 0.0))
    n[0] = np.int8(int(U[0] < p0))
    for t in range(1, T):
        lam[t] = lam0 + decay * (lam[t-1] - lam0) + alpha * n[t-1]
        p = 1.0 - math.exp(-max(lam[t], 0.0))
        n[t] = np.int8(int(U[t] < p))
    return n, lam

def build_default_transition(p_stay: float = 0.985) -> np.ndarray:
    off = (1.0 - p_stay) / 2.0
    P = np.array([
        [p_stay, off,    off   ],
        [off,    p_stay, off   ],
        [off,    off,    p_stay]
    ], dtype=DT)
    return P

def difficulty_scale(x: float, d: float, toward: float = 0.0) -> float:
    return (1.0 - d) * x + d * toward

# ───────────────── motivos (patrones) ─────────────────

def motif_v(n):
    # descenso lineal y subida lineal simétrica
    half = n//2
    down = np.linspace( +1.0, -1.0, num=half, endpoint=False)
    up   = np.linspace( -1.0, +1.0, num=n-half, endpoint=True)
    return np.concatenate([down, up])

def motif_inv_v(n):  # ∧
    return -motif_v(n)

def motif_u(n):
    x = np.linspace(-1, 1, n)
    y = (x**2)
    y = (y - y.mean())
    return y

def motif_cap(n):  # ∩
    return -motif_u(n)

def motif_stairs_up(n, k=4):
    # k escalones
    edges = np.linspace(0, n, k+1, dtype=int)
    y = np.zeros(n, dtype=DT)
    for i in range(k):
        y[edges[i]:edges[i+1]] = i+1
    y = (y - y.mean())
    return y

def motif_stairs_down(n, k=4):
    return -motif_stairs_up(n, k=k)

def motif_ramp(n, sign=+1):
    y = np.linspace(-1, 1, n) * sign
    y = y - y.mean()
    return y

def motif_damped_sine(n, periods=1.5, decay=3.0):
    t = np.linspace(0, 1, n)
    y = np.sin(2*np.pi*periods*t) * np.exp(-decay*t)
    return y - y.mean()

MOTIF_FUNS = [
    ("V", motif_v),
    ("INV_V", motif_inv_v),
    ("U", motif_u),
    ("CAP", motif_cap),
    ("STAIRS_UP", motif_stairs_up),
    ("STAIRS_DOWN", motif_stairs_down),
    ("RAMP_UP", lambda n: motif_ramp(n, +1)),
    ("RAMP_DOWN", lambda n: motif_ramp(n, -1)),
    ("DAMPED_SINE", motif_damped_sine),
]

def place_motifs(T: int,
                 n_types: int = 5,
                 reps_per_type: int = 8,
                 len_min: int = 40,
                 len_max: int = 180,
                 strength_bps: float = 1.2,      # bps por barra (escala multiplicativa)
                 min_gap: int = 48,
                 rng_seed: int = 0):
    """
    Devuelve:
      offsets: np.ndarray[T] con la suma de los motivos (en unidades base, sin escalar por σ)
      meta:    lista de (start, length, type_id, strength_bps)
    """
    rng = np.random.default_rng(rng_seed)
    types_idx = rng.choice(len(MOTIF_FUNS), size=n_types, replace=False)
    occupied = np.zeros(T, dtype=np.int8)
    offsets = np.zeros(T, dtype=DT)
    meta = []

    def can_place(s, L):
        a = max(0, s - min_gap)
        b = min(T, s + L + min_gap)
        return np.all(occupied[a:b] == 0)

    for tid in types_idx:
        name, fn = MOTIF_FUNS[tid]
        reps = int(reps_per_type)
        for _ in range(reps):
            L = int(rng.integers(len_min, len_max+1))
            # intenta varias veces ubicaciones que no solapen
            for _try in range(50):
                s = int(rng.integers(0, max(1, T - L)))
                if can_place(s, L):
                    # construye motivo normalizado a var≈1 y escálalo a bps/barra
                    y = fn(L).astype(DT)
                    if np.std(y) < 1e-8:
                        continue
                    y = y / (np.std(y) + 1e-9)
                    amp = (strength_bps / 1e4)  # bps → fracción
                    offsets[s:s+L] += amp * y
                    occupied[s:s+L] = 1
                    meta.append((int(s), int(L), int(tid), float(strength_bps)))
                    break
            # si no encontró hueco, se omite esa repetición
    return offsets, meta

# ───────────────── main ─────────────────

def main():
    pa = argparse.ArgumentParser()
    # Longitud y granularidad
    pa.add_argument("--length", type=int, default=200_000, help="Nº de barras")
    pa.add_argument("--step-min", type=int, default=5, help="Minutos por barra")
    pa.add_argument("--price0", type=float, default=100.0, help="Precio inicial")

    # Regímenes (μ_*_bps por día)
    pa.add_argument("--mu_trend_bps", type=float, default=2.0)
    pa.add_argument("--phi_trend", type=float, default=0.7)
    pa.add_argument("--mu_mr_bps", type=float, default=0.0)
    pa.add_argument("--phi_mr", type=float, default=-0.35)
    pa.add_argument("--mu_chop_bps", type=float, default=0.0)
    pa.add_argument("--phi_chop", type=float, default=0.10)
    pa.add_argument("--p_stay", type=float, default=0.985, help="Persistencia régimen")

    # Estacionalidad (bps **por barra** en log-ret)
    pa.add_argument("--season_amp_bps", type=float, default=0.6)
    pa.add_argument("--season_a_bps", type=float, default=None)
    pa.add_argument("--season_b_bps", type=float, default=None)

    # Volatilidad objetivo (diaria)
    pa.add_argument("--target-daily-vol-bps", type=float, default=160.0)
    pa.add_argument("--garch_alpha", type=float, default=0.05)
    pa.add_argument("--garch_beta", type=float, default=0.92)
    pa.add_argument("--sigma-min-mult", type=float, default=0.30)
    pa.add_argument("--sigma-max-mult", type=float, default=3.5)

    # Hawkes / jumps (más suaves por defecto)
    pa.add_argument("--jump_base_lambda", type=float, default=0.0012)  # ↓ tasa base
    pa.add_argument("--hawkes_alpha", type=float, default=0.008)       # ↓ clustering
    pa.add_argument("--hawkes_beta", type=float, default=0.45)         # ↑ decaimiento → menos memoria
    pa.add_argument("--jump_sigma_bps", type=float, default=18.0)      # ↓ tamaño salto
    pa.add_argument("--jump_mu_bps", type=float, default=0.0)
    pa.add_argument("--jump-clip-bps", type=float, default=120.0)
    pa.add_argument("--jump-dist", choices=["normal","t"], default="normal")
    pa.add_argument("--jump-t-df", type=float, default=6.0)

    # Clipping de retornos por barra (log-ret)
    pa.add_argument("--clip-ret-bps", type=float, default=350.0)
    pa.add_argument("--logp-clip", type=float, default=60.0)

    # Dificultad (SNR)
    pa.add_argument("--difficulty", type=float, default=0.0)  # 0=fácil (patrones marcados), 1=difícil

    # Motivos (patrones)
    pa.add_argument("--motif-types", type=int, default=6, help="nº tipos distintos (≤9)")
    pa.add_argument("--motif-reps", type=int, default=10, help="repeticiones por tipo")
    pa.add_argument("--motif-len-min", type=int, default=50)
    pa.add_argument("--motif-len-max", type=int, default=160)
    pa.add_argument("--motif-strength-bps", type=float, default=1.4, help="bps por barra (escala del motivo)")
    pa.add_argument("--motif-min-gap", type=int, default=64, help="gap mínimo entre motivos")

    # Reproducibilidad / salida
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--outdir", type=str, default="prepared/synth")
    pa.add_argument("--name", type=str, default="msar5m_patterns_seed0")

    args = pa.parse_args()
    set_seed(args.seed)

    T = int(args.length)
    step_min = int(args.step_min)
    period = int((24 * 60) / max(1, step_min))  # barras por día

    # ----- parámetros por régimen (μ en bps/día → barra) -----
    trend_sign = 1.0 if (np.random.rand() < 0.5) else -1.0
    mu_trend_day = difficulty_scale(args.mu_trend_bps / 1e4, args.difficulty, toward=0.0) * trend_sign
    mu_mr_day    = difficulty_scale(args.mu_mr_bps    / 1e4, args.difficulty, toward=0.0)
    mu_chop_day  = difficulty_scale(args.mu_chop_bps  / 1e4, args.difficulty, toward=0.0)
    mu_trend = mu_trend_day / period
    mu_mr    = mu_mr_day    / period
    mu_chop  = mu_chop_day  / period

    phi_trend = difficulty_scale(args.phi_trend, args.difficulty, toward=0.0)
    phi_mr    = difficulty_scale(args.phi_mr,    args.difficulty, toward=0.0)
    phi_chop  = difficulty_scale(args.phi_chop,  args.difficulty, toward=0.0)

    season_a = (args.season_a_bps / 1e4) if args.season_a_bps is not None else (args.season_amp_bps / 1e4)
    season_b = (args.season_b_bps / 1e4) if args.season_b_bps is not None else (args.season_amp_bps / 1e4)
    season_a = difficulty_scale(season_a, args.difficulty, toward=0.0)
    season_b = difficulty_scale(season_b, args.difficulty, toward=0.0)
    season = make_seasonality(T, period=period, a=season_a, b=season_b)

    # Markov regimes
    P = build_default_transition(p_stay=args.p_stay)
    s0 = np.random.choice(3)
    regimes = sample_markov_chain(T, P, s0)

    # ruido + GARCH
    z = np.random.randn(T).astype(DT)
    vol_d = (args.target_daily_vol_bps / 1e4)
    sigma_5m_target = vol_d / math.sqrt(period)
    denom = max(1e-6, 1.0 - args.garch_alpha - args.garch_beta)
    omega = (sigma_5m_target ** 2) * denom
    sigma_floor = args.sigma_min_mult * sigma_5m_target
    sigma_cap   = args.sigma_max_mult * sigma_5m_target
    sigma = garch11(T, omega=omega, alpha=args.garch_alpha, beta=args.garch_beta,
                    shock=z, sigma_floor=sigma_floor, sigma_cap=sigma_cap)

    # Hawkes / saltos (suaves)
    hawkes_alpha = args.hawkes_alpha
    if hawkes_alpha / max(1e-6, args.hawkes_beta) >= 1.0:
        print("[WARN] hawkes_alpha/beta >= 1; ajustando alpha para estabilidad.")
        hawkes_alpha = 0.9 * args.hawkes_beta
    n_jump, lam_series = simulate_hawkes_bernoulli(
        T=T, lam0=args.jump_base_lambda, alpha=hawkes_alpha, beta=args.hawkes_beta
    )
    jump_mu    = (args.jump_mu_bps    / 1e4)
    jump_sigma = (args.jump_sigma_bps / 1e4)
    if args.jump_dist == "normal":
        raw_jump = jump_mu + jump_sigma * np.random.randn(T)
    else:
        df = max(1.5, args.jump_t_df)
        raw_jump = jump_mu + jump_sigma * (np.random.standard_t(df, size=T) / math.sqrt(df/(df-2.0)))
    jump_clip = (args.jump_clip_bps / 1e4)
    jump_sizes = np.clip(raw_jump, -jump_clip, +jump_clip).astype(DT)
    J = (n_jump.astype(DT) * jump_sizes).astype(DT)

    # Motivos repetidos (offsets en log-ret)
    offsets, motifs_meta = place_motifs(
        T=T,
        n_types=min(args.motif_types, len(MOTIF_FUNS)),
        reps_per_type=args.motif_reps,
        len_min=args.motif_len_min,
        len_max=args.motif_len_max,
        strength_bps=args.motif_strength_bps,
        min_gap=args.motif_min_gap,
        rng_seed=args.seed + 17,
    )

    # log-ret base + season + GARCH + saltos + offsets de motivo
    r = np.zeros(T, dtype=DT)
    clip_ret = (args.clip_ret_bps / 1e4)
    for t in range(T):
        s = regimes[t]
        if s == 0:   mu_bar, phi = mu_trend, phi_trend
        elif s == 1: mu_bar, phi = mu_mr,    phi_mr
        else:        mu_bar, phi = mu_chop,  phi_chop
        ar = phi * (r[t-1] if t > 0 else 0.0)
        r_t = mu_bar + season[t] + ar + sigma[t] * z[t] + J[t] + offsets[t]
        r[t] = float(np.clip(r_t, -clip_ret, +clip_ret))
        if not np.isfinite(r[t]):
            r[t] = 0.0

    # precio por cumsum(log-ret)
    logp = np.cumsum(r); logp -= logp[0]
    if args.logp_clip is not None and args.logp_clip > 0:
        logp = np.clip(logp, -abs(args.logp_clip), abs(args.logp_clip))
    price = np.exp(logp) * float(args.price0)
    price = np.maximum(price, 1e-12)

    # guardar
    outdir = Path(args.outdir) / args.name
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "SYN_PRICE.npy", price.astype(np.float64))
    np.save(outdir / "PRICE_5m.npy", price.astype(np.float64))  # compat
    np.save(outdir / "RETURNS.npy", r.astype(np.float32))
    np.save(outdir / "REGIMES.npy", regimes.astype(np.int16))
    np.save(outdir / "JUMPS.npy", J.astype(np.float32))               # tamaño salto en log-ret
    np.save(outdir / "HAWKES_LAMBDA.npy", lam_series.astype(np.float32))
    # MOTIFS: [start, length, type_id, strength_bps]
    if motifs_meta:
        M = np.array(motifs_meta, dtype=np.float32)
    else:
        M = np.zeros((0,4), dtype=np.float32)
    np.save(outdir / "MOTIFS.npy", M)

    meta = {
        "length": int(T),
        "step_min": int(step_min),
        "period_per_day": int(period),
        "price0": float(args.price0),
        "returns_type": "log",
        "regimes": {
            "trend": {"mu_day_bps": float(mu_trend_day*1e4), "mu_bar_bps": float(mu_trend*1e4), "phi": float(phi_trend)},
            "mr":    {"mu_day_bps": float(mu_mr_day*1e4),    "mu_bar_bps": float(mu_mr*1e4),    "phi": float(phi_mr)},
            "chop":  {"mu_day_bps": float(mu_chop_day*1e4),  "mu_bar_bps": float(mu_chop*1e4),  "phi": float(phi_chop)},
            "P": np.asarray(P).tolist()
        },
        "seasonality": {"a_bps_per_bar": float(season_a*1e4), "b_bps_per_bar": float(season_b*1e4), "period": int(period)},
        "garch": {
            "alpha": float(args.garch_alpha),
            "beta": float(args.garch_beta),
            "target_daily_vol_bps": float(args.target_daily_vol_bps),
            "sigma_5m_target": float(sigma_5m_target),
            "sigma_floor": float(sigma_floor),
            "sigma_cap": float(sigma_cap),
            "omega": float(omega)
        },
        "hawkes": {"lam0": float(args.jump_base_lambda), "alpha": float(hawkes_alpha), "beta": float(args.hawkes_beta)},
        "jumps": {
            "mu_bps": float(args.jump_mu_bps),
            "sigma_bps": float(args.jump_sigma_bps),
            "clip_bps": float(args.jump_clip_bps),
            "dist": args.jump_dist,
            "t_df": float(args.jump_t_df)
        },
        "motifs": {
            "types_used": int(min(args.motif_types, len(MOTIF_FUNS))),
            "reps_per_type": int(args.motif_reps),
            "len_min": int(args.motif_len_min),
            "len_max": int(args.motif_len_max),
            "strength_bps": float(args.motif_strength_bps),
            "min_gap": int(args.motif_min_gap),
            "catalog": [name for name, _ in MOTIF_FUNS]
        },
        "clip_ret_bps": float(args.clip_ret_bps),
        "logp_clip": float(args.logp_clip),
        "difficulty": float(args.difficulty),
        "seed": int(args.seed)
    }
    with open(outdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    sharpe_like = float(np.mean(r) / (np.std(r) + 1e-12))
    print(f"✔ Serie generada en {outdir}")
    print(f"  barras={T} step_min={step_min} period={period}")
    print(f"  motivos: {M.shape[0]} instancias; tipos={meta['motifs']['types_used']} reps/type={args.motif_reps} "
          f"len∈[{args.motif_len_min},{args.motif_len_max}] fuerza={args.motif_strength_bps} bps/barra")
    print(f"  saltos_total={int(np.sum(n_jump))}  lambda0={args.jump_base_lambda}")
    print(f"  Sharpe-like(log-ret)={sharpe_like:.3f}")
    print("  ficheros: SYN_PRICE.npy, RETURNS.npy, REGIMES.npy, JUMPS.npy, HAWKES_LAMBDA.npy, MOTIFS.npy, meta.json")

if __name__ == "__main__":
    main()

