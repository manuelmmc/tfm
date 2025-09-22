#!/usr/bin/env python
# scripts/prepare_windows_synth.py
"""
Prepara ventanas para mundos sintéticos 24/7 generados por synth_msar_garch_hawkes.py

Entradas mínimas en la carpeta del mundo:
  - SYN_PRICE.npy
  - meta.json   (contiene step_min; si no existe, se asume 5)

Features por ventana (compatibles con tus modelos):
  - price (nivel)
  - sigma_rolling_60min (std de log-ret con ventana retrospectiva de 60 min)

Ventanas con left-pad de NaN para mantener longitud fija.

Salidas:
  - SYN_SERIE_5m.npy                (alias del precio para compatibilidad)
  - SYN_TS_5m.npy                   (opcional si --ts-start)
  - SYN_WINDOWS_{H}m_X.npy          (tensor [N, win_len, 2])
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

DT = np.float32
DEF_STEP_MIN = 5
ROLL_VOL_MIN = 60   # minutos para la rolling sigma (estrictamente retrospectiva)


def trailing_std_from_price(price: np.ndarray, step_min: int, roll_vol_min: int) -> np.ndarray:
    """Std retrospectiva de log-ret, ventana 'roll_vol_min' minutos."""
    logp = np.log(np.clip(price, 1e-9, None)).astype(np.float64)
    r = np.diff(logp, prepend=logp[0]).astype(np.float64)
    w = max(1, roll_vol_min // max(1, step_min))
    if w <= 1:
        sig = np.zeros_like(r, dtype=DT)
        return sig
    # std retrospectiva (sólo pasado) con cumsums
    c1 = np.cumsum(np.insert(r, 0, 0.0))
    c2 = np.cumsum(np.insert(r * r, 0, 0.0))
    m = (c1[w:] - c1[:-w]) / w
    v = np.maximum(0.0, (c2[w:] - c2[:-w]) / w - m * m)
    sig = np.zeros_like(r, dtype=np.float64)
    sig[w-1:] = np.sqrt(v)
    return np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0).astype(DT)


def make_windows(price: np.ndarray, win_len: int, step_min: int) -> np.ndarray:
    """Construye ventanas con features [price, sigma_rolling_60m]. Left-pad con NaN."""
    n = price.shape[0]
    sigma = trailing_std_from_price(price, step_min=step_min, roll_vol_min=ROLL_VOL_MIN)
    X = []
    for t in range(n):
        s = max(0, t - win_len + 1)
        w_p = price[s:t + 1].astype(DT)
        w_s = sigma[s:t + 1].astype(DT)
        pad = win_len - w_p.shape[0]
        if pad > 0:
            wp = np.concatenate([np.full(pad, np.nan, dtype=DT), w_p])
            ws = np.concatenate([np.full(pad, np.nan, dtype=DT), w_s])
        else:
            wp, ws = w_p, w_s
        X.append(np.stack([wp, ws], axis=-1))
    return np.stack(X, axis=0) if X else np.empty((0, win_len, 2), dtype=DT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True, help="Carpeta del mundo sintético.")
    ap.add_argument("--dst", default=None, help="Carpeta destino (por defecto, la misma del mundo).")
    ap.add_argument("--horizon-min", type=int, required=True,
                    help="Horizonte H en minutos. Se usa win_len = 2*(H/step_min).")
    ap.add_argument("--ts-start", default=None,
                    help='Opcional: timestamp inicial para generar TS (ej: "2010-01-01T00:00:00").')
    ap.add_argument("--step-min", type=int, default=None,
                    help="Override de step_min. Si no, se lee de meta.json; si no, 5.")
    args = ap.parse_args()

    world = Path(args.world).expanduser().resolve()
    p_price = world / "SYN_PRICE.npy"
    if not p_price.exists():
        raise SystemExit(f"No encuentro {p_price}")

    # meta.json o legacy
    meta = {}
    for fn in ["meta.json", "SYN_META.json"]:
        f = world / fn
        if f.exists():
            try:
                meta = json.loads(f.read_text())
            except Exception:
                meta = {}
            break

    step_min = int(args.step_min if args.step_min is not None else meta.get("step_min", DEF_STEP_MIN))
    if step_min <= 0:
        raise SystemExit("step_min inválido (<=0).")

    price = np.load(p_price).astype(np.float64)

    dst = Path(args.dst).expanduser().resolve() if args.dst else world
    dst.mkdir(parents=True, exist_ok=True)

    H = int(args.horizon_min)
    if H % step_min != 0:
        raise SystemExit(f"--horizon-min debe ser múltiplo de step_min ({step_min}).")
    win_len = 2 * (H // step_min)

    # Aliases de compatibilidad
    np.save(dst / "SYN_SERIE_5m.npy", price.astype(DT))
    if args.ts_start:
        ts = pd.date_range(start=pd.Timestamp(args.ts_start), periods=price.size, freq=f"{step_min}T")
        np.save(dst / "SYN_TS_5m.npy", ts.to_numpy(dtype="datetime64[ns]"))

    X = make_windows(price, win_len=win_len, step_min=step_min)
    outX = dst / f"SYN_WINDOWS_{H}m_X.npy"
    # memmap por si son grandes
    mm = np.lib.format.open_memmap(str(outX), mode="w+", dtype=DT, shape=X.shape)
    mm[:] = X

    print(f"✔ Ventanas creadas")
    print(f"  world: {world}")
    print(f"  barras={price.size:,}  step_min={step_min}  H={H}  win_len={win_len}")
    print(f"  X.shape={X.shape}  → {outX}")

if __name__ == "__main__":
    main()

