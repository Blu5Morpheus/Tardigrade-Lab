#!/usr/bin/env python3
"""generate_4class_ligo.py — synthesize a 4-class GW source classifier fixture.

Classes (250 examples each by default, 1000 total):

  0 = BBH        — binary black hole (stellar-mass), PyCBC IMRPhenomD
  1 = BNS        — binary neutron star (low-mass long inspiral), IMRPhenomD
  2 = ECO        — exotic compact object: BBH waveform + late-time echo
                   (delayed reflected copy at t_echo ≈ 10–20 ms post-merger)
  3 = Beyond-GR  — BBH with dispersive phase correction
                   φ(f) → φ_GR(f) + α·(πMf)^β        (massive-graviton style)

Output: data/ligo_4class.npz with keys
  windows:   (4N, 256) float32   whitened, unit-norm
  labels:    (4N,)     int8
  source:    "synthetic-4class"
  sample_rate: 4096

Usage:
    pip install pycbc numpy scipy
    python scripts/generate_4class_ligo.py --output data/ligo_4class.npz

Time: ~1 minute on a laptop. Memory: ~80 MB peak.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SAMPLE_RATE = 4096
N_SAMPLES = 256
DEFAULT_PER_CLASS = 250
SEED = 20260507


def _whiten_normalize(window: np.ndarray) -> np.ndarray | None:
    """Light whitening + unit-norm crop, robust against degenerate windows."""
    w = np.asarray(window, dtype=np.float32)
    # high-pass via cumulative-trend subtraction (cheap surrogate for the
    # gwpy whitening done on real GWOSC data)
    if w.size > 8:
        trend = np.linspace(w[0], w[-1], w.size, dtype=np.float32)
        w = w - trend
    norm = float(np.linalg.norm(w))
    if norm <= 0 or not np.isfinite(norm):
        return None
    return (w / norm).astype(np.float32)


def _crop_around_peak(arr: np.ndarray) -> np.ndarray:
    peak = int(np.argmax(np.abs(arr)))
    start = max(0, peak - N_SAMPLES // 2)
    end = start + N_SAMPLES
    if end > len(arr):
        end = len(arr)
        start = end - N_SAMPLES
    window = arr[start:end]
    if len(window) < N_SAMPLES:
        window = np.concatenate([np.zeros(N_SAMPLES - len(window), dtype=window.dtype), window])
    return window


def _pycbc_waveform(m1: float, m2: float, spin1z: float, spin2z: float) -> np.ndarray | None:
    """Time-domain IMRPhenomD waveform → whitened cropped window."""
    try:
        from pycbc.waveform import get_td_waveform
    except ImportError:
        return None
    try:
        hp, _ = get_td_waveform(
            approximant="IMRPhenomD",
            mass1=m1, mass2=m2, spin1z=spin1z, spin2z=spin2z,
            delta_t=1.0 / SAMPLE_RATE, f_lower=30.0, distance=400.0,
        )
    except RuntimeError:
        return None
    arr = hp.numpy().astype(np.float32)
    return _crop_around_peak(arr)


def _make_bbh(rng: np.random.Generator) -> np.ndarray | None:
    m1 = float(rng.uniform(15.0, 50.0))
    m2 = float(rng.uniform(15.0, m1))
    s1 = float(rng.uniform(-0.5, 0.5))
    s2 = float(rng.uniform(-0.5, 0.5))
    arr = _pycbc_waveform(m1, m2, s1, s2)
    if arr is None:
        return None
    return _whiten_normalize(arr)


def _make_bns(rng: np.random.Generator) -> np.ndarray | None:
    m1 = float(rng.uniform(1.2, 2.0))
    m2 = float(rng.uniform(1.0, m1))
    s1 = float(rng.uniform(-0.05, 0.05))
    s2 = float(rng.uniform(-0.05, 0.05))
    arr = _pycbc_waveform(m1, m2, s1, s2)
    if arr is None:
        return None
    return _whiten_normalize(arr)


def _make_eco(rng: np.random.Generator) -> np.ndarray | None:
    """BBH waveform + late-time echo (a hallmark of ECO models)."""
    base = _pycbc_waveform(
        float(rng.uniform(20.0, 45.0)),
        float(rng.uniform(15.0, 35.0)),
        float(rng.uniform(-0.3, 0.3)),
        float(rng.uniform(-0.3, 0.3)),
    )
    if base is None:
        return None
    t_echo_samples = int(rng.uniform(0.010, 0.020) * SAMPLE_RATE)   # 10–20 ms
    amp = float(rng.uniform(0.3, 0.55))                              # 30–55%
    echo = np.zeros_like(base)
    if t_echo_samples < len(base):
        echo[t_echo_samples:] = amp * base[: len(base) - t_echo_samples]
    return _whiten_normalize(base + echo)


def _make_beyond_gr(rng: np.random.Generator) -> np.ndarray | None:
    """BBH waveform with dispersive phase modification.

    Apply a frequency-domain phase term φ_α(f) = α (π M_c f)^β to the FFT of
    the BBH window. Different (α, β) sample a region of the dispersion
    space — α=0 recovers GR, β=−1 mimics massive-graviton, β=2 mimics
    Lorentz-violating dispersion.
    """
    base = _pycbc_waveform(
        float(rng.uniform(20.0, 45.0)),
        float(rng.uniform(15.0, 35.0)),
        float(rng.uniform(-0.3, 0.3)),
        float(rng.uniform(-0.3, 0.3)),
    )
    if base is None:
        return None
    arr = np.asarray(base, dtype=np.float32)
    spec = np.fft.rfft(arr)
    freqs = np.fft.rfftfreq(arr.size, d=1.0 / SAMPLE_RATE)
    M_chirp_hz = 30.0  # rough chirp-mass scale in Hz⁻¹ units
    alpha = float(rng.uniform(0.6, 1.6)) * rng.choice([-1.0, 1.0])
    beta = float(rng.choice([-1.0, 2.0, 3.0]))
    # avoid f=0 divergence
    phase = alpha * np.power(np.maximum(freqs / 50.0, 1e-3), beta)
    spec_mod = spec * np.exp(1j * phase)
    arr_mod = np.fft.irfft(spec_mod, n=arr.size).astype(np.float32)
    return _whiten_normalize(arr_mod)


GENERATORS = {
    0: ("BBH",       _make_bbh),
    1: ("BNS",       _make_bns),
    2: ("ECO",       _make_eco),
    3: ("Beyond-GR", _make_beyond_gr),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--per-class", type=int, default=DEFAULT_PER_CLASS)
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    all_windows: list[np.ndarray] = []
    all_labels: list[int] = []

    for cls, (name, fn) in GENERATORS.items():
        print(f"[{cls + 1}/4] generating {args.per_class} × {name}…", file=sys.stderr)
        got = 0
        attempts = 0
        while got < args.per_class and attempts < args.per_class * 4:
            attempts += 1
            w = fn(rng)
            if w is None or w.shape != (N_SAMPLES,):
                continue
            all_windows.append(w)
            all_labels.append(cls)
            got += 1
            if got % 50 == 0:
                sys.stderr.write(f"\r      {got}/{args.per_class}\n")
        if got < args.per_class:
            print(f"  ! only got {got}/{args.per_class} for {name}", file=sys.stderr)

    windows = np.stack(all_windows).astype(np.float32)
    labels = np.array(all_labels, dtype=np.int8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        windows=windows,
        labels=labels,
        sample_rate=np.array(SAMPLE_RATE),
        n_samples=np.array(N_SAMPLES),
        seed=np.array(SEED),
        source=np.array("synthetic-4class"),
    )
    size_mb = args.output.stat().st_size / 1024 / 1024
    print(f"\nWrote {args.output} ({size_mb:.2f} MB)")
    print(f"  windows: shape {windows.shape}, dtype {windows.dtype}")
    print(f"  labels:  shape {labels.shape}, balance",
          {GENERATORS[c][0]: int((labels == c).sum()) for c in GENERATORS})
    return 0


if __name__ == "__main__":
    sys.exit(main())
