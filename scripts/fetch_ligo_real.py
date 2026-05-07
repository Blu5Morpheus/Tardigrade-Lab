#!/usr/bin/env python3
"""fetch_ligo_real.py — build the LIGO strain fixture from real GWOSC data.

Replaces the all-synthetic generate_ligo_fixture.py. Pulls:

  * Strain windows around confirmed O3 BBH events (GWTC-3-confident, plus
    GWTC-2.1-confident as fallback) — these become the positive class.
  * Random non-event GPS segments from O3a — these become the noise floor
    for the glitch class. We then inject sine-Gaussian / blip-shaped
    transients into that real noise. The result is a glitch class with
    *real detector colored noise* + *known glitch morphology*.
  * If GWOSC has fewer real events than N_EACH_CLASS, we top up the
    signal class with PyCBC IMRPhenomD templates (matched-filter style).

Output: data/ligo_strain_sample.npz with keys:
  signals:  (N_EACH_CLASS, 256) float32, unit-norm
  glitches: (N_EACH_CLASS, 256) float32, unit-norm
  labels:   (2*N_EACH_CLASS,)   int8, 1 for signal, 0 for glitch

Usage:
    pip install gwpy gwosc pycbc
    python scripts/fetch_ligo_real.py --output data/ligo_strain_sample.npz

Network: ~50–200 MB of strain segments downloaded.
Time: 5–10 minutes on a normal connection (GWOSC isn't fast).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

# Quiet down LIGO software's chatty deprecation warnings.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SAMPLE_RATE = 4096
N_SAMPLES = 256
N_EACH_CLASS = 500
SEED = 20260101

# Catalog priority (best-curated first)
CATALOGS = ["GWTC-3-confident", "GWTC-2.1-confident", "GWTC-1-confident"]

# Detectors to try, in order
DETECTORS = ["H1", "L1"]

# Pull a 16-second segment around each merger; whiten over the full span,
# then crop ±31 ms (256 samples at 4096 Hz) around the peak.
PRE_MERGER_SECONDS = 8
POST_MERGER_SECONDS = 8


# ─────────────────────────────────────────────────────────────────────
# GWOSC fetch helpers
# ─────────────────────────────────────────────────────────────────────

def _list_events() -> list[str]:
    from gwosc.datasets import find_datasets
    events: list[str] = []
    for cat in CATALOGS:
        try:
            events.extend(find_datasets(type="events", catalog=cat))
        except Exception as e:
            print(f"  ! could not query {cat}: {e}", file=sys.stderr)
    # de-dupe but keep first-seen order (GWTC-3 events ahead of GWTC-2.1, etc.)
    seen: set[str] = set()
    deduped: list[str] = []
    for ev in events:
        # event names look like "GW190521-v3" — strip version suffix
        base = ev.split("-v")[0]
        if base in seen:
            continue
        seen.add(base)
        deduped.append(ev)
    return deduped


def _event_strain(event: str) -> np.ndarray | None:
    """Fetch a 16-second strain segment around the merger, whiten, crop."""
    from gwosc.datasets import event_gps
    from gwpy.timeseries import TimeSeries

    try:
        gps = event_gps(event)
    except Exception:
        return None

    for detector in DETECTORS:
        try:
            ts = TimeSeries.fetch_open_data(
                detector,
                gps - PRE_MERGER_SECONDS,
                gps + POST_MERGER_SECONDS,
                sample_rate=SAMPLE_RATE,
                cache=True,
                verbose=False,
            )
        except Exception:
            continue

        try:
            white = ts.whiten(2, 1, low_frequency_cutoff=20.0)
        except Exception:
            continue

        arr = white.value.astype(np.float32)
        # crop to ±N_SAMPLES/2 around the peak in the central window
        center = len(arr) // 2
        search_radius = SAMPLE_RATE // 2  # search peak within ±0.5s of center
        lo = max(0, center - search_radius)
        hi = min(len(arr), center + search_radius)
        peak = lo + int(np.argmax(np.abs(arr[lo:hi])))
        start = peak - N_SAMPLES // 2
        end = start + N_SAMPLES
        if start < 0 or end > len(arr):
            continue
        window = arr[start:end]
        norm = float(np.linalg.norm(window))
        if norm <= 0 or not np.isfinite(norm):
            continue
        return (window / norm).astype(np.float32)
    return None


def _random_noise_segment(rng: np.random.Generator, gps_pool: list[int]) -> np.ndarray | None:
    """Fetch a 4-second noise segment, whiten, crop to N_SAMPLES."""
    from gwpy.timeseries import TimeSeries
    for _ in range(4):  # up to 4 retries with different GPS times
        gps = int(rng.choice(gps_pool))
        for detector in DETECTORS:
            try:
                ts = TimeSeries.fetch_open_data(
                    detector, gps, gps + 4,
                    sample_rate=SAMPLE_RATE, cache=True, verbose=False,
                )
                white = ts.whiten(2, 1, low_frequency_cutoff=20.0)
            except Exception:
                continue
            arr = white.value.astype(np.float32)
            if len(arr) < N_SAMPLES + 1024:
                continue
            start = rng.integers(512, len(arr) - N_SAMPLES - 512)
            window = arr[start:start + N_SAMPLES]
            return window
    return None


# ─────────────────────────────────────────────────────────────────────
# Glitch injection — synthetic morphology onto real noise
# ─────────────────────────────────────────────────────────────────────

def _make_sine_gaussian(rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(-1, 1, N_SAMPLES, dtype=np.float32)
    Q = float(rng.uniform(3, 12))
    f = float(rng.uniform(50, 250))
    tau = Q / (np.sqrt(2) * np.pi * f)
    phi = float(rng.uniform(0, 2 * np.pi))
    amplitude = float(rng.uniform(0.4, 1.0))
    return amplitude * np.exp(-(t / tau) ** 2) * np.cos(2 * np.pi * f * t + phi).astype(np.float32)


def _make_blip(rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(-1, 1, N_SAMPLES, dtype=np.float32)
    width = float(rng.uniform(0.04, 0.12))
    f = float(rng.uniform(60, 200))
    amplitude = float(rng.uniform(0.3, 1.0))
    sign = float(rng.choice([-1.0, 1.0]))
    return sign * amplitude * np.exp(-(t / width) ** 2) * np.cos(2 * np.pi * f * t).astype(np.float32)


def _inject_glitch(noise: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    morph = _make_sine_gaussian(rng) if rng.random() < 0.5 else _make_blip(rng)
    snr_target = float(rng.uniform(4.0, 10.0))
    morph_norm = float(np.linalg.norm(morph))
    noise_norm = float(np.linalg.norm(noise))
    if morph_norm <= 0 or noise_norm <= 0:
        out = noise
    else:
        out = noise + (snr_target * noise_norm / morph_norm) * morph
    n = float(np.linalg.norm(out))
    if n <= 0 or not np.isfinite(n):
        return noise.astype(np.float32)
    return (out / n).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────
# Synthetic fallback (PyCBC IMRPhenomD) — used to pad the signal class
# ─────────────────────────────────────────────────────────────────────

def _make_pycbc_chirp(rng: np.random.Generator) -> np.ndarray | None:
    try:
        from pycbc.psd.analytical import aLIGOZeroDetHighPower
        from pycbc.types import TimeSeries as PyCBCTimeSeries
        from pycbc.waveform import get_td_waveform
    except ImportError:
        return None
    delta_t = 1.0 / SAMPLE_RATE
    delta_f = 1.0 / 4.0
    flen = int(SAMPLE_RATE / 2 / delta_f) + 1
    psd = aLIGOZeroDetHighPower(flen, delta_f, 20.0)
    m1 = float(rng.uniform(15.0, 50.0))
    m2 = float(rng.uniform(15.0, m1))
    s1 = float(rng.uniform(-0.5, 0.5))
    s2 = float(rng.uniform(-0.5, 0.5))
    try:
        hp, _ = get_td_waveform(
            approximant="IMRPhenomD",
            mass1=m1, mass2=m2, spin1z=s1, spin2z=s2,
            delta_t=delta_t, f_lower=30.0, distance=400.0,
        )
    except RuntimeError:
        return None
    ts = PyCBCTimeSeries(hp.numpy(), delta_t=delta_t)
    try:
        white = ts.whiten(0.125, 0.125, low_frequency_cutoff=20.0).numpy()
    except Exception:
        return None
    peak = int(np.argmax(np.abs(white)))
    start = max(0, peak - N_SAMPLES // 2)
    end = start + N_SAMPLES
    if end > len(white):
        end = len(white)
        start = end - N_SAMPLES
    window = white[start:end]
    if len(window) < N_SAMPLES:
        window = np.concatenate([np.zeros(N_SAMPLES - len(window), dtype=window.dtype), window])
    norm = float(np.linalg.norm(window))
    if norm <= 0 or not np.isfinite(norm):
        return None
    return (window / norm).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-real", type=int, default=N_EACH_CLASS,
                    help="Cap on real-event downloads (defaults to all available).")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    print(f"Building LIGO fixture → {args.output}")
    print(f"  signals: {N_EACH_CLASS} examples, {N_SAMPLES}-sample windows")

    # ── signals: real events first
    print("\n[1/3] Discovering O3 events from GWOSC catalog…")
    events = _list_events()
    print(f"      {len(events)} events available across {', '.join(CATALOGS)}")

    real_signals: list[np.ndarray] = []
    target = min(args.max_real, len(events))
    for i, ev in enumerate(events[:target]):
        sys.stdout.write(f"\r      fetching {i + 1}/{target}: {ev:24s}")
        sys.stdout.flush()
        win = _event_strain(ev)
        if win is not None:
            real_signals.append(win)
        if len(real_signals) >= N_EACH_CLASS:
            break
    print(f"\n      {len(real_signals)} real-event signal windows OK")

    # ── pad with synthetic templates if needed
    needed = N_EACH_CLASS - len(real_signals)
    if needed > 0:
        print(f"\n[2/3] Padding signal class with {needed} PyCBC IMRPhenomD templates…")
        synth: list[np.ndarray] = []
        attempts = 0
        while len(synth) < needed and attempts < needed * 3:
            attempts += 1
            t = _make_pycbc_chirp(rng)
            if t is not None:
                synth.append(t)
            if attempts % 50 == 0:
                sys.stdout.write(f"\r      {len(synth)}/{needed}")
                sys.stdout.flush()
        print(f"\n      generated {len(synth)} synthetic templates")
        real_signals.extend(synth)
    else:
        print("\n[2/3] Skipping synthetic padding — real events sufficient.")

    signals = np.stack(real_signals[:N_EACH_CLASS]).astype(np.float32)

    # ── glitches: synthetic morphology on real noise
    print(f"\n[3/3] Building glitch class — synthetic morphology on real noise…")
    # GPS pool: random times in O3a (Apr 2019 – Oct 2019)
    O3A_START = 1238166018  # 2019-04-01 UTC
    O3A_END = 1253977218    # 2019-10-01 UTC
    gps_pool = list(range(O3A_START + 100, O3A_END - 100, 600))  # every 10 min

    glitches: list[np.ndarray] = []
    attempts = 0
    while len(glitches) < N_EACH_CLASS and attempts < N_EACH_CLASS * 3:
        attempts += 1
        noise = _random_noise_segment(rng, gps_pool)
        if noise is None:
            continue
        glitches.append(_inject_glitch(noise, rng))
        if attempts % 25 == 0:
            sys.stdout.write(f"\r      {len(glitches)}/{N_EACH_CLASS}")
            sys.stdout.flush()
    print(f"\n      {len(glitches)} glitch examples OK")

    if len(glitches) < N_EACH_CLASS:
        # fall back to fully-synthetic glitches (no real noise) for any shortfall
        print("      ! GWOSC noise pool insufficient — falling back to synthetic noise for the rest")
        while len(glitches) < N_EACH_CLASS:
            morph = _make_sine_gaussian(rng) if rng.random() < 0.5 else _make_blip(rng)
            noise = rng.standard_normal(N_SAMPLES).astype(np.float32) * 0.1
            out = noise + morph
            out /= np.linalg.norm(out)
            glitches.append(out.astype(np.float32))

    glitches_arr = np.stack(glitches[:N_EACH_CLASS]).astype(np.float32)
    labels = np.concatenate([
        np.ones(N_EACH_CLASS, dtype=np.int8),
        np.zeros(N_EACH_CLASS, dtype=np.int8),
    ])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        signals=signals,
        glitches=glitches_arr,
        labels=labels,
        sample_rate=np.array(SAMPLE_RATE),
        n_samples=np.array(N_SAMPLES),
        seed=np.array(SEED),
        source=np.array("gwosc-real-o3"),
        n_real_events=np.array(len(real_signals) - max(0, needed)),
    )
    size_mb = args.output.stat().st_size / 1024 / 1024
    print(f"\nWrote {args.output} ({size_mb:.2f} MB)")
    print(f"  signals:  shape {signals.shape}, dtype {signals.dtype}")
    print(f"  glitches: shape {glitches_arr.shape}, dtype {glitches_arr.dtype}")
    print(f"  labels:   shape {labels.shape}, balance "
          f"{int((labels == 1).sum())}/{int((labels == 0).sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
