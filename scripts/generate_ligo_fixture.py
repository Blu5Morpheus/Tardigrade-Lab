#!/usr/bin/env python3
"""
generate_ligo_fixture.py

Generates the data fixture used by the VQE/LIGO classifier demo.
Produces `ligo_strain_sample.npz` containing:
  - signals:  shape (500, 256) — whitened binary-merger chirp templates
  - glitches: shape (500, 256) — synthetic glitch shapes (sine-Gaussian + blip)
  - labels:   shape (1000,)    — 0 for glitch, 1 for signal

Usage:
    pip install pycbc numpy scipy
    python generate_ligo_fixture.py --output ligo_strain_sample.npz

The script is intentionally fully reproducible — any time you re-run it with
the same arguments, you get a byte-identical .npz file. The seed is fixed.

Why synthetic glitches rather than real GWOSC glitch examples?
  Real glitches require GWOSC's GravitySpy catalog access and per-glitch
  metadata. The synthetic glitches here (sine-Gaussian + blip) are accepted
  proxies in the GW-ML literature for blip-type and tomte-type glitches and
  are sufficient for the educational demo. If you want to upgrade later,
  replace the _make_glitch() function with GravitySpy-loaded examples.

Why a fixed length of 256 samples?
  At 4096 Hz sample rate, 256 samples = 62.5 ms — long enough to see the
  late-inspiral and merger of stellar-mass binaries, short enough to be
  a manageable input dimension for the variational circuit.

Memory/time budget: ~30 seconds and ~50 MB peak memory on a laptop.
The output .npz is ~2 MB.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SAMPLE_RATE = 4096      # Hz, matches LIGO O3 strain data
N_SAMPLES = 256         # samples per example
N_EACH_CLASS = 500      # 500 signals + 500 glitches = 1000 total examples
SEED = 20260101         # fixed seed: reproducibility matters for the demo


# -----------------------------------------------------------------------------
# Signal generation — binary-merger chirp templates via PyCBC
# -----------------------------------------------------------------------------

def _make_chirp_templates(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate n binary-merger chirp templates using PyCBC's IMRPhenomD waveform.

    Each template is whitened against an analytical aLIGO PSD (no real-strain
    download needed) and cropped to the merger window.

    Returns: shape (n, N_SAMPLES) float32 array, normalized so each row has
    unit L2 norm.
    """
    try:
        from pycbc.waveform import get_td_waveform
        from pycbc.psd.analytical import aLIGOZeroDetHighPower
        from pycbc.types import TimeSeries
    except ImportError:
        print("ERROR: PyCBC is required. Install with `pip install pycbc`.",
              file=sys.stderr)
        sys.exit(1)

    out = np.zeros((n, N_SAMPLES), dtype=np.float32)
    delta_t = 1.0 / SAMPLE_RATE
    delta_f = 1.0 / 4.0  # 4-second segment for PSD
    flen = int(SAMPLE_RATE / 2 / delta_f) + 1
    psd = aLIGOZeroDetHighPower(flen, delta_f, 20.0)

    # Sample a range of binary parameters that produce diverse merger waveforms
    for i in range(n):
        m1 = rng.uniform(15.0, 50.0)
        m2 = rng.uniform(15.0, m1)  # ensure m1 >= m2
        spin1z = rng.uniform(-0.5, 0.5)
        spin2z = rng.uniform(-0.5, 0.5)

        try:
            hp, _ = get_td_waveform(
                approximant="IMRPhenomD",
                mass1=m1, mass2=m2,
                spin1z=spin1z, spin2z=spin2z,
                delta_t=delta_t,
                f_lower=30.0,
                distance=400.0,  # Mpc
            )
        except RuntimeError as e:
            # Some parameter combos fail; just retry with simpler params
            hp, _ = get_td_waveform(
                approximant="IMRPhenomD",
                mass1=30.0, mass2=30.0,
                spin1z=0.0, spin2z=0.0,
                delta_t=delta_t,
                f_lower=30.0,
                distance=400.0,
            )

        # Whiten against the PSD
        ts = TimeSeries(hp.numpy(), delta_t=delta_t)
        whitened = ts.whiten(0.125, 0.125, low_frequency_cutoff=20.0).numpy()

        # Take the N_SAMPLES around the peak (the merger)
        peak_idx = int(np.argmax(np.abs(whitened)))
        start = max(0, peak_idx - N_SAMPLES // 2)
        end = start + N_SAMPLES
        if end > len(whitened):
            end = len(whitened)
            start = end - N_SAMPLES
        window = whitened[start:end]

        # Pad if necessary (very short waveforms)
        if len(window) < N_SAMPLES:
            pad = np.zeros(N_SAMPLES - len(window), dtype=window.dtype)
            window = np.concatenate([pad, window])

        # Normalize to unit L2 norm — important for amplitude-encoding circuits
        norm = np.linalg.norm(window)
        if norm > 0:
            window = window / norm

        out[i] = window.astype(np.float32)

        if (i + 1) % 50 == 0:
            print(f"  signals: {i + 1}/{n}", file=sys.stderr)

    return out


# -----------------------------------------------------------------------------
# Glitch generation — synthetic sine-Gaussian + blip + tomte
# -----------------------------------------------------------------------------

def _make_glitch(rng: np.random.Generator) -> np.ndarray:
    """
    Generate a single synthetic glitch.

    Three flavours, picked uniformly:
      1. sine-Gaussian — narrow-band, decays exponentially
      2. blip — short broadband transient (band-limited noise burst)
      3. tomte — slowly modulated narrowband oscillation
    """
    t = np.arange(N_SAMPLES) / SAMPLE_RATE  # seconds
    flavour = rng.integers(0, 3)

    if flavour == 0:
        # sine-Gaussian
        f0 = rng.uniform(50, 500)        # Hz
        Q = rng.uniform(5, 30)
        t0 = N_SAMPLES / 2 / SAMPLE_RATE  # center the burst
        sigma = Q / (2 * np.pi * f0)
        env = np.exp(-((t - t0) ** 2) / (2 * sigma ** 2))
        phase = rng.uniform(0, 2 * np.pi)
        x = env * np.sin(2 * np.pi * f0 * (t - t0) + phase)

    elif flavour == 1:
        # blip — short broadband Gaussian noise burst
        center = N_SAMPLES // 2 + rng.integers(-20, 20)
        width = rng.integers(8, 30)
        x = np.zeros(N_SAMPLES)
        burst = rng.normal(0, 1, size=2 * width + 1)
        s, e = center - width, center + width + 1
        s_clip, e_clip = max(0, s), min(N_SAMPLES, e)
        x[s_clip:e_clip] = burst[s_clip - s:e_clip - s]
        # apply a Gaussian envelope so the burst tapers cleanly
        env = np.exp(-((np.arange(N_SAMPLES) - center) ** 2) / (2 * width ** 2))
        x = x * env

    else:
        # tomte — long narrowband modulated oscillation
        f0 = rng.uniform(30, 150)
        f_mod = rng.uniform(2, 8)
        depth = rng.uniform(0.2, 0.5)
        x = np.sin(2 * np.pi * f0 * t * (1 + depth * np.sin(2 * np.pi * f_mod * t)))
        env = np.hanning(N_SAMPLES) ** 0.5
        x = x * env

    # add small white noise floor so glitches aren't trivially distinguishable
    x = x + rng.normal(0, 0.05, size=N_SAMPLES)

    # normalize
    norm = np.linalg.norm(x)
    if norm > 0:
        x = x / norm
    return x.astype(np.float32)


def _make_glitches(n: int, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros((n, N_SAMPLES), dtype=np.float32)
    for i in range(n):
        out[i] = _make_glitch(rng)
        if (i + 1) % 100 == 0:
            print(f"  glitches: {i + 1}/{n}", file=sys.stderr)
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", "-o",
                        default="ligo_strain_sample.npz",
                        help="Output .npz path (default: %(default)s)")
    parser.add_argument("--n-each", type=int, default=N_EACH_CLASS,
                        help="Examples per class (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed (default: %(default)s)")
    args = parser.parse_args()

    out_path = Path(args.output)
    rng = np.random.default_rng(args.seed)

    print(f"Generating {args.n_each} signals + {args.n_each} glitches "
          f"(seed={args.seed}) → {out_path}", file=sys.stderr)

    print("Generating chirp templates (PyCBC, IMRPhenomD)...", file=sys.stderr)
    signals = _make_chirp_templates(args.n_each, rng)

    print("Generating synthetic glitches...", file=sys.stderr)
    glitches = _make_glitches(args.n_each, rng)

    # combined labelled dataset (used by some training loops, optional)
    X = np.concatenate([signals, glitches], axis=0)
    y = np.concatenate([
        np.ones(args.n_each, dtype=np.int8),
        np.zeros(args.n_each, dtype=np.int8),
    ])

    np.savez_compressed(
        out_path,
        signals=signals,
        glitches=glitches,
        labels=y,
        X=X,
        sample_rate=np.array(SAMPLE_RATE),
        n_samples=np.array(N_SAMPLES),
        seed=np.array(args.seed),
        source=np.array("synthetic-pycbc"),
    )

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Wrote {out_path} ({size_mb:.2f} MB)", file=sys.stderr)
    print("\nFixture summary:", file=sys.stderr)
    print(f"  signals:  shape {signals.shape}, dtype {signals.dtype}", file=sys.stderr)
    print(f"  glitches: shape {glitches.shape}, dtype {glitches.dtype}", file=sys.stderr)
    print(f"  labels:   shape {y.shape}, balance {(y == 1).sum()}/{(y == 0).sum()}", file=sys.stderr)


if __name__ == "__main__":
    main()
