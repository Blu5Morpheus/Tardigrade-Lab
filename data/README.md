# Lab data fixtures

## `ligo_strain_sample.npz`

Real LIGO O3 strain data. Contains:

- `signals` — `(500, 256)` whitened windows around real GWTC-3 / GWTC-2.1
  / GWTC-1 confident events (cropped ±31 ms around merger peak), padded to
  500 with PyCBC IMRPhenomD synthetic templates if catalog is exhausted.
- `glitches` — `(500, 256)` real O3a detector noise with synthetic glitch
  morphology (sine-Gaussian + blip) injected at SNR 4–10. The noise floor
  is real LIGO data; the transient shape is synthetic.
- `labels` — `(1000,)` 1 for signal, 0 for glitch.

Sample rate 4096 Hz, 256 samples = 62.5 ms windows. Whitening is GWpy's
default (2-second segment, 1-second overlap, 20 Hz low-frequency cutoff).

### Build it

Real-data path (preferred — what the preprint claims):

```bash
pip install gwpy gwosc pycbc numpy scipy
python scripts/fetch_ligo_real.py --output data/ligo_strain_sample.npz
```

Network: ~50–200 MB downloaded. Time: 5–10 min.

All-synthetic fallback (no network needed — useful for offline dev):

```bash
pip install pycbc numpy scipy
python scripts/generate_ligo_fixture.py --output data/ligo_strain_sample.npz
```

### Commit the `.npz`

**Yes, commit it.** Render's build doesn't have gwpy / pycbc / lalsuite
and we do not want to install them server-side — they're heavy and the
fixture is static. The `.npz` is ~2 MB compressed; that's fine in git.
