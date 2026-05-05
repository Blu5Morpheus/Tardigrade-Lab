# tardigrade-lab

Streamlit companion to the Astro site. Hosts:

1. **5 interactive demos** iframed into the Astro site at `/lab/<slug>`:
   - `vqe` — VQE × LIGO signal classifier (PennyLane lightning.qubit)
   - `clifford` — Cl(3,0) and Cl(3,1) equivariance demo
   - `amplituhedron` — cyclic-polytope amplituhedron explorer
   - `lattice` — 2D U(1) / SU(2) Wilson lattice gauge sandbox
   - `page-curve` — Haar-random Page curve simulator
2. **Me-bot** — RAG chat at `?demo=me-bot` (iframed at `/chat`).
3. **Admin panel** — `?demo=admin`, password-gated, 8 tabs.

One Streamlit app, one process, routed by `?demo=<slug>` query param.
`?embed=true` strips Streamlit chrome for clean iframe embedding.

---

## Develop

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy and fill the secrets file
cp .streamlit/secrets.toml.example .streamlit/secrets.toml

# Generate the LIGO data fixture (one-time, ~30s)
pip install pycbc
python scripts/generate_ligo_fixture.py --output data/ligo_strain_sample.npz

# Build the me-bot index (re-run after corpus edits)
python -m me_bot.reindex

# Run
streamlit run app.py
```

Visit:
- `http://localhost:8501/?demo=vqe` (or any other slug)
- `http://localhost:8501/?demo=admin` for the admin panel
- `http://localhost:8501/?demo=me-bot` for the bot

---

## Deploy

Push to GitHub, connect to Render as a web service. `render.yaml` has the
build/start commands. **Set every environment variable listed in `render.yaml`'s
comments via the Render dashboard** before the first deploy — the app refuses to
configure parts of itself that lack secrets.

The first deploy takes 3–4 minutes (PennyLane + numpy + scipy + sentence-transformers
+ torch is a chunky install). Subsequent deploys ~90s.

---

## How to add a new demo

1. Create `demos/<your_slug>.py` exposing `def render(embed: bool = False) -> None`.
2. Register in `app.py`'s `DEMO_REGISTRY`.
3. Add a corresponding `src/content/demos/<your_slug>.md` in the Astro repo.
4. Add a row to Supabase `demo_settings` (or click "Save" in the admin Demos tab).

## How to add an admin tab

1. Create `admin/<your_tab>.py` exposing `def render() -> None`.
2. Add it to `_render_dashboard()` in `admin/auth.py` (label + tab body).
3. Add any required secret to `secrets.toml.example` and `render.yaml`.

## How to update the me-bot corpus

1. Edit / add files under `me_bot/corpus/` (either locally or via the admin
   content-editor tab — the editor commits to GitHub for you).
2. Run `python -m me_bot.reindex` (or click "Rebuild index" in the admin Me-bot tab).
3. Run `python -m me_bot.eval_runner` (or "Run evals" in the admin tab).
4. Commit `me_bot/index.json`.

## Killswitch

`me_bot/.disabled` (a zero-byte file). Touch it to take the bot offline; remove
it to bring it back. The admin Me-bot → Killswitch tab does both with a checkbox.

## Memory budget

Render free tier OOM-kills above 512 MB. Working budget:

| Component | RSS |
|-----------|-----|
| Streamlit + Python baseline | ~150 MB |
| Active demo (worst case)    | ~80 MB  |
| MiniLM model loaded         | ~90 MB  |
| Index + admin panel         | ~50 MB  |
| **Total ceiling**           | **~370 MB** |

The admin **Diagnostics** tab is the source of truth for live numbers.
