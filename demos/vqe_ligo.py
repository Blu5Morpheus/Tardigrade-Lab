"""Demo 01 — VQE × LIGO signal classifier.

A 2–6 qubit variational quantum classifier (PennyLane lightning.qubit)
trained to discriminate binary-merger chirp signals from glitch artefacts
in LIGO O3 strain data. Operates on a precomputed strain fixture
(data/ligo_strain_sample.npz) — see scripts/generate_ligo_fixture.py.

Memory budget: ~30 MB peak with default settings.
"""

from __future__ import annotations

import inspect
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from lib.theme import PALETTE, apply_plotly_theme

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "data" / "ligo_strain_sample.npz"


@st.cache_data(ttl=600, show_spinner=False)
def load_fixture():
    if not FIXTURE_PATH.exists():
        return None
    npz = np.load(FIXTURE_PATH, allow_pickle=False)
    out = {"signals": npz["signals"], "glitches": npz["glitches"]}
    # source/provenance — tolerate older fixtures without these keys
    out["source"] = str(npz["source"]) if "source" in npz.files else "unknown"
    out["sample_rate"] = int(npz["sample_rate"]) if "sample_rate" in npz.files else 4096
    out["n_real_events"] = int(npz["n_real_events"]) if "n_real_events" in npz.files else 0
    return out


def _fetch_real_ligo_data() -> tuple[bool, str]:
    """Run scripts/fetch_ligo_real.py to materialize a real-data fixture.

    Returns (ok, message). Heavy: 5–10 min, ~100 MB downloaded. The button
    that calls this is therefore admin-gated in production.
    """
    import subprocess
    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "scripts" / "fetch_ligo_real.py"
    out = repo_root / "data" / "ligo_strain_sample.npz"
    if not script.exists():
        return False, f"Fetch script not found at {script}"
    try:
        proc = subprocess.run(
            ["python", str(script), "--output", str(out)],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        return False, "Fetch timed out (>15 min). GWOSC may be slow — try again later."
    if proc.returncode != 0:
        return False, f"Fetch failed: {proc.stderr.strip()[-400:]}"
    return True, f"Wrote {out}"


def _features_from_strain(strain: np.ndarray, n_qubits: int, encoding: str) -> np.ndarray:
    """Reduce a 256-sample strain window to n_qubits features.

    For amplitude encoding we need a 2^n_qubits-length vector (after norm).
    For angle / IQP we need n_qubits scalars.
    """
    if encoding == "Amplitude":
        target = 1 << n_qubits
        # compress strain via low-pass averaging
        if strain.size >= target:
            chunks = np.array_split(strain, target)
            feat = np.array([c.mean() for c in chunks])
        else:
            feat = np.pad(strain, (0, target - strain.size))
        return feat / (np.linalg.norm(feat) + 1e-12)
    # Angle / IQP — coarsen to n_qubits scalars
    chunks = np.array_split(strain, n_qubits)
    feats = np.array([c.mean() for c in chunks])
    # bound to [-π, π]
    feats = np.clip(feats / (np.std(feats) + 1e-12), -np.pi, np.pi)
    return feats


def _build_circuit(n_qubits: int, depth: int, encoding: str):
    import pennylane as qml
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(features, weights):
        if encoding == "Angle":
            qml.AngleEmbedding(features, wires=range(n_qubits))
        elif encoding == "Amplitude":
            qml.AmplitudeEmbedding(features, wires=range(n_qubits), normalize=True)
        else:  # IQP
            qml.IQPEmbedding(features, wires=range(n_qubits), n_repeats=2)
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    return circuit, dev


def _train(n_qubits, depth, encoding, X, y, optimizer, lr, epochs, seed):
    import pennylane as qml
    from pennylane import numpy as pnp

    rng = np.random.default_rng(int(seed))
    weights_shape = (depth, n_qubits, 3)
    weights = pnp.array(rng.normal(0, 0.1, weights_shape), requires_grad=True)
    circuit, _ = _build_circuit(n_qubits, depth, encoding)

    def loss_fn(w, batch_X, batch_y):
        preds = pnp.array([circuit(x, w) for x in batch_X])
        # convert label {0,1} → expectation in [-1, +1]
        targets = 2 * batch_y - 1
        return pnp.mean((preds - targets) ** 2)

    if optimizer == "Adam":
        opt = qml.AdamOptimizer(stepsize=lr)
    elif optimizer == "Nesterov":
        opt = qml.NesterovMomentumOptimizer(stepsize=lr)
    else:
        opt = qml.SPSAOptimizer(maxiter=epochs)

    history = []
    for ep in range(epochs):
        idx = rng.choice(len(X), size=min(32, len(X)), replace=False)
        bX = pnp.array(X[idx], requires_grad=False)
        by = pnp.array(y[idx], requires_grad=False)
        if optimizer == "SPSA":
            weights = opt.step(lambda w: loss_fn(w, bX, by), weights)
            loss_val = float(loss_fn(weights, bX, by))
        else:
            weights, loss_val = opt.step_and_cost(lambda w: loss_fn(w, bX, by), weights)
        # accuracy on full set
        full_preds = np.array([float(circuit(x, weights)) for x in X])
        acc = float(((full_preds > 0).astype(int) == y).mean())
        history.append({"epoch": ep, "loss": float(loss_val), "accuracy": acc})

    final_preds = np.array([float(circuit(x, weights)) for x in X])
    return weights, history, final_preds


def render(embed: bool = False) -> None:
    if not embed:
        st.caption(
            "4-qubit variational quantum classifier on LIGO O3 strain windows. "
            "PennyLane `lightning.qubit` simulator. The full version of this work runs on "
            "IBM `ibm_nairobi` hardware — preprint in preparation."
        )

    fixture = load_fixture()
    if fixture is None:
        st.warning(
            "**Strain fixture missing.** Run "
            "`python scripts/fetch_ligo_real.py --output data/ligo_strain_sample.npz` "
            "(real O3 data from GWOSC) or "
            "`python scripts/generate_ligo_fixture.py --output data/ligo_strain_sample.npz` "
            "(all-synthetic, no network) to materialize the fixture. The demo is otherwise ready."
        )
        if not embed and st.button("Fetch real LIGO data now (5–10 min)"):
            with st.spinner("Downloading O3 strain windows from GWOSC…"):
                ok, msg = _fetch_real_ligo_data()
            if ok:
                st.success(msg + " — reload the page.")
                load_fixture.clear()
            else:
                st.error(msg)
        return

    signals, glitches = fixture["signals"], fixture["glitches"]
    if signals.size == 0 or glitches.size == 0:
        st.error("Fixture is empty.")
        return

    # ── data-provenance banner
    src = fixture.get("source", "unknown")
    n_real = fixture.get("n_real_events", 0)
    sr = fixture.get("sample_rate", 4096)
    cols = st.columns(3)
    if src == "gwosc-real-o3":
        cols[0].metric("Data source", "GWOSC O3 (real)")
        cols[1].metric("Real events", n_real)
    elif src == "synthetic-pycbc":
        cols[0].metric("Data source", "PyCBC (synthetic)")
        cols[1].metric("Real events", "0")
    else:
        cols[0].metric("Data source", src)
        cols[1].metric("Real events", n_real)
    cols[2].metric("Sample rate", f"{sr} Hz")

    with st.sidebar:
        st.markdown("### Hyperparameters")
        n_qubits = st.slider("Qubits", 2, 6, 4)
        depth = st.slider("Layers", 1, 6, 3)
        encoding = st.radio("Encoding", ["Angle", "Amplitude", "IQP-style"], horizontal=False)
        n_train = st.slider("Training samples", 50, min(500, len(signals) + len(glitches)), 200)
        optimizer = st.radio("Optimizer", ["Adam", "Nesterov", "SPSA"], horizontal=True)
        lr = st.number_input("Learning rate", 0.001, 0.5, 0.05, 0.001, format="%.3f")
        epochs = st.slider("Epochs", 5, 50, 15)
        seed = st.number_input("Seed", value=42, step=1)
        run = st.button("Train classifier", type="primary", use_container_width=True)

    # build training matrix
    rng = np.random.default_rng(int(seed))
    n_each = n_train // 2
    sig_idx = rng.choice(len(signals), size=n_each, replace=False)
    gli_idx = rng.choice(len(glitches), size=n_each, replace=False)
    raw = np.vstack([signals[sig_idx], glitches[gli_idx]])
    labels = np.concatenate([np.ones(n_each, dtype=int), np.zeros(n_each, dtype=int)])
    perm = rng.permutation(len(raw))
    raw, labels = raw[perm], labels[perm]

    # encode
    enc_key = encoding.replace("-style", "")
    X = np.array([_features_from_strain(s, n_qubits, enc_key) for s in raw])

    if run:
        with st.spinner("Compiling and training the variational circuit…"):
            try:
                weights, history, preds = _train(
                    n_qubits, depth, enc_key, X, labels, optimizer, lr, epochs, seed
                )
            except Exception as e:
                st.error(f"Training failed: {e}")
                return
        st.session_state.vqe_results = dict(
            weights=np.asarray(weights), history=history, preds=preds, X=X, y=labels,
        )

    if "vqe_results" not in st.session_state:
        st.info("Configure hyperparameters in the sidebar, then click **Train classifier**.")
        return

    res = st.session_state.vqe_results
    history = res["history"]

    # learning curves
    losses = [h["loss"] for h in history]
    accs = [h["accuracy"] for h in history]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=losses, name="loss", line=dict(color=PALETTE["amber"])))
    fig.add_trace(go.Scatter(y=accs, name="accuracy", line=dict(color=PALETTE["phosphor"]), yaxis="y2"))
    fig.update_layout(
        title="Training", xaxis_title="epoch",
        yaxis=dict(title="loss"),
        yaxis2=dict(title="accuracy", overlaying="y", side="right", range=[0, 1]),
        height=320,
    )
    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    # confusion + ROC
    preds_bin = (res["preds"] > 0).astype(int)
    tp = int(((preds_bin == 1) & (res["y"] == 1)).sum())
    tn = int(((preds_bin == 0) & (res["y"] == 0)).sum())
    fp = int(((preds_bin == 1) & (res["y"] == 0)).sum())
    fn = int(((preds_bin == 0) & (res["y"] == 1)).sum())

    col1, col2 = st.columns(2)
    with col1:
        cm = go.Figure(data=go.Heatmap(
            z=[[tn, fp], [fn, tp]],
            x=["Pred glitch", "Pred chirp"],
            y=["Actual glitch", "Actual chirp"],
            text=[[tn, fp], [fn, tp]],
            texttemplate="%{text}",
            colorscale=[[0, PALETTE["ink_2"]], [1, PALETTE["phosphor"]]],
            showscale=False,
        ))
        cm.update_layout(title="Confusion matrix", height=320)
        apply_plotly_theme(cm)
        st.plotly_chart(cm, use_container_width=True)

    with col2:
        # ROC
        scores = (res["preds"] + 1) / 2
        order = np.argsort(-scores)
        y_sorted = res["y"][order]
        tpr = np.cumsum(y_sorted) / max(1, y_sorted.sum())
        fpr = np.cumsum(1 - y_sorted) / max(1, (1 - y_sorted).sum())
        auc = float(np.trapz(tpr, fpr))
        roc = go.Figure()
        roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {auc:.3f}",
                                 line=dict(color=PALETTE["phosphor"])))
        roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line=dict(color=PALETTE["rule"], dash="dash"),
                                 showlegend=False))
        roc.update_layout(title="ROC", xaxis_title="FPR", yaxis_title="TPR", height=320)
        apply_plotly_theme(roc)
        st.plotly_chart(roc, use_container_width=True)

    # downloadable weights
    buf = BytesIO()
    np.savez(buf, weights=res["weights"], history=np.array(history))
    st.download_button(
        "Download trained weights (.npz)",
        buf.getvalue(),
        file_name=f"vqe_weights_{n_qubits}q_{depth}d.npz",
        mime="application/octet-stream",
    )

    if not embed:
        with st.expander("Show code"):
            st.code(inspect.getsource(_build_circuit) + "\n\n" + inspect.getsource(_train), language="python")
