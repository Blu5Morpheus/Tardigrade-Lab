"""Demo 01 — Clifford-Geometric VQE classifier for gravitational-wave sources.

Hybrid quantum-classical pipeline:

  Classical side
    LIGO h(t) → PyCBC whitening / bandpass / PSD (fixture-time, see scripts/)
              → Clifford geometric encoding in Cl(3,0)
                  scalar / vector (e1,e2,e3) / bivector (e12,e13,e23) / pseudoscalar (e123)

  Quantum side
    PennyLane VQC, target backend `ibm_kingston` (simulated locally on
    lightning.qubit). Equivariant ansatz built from Cl(3,0) bivector
    generators in the σ_i⊗σ_j spin representation, plus a Hodge-dual
    generator G_hodge = X₀Y₁X₂ to break symmetry-locking. Parameter-shift
    gradients close the loop with a classical Adam / SPSA optimizer.

  Classes
    Binary fixture (current default):
      0 = glitch, 1 = signal (BBH-like chirp)
    Four-class fixture (scripts/generate_4class_ligo.py):
      0 = BBH, 1 = BNS, 2 = ECO, 3 = Beyond-GR

The legacy "Standard" path (Angle/Amplitude/IQP embeddings into
StronglyEntanglingLayers) is preserved as an A/B comparison.
"""

from __future__ import annotations

import inspect
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import pennylane as qml

from lib.theme import PALETTE, apply_plotly_theme

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "data" / "ligo_strain_sample.npz"
FIXTURE_4CLASS = Path(__file__).resolve().parent.parent / "data" / "ligo_4class.npz"

CLASS_NAMES_4 = ["BBH", "BNS", "ECO", "Beyond-GR"]
CLASS_NAMES_2 = ["glitch", "chirp"]


# ─────────────────────────────────────────────────────────────────────────
# Fixture loading
# ─────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def load_binary_fixture():
    if not FIXTURE_PATH.exists():
        return None
    npz = np.load(FIXTURE_PATH, allow_pickle=False)
    out = {"signals": npz["signals"], "glitches": npz["glitches"]}
    out["source"] = str(npz["source"]) if "source" in npz.files else "unknown"
    out["sample_rate"] = int(npz["sample_rate"]) if "sample_rate" in npz.files else 4096
    out["n_real_events"] = int(npz["n_real_events"]) if "n_real_events" in npz.files else 0
    return out


@st.cache_data(ttl=600, show_spinner=False)
def load_4class_fixture():
    if not FIXTURE_4CLASS.exists():
        return None
    npz = np.load(FIXTURE_4CLASS, allow_pickle=False)
    out = {"windows": npz["windows"], "labels": npz["labels"].astype(int)}
    out["source"] = str(npz["source"]) if "source" in npz.files else "synthetic"
    out["sample_rate"] = int(npz["sample_rate"]) if "sample_rate" in npz.files else 4096
    return out


def _fetch_real_ligo_data() -> tuple[bool, str]:
    """Run scripts/fetch_ligo_real.py to materialize a real-data fixture."""
    import subprocess
    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "scripts" / "fetch_ligo_real.py"
    out = repo_root / "data" / "ligo_strain_sample.npz"
    if not script.exists():
        return False, f"Fetch script not found at {script}"
    try:
        proc = subprocess.run(
            ["python", str(script), "--output", str(out)],
            cwd=str(repo_root), capture_output=True, text=True, timeout=900,
        )
    except subprocess.TimeoutExpired:
        return False, "Fetch timed out (>15 min). GWOSC may be slow — try again later."
    if proc.returncode != 0:
        return False, f"Fetch failed: {proc.stderr.strip()[-400:]}"
    return True, f"Wrote {out}"


# ─────────────────────────────────────────────────────────────────────────
# Classical preprocessing — Cl(3,0) geometric encoder
# ─────────────────────────────────────────────────────────────────────────

def _clifford_encode(strain: np.ndarray) -> np.ndarray:
    """Map a 256-sample whitened strain window to a Cl(3,0) multivector.

    The 8-component output is laid out as
        [scalar, e1, e2, e3, e12, e13, e23, e123]
    so downstream code can index basis blades by bitmask
        (e1=1, e2=2, e12=3, e3=4, e13=5, e23=6, e123=7)
    via the same convention used by the Clifford-agent demo.

    Encoding rule (chosen so each blade carries distinct physical content):
      - 3 temporal sub-windows → band amplitudes  b₁, b₂, b₃   (vector part)
      - pairwise band correlations → bivector components       (rotational content)
      - total RMS → scalar
      - signed triple-band skewness → pseudoscalar             (chirality / asymmetry)
    All components are tanh-bounded into (-π, π) so they're directly usable
    as rotation angles in the variational circuit.
    """
    s = np.asarray(strain, dtype=np.float32)
    bands = np.array_split(s, 3)
    b = np.array([np.sqrt(np.mean(c ** 2) + 1e-12) for c in bands])    # b1, b2, b3
    # standardize bands so RY rotations are well-conditioned
    b = (b - b.mean()) / (b.std() + 1e-9)

    scalar = float(np.sqrt(np.mean(s ** 2) + 1e-12))
    scalar = (scalar - 0.05) / 0.05   # ≈ unit-scaled

    # bivector = pairwise band correlation, sign-preserving
    def corr(u: np.ndarray, v: np.ndarray) -> float:
        n = min(len(u), len(v))
        u, v = u[:n], v[:n]
        cu = u - u.mean(); cv = v - v.mean()
        denom = (np.linalg.norm(cu) * np.linalg.norm(cv) + 1e-9)
        return float((cu @ cv) / denom)

    e12 = corr(bands[0], bands[1])
    e13 = corr(bands[0], bands[2])
    e23 = corr(bands[1], bands[2])

    # pseudoscalar — signed triple-product / asymmetry across bands
    e123 = float(np.cbrt(b[0] * b[1] * b[2]))

    mv = np.array([scalar, b[0], b[1], b[2], e12, e13, e23, e123], dtype=np.float32)
    return np.tanh(mv) * np.pi


def _features_from_strain(strain: np.ndarray, n_qubits: int, encoding: str) -> np.ndarray:
    """Legacy encoders — Angle / Amplitude / IQP. Kept for the A/B comparison."""
    if encoding == "Amplitude":
        target = 1 << n_qubits
        if strain.size >= target:
            chunks = np.array_split(strain, target)
            feat = np.array([c.mean() for c in chunks])
        else:
            feat = np.pad(strain, (0, target - strain.size))
        return feat / (np.linalg.norm(feat) + 1e-12)
    chunks = np.array_split(strain, n_qubits)
    feats = np.array([c.mean() for c in chunks])
    feats = np.clip(feats / (np.std(feats) + 1e-12), -np.pi, np.pi)
    return feats


# ─────────────────────────────────────────────────────────────────────────
# Quantum side — circuits
# ─────────────────────────────────────────────────────────────────────────

def _build_legacy_circuit(n_qubits: int, depth: int, encoding: str, n_classes: int):
    """Original StronglyEntanglingLayers path. n_classes ∈ {2, 4}."""
    import pennylane as qml
    dev = qml.device("lightning.qubit", wires=n_qubits)

    if n_classes == 2:
        observables = [qml.PauliZ(0)]
    else:
        # 4 quasi-independent Pauli strings — pseudoscalar + three "axis" Zs
        observables = [qml.PauliZ(0), qml.PauliZ(1), qml.PauliZ(min(2, n_qubits - 1)),
                       qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(min(2, n_qubits - 1))]

    @qml.qnode(dev, interface="autograd")
    def circuit(features, weights):
        if encoding == "Angle":
            qml.AngleEmbedding(features, wires=range(n_qubits))
        elif encoding == "Amplitude":
            qml.AmplitudeEmbedding(features, wires=range(n_qubits), normalize=True)
        else:
            qml.IQPEmbedding(features, wires=range(n_qubits), n_repeats=2)
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(o) for o in observables]

    weights_shape = (depth, n_qubits, 3)
    return circuit, weights_shape


def _build_clifford_circuit(depth: int, n_classes: int, use_hodge: bool):
    """Equivariant Clifford VQC.

    Three qubits, one per Cl(3,0) vector basis. The encoding layer writes
    the 8 multivector components into the state:
      scalar     → global RZ
      e_i        → RY on qubit i
      e_ij       → ZZ(i,j)
      e_123      → triply-controlled phase (approximated by RZZZ)

    Variational layers act with bivector generators in the σ⊗σ spin rep:
      G_{ij} = X_i X_j + Y_i Y_j        (XY-Heisenberg, preserves e_ij plane)
      G_{i}  = Z_i                       (vector-axis rotation)
    The Hodge-dual generator G_hodge = X₀ Y₁ X₂ is applied once per layer
    with its own learnable angle when use_hodge is True — it mixes the
    pseudoscalar with the bivector subspace and breaks the symmetry-locking
    that pure bivector entanglers exhibit.

    Measurement:
      n_classes == 2  → ⟨Z_0⟩ only
      n_classes == 4  → ⟨Z_0⟩, ⟨Z_1⟩, ⟨Z_2⟩, ⟨Z_0 Z_1 Z_2⟩  (softmax-ready)
    """
    import pennylane as qml

    n_qubits = 3
    dev = qml.device("lightning.qubit", wires=n_qubits)

    if n_classes == 2:
        observables = [qml.PauliZ(0)]
    else:
        observables = [qml.PauliZ(0), qml.PauliZ(1), qml.PauliZ(2),
                       qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2)]

    n_bivectors = 3
    n_vectors = 3
    n_params_per_layer = n_bivectors + n_vectors + (1 if use_hodge else 0)
    weights_shape = (depth, n_params_per_layer)

    @qml.qnode(dev, interface="autograd")
    def circuit(features, weights):
        # features: 8-component multivector [scalar, e1, e2, e3, e12, e13, e23, e123]
        scalar, e1, e2, e3, e12, e13, e23, e123 = [features[i] for i in range(8)]

        # ── classical → quantum encoding (geometric)
        for w in range(n_qubits):
            qml.RZ(scalar, wires=w)
        qml.RY(e1, wires=0)
        qml.RY(e2, wires=1)
        qml.RY(e3, wires=2)
        qml.IsingZZ(e12, wires=[0, 1])
        qml.IsingZZ(e13, wires=[0, 2])
        qml.IsingZZ(e23, wires=[1, 2])
        # pseudoscalar — triple-Z phase. Trotterized as ZZ(0,1) sandwiched
        # by CNOTs targeting qubit 2 (exact for diagonal generators).
        qml.CNOT(wires=[2, 1])
        qml.IsingZZ(e123, wires=[0, 1])
        qml.CNOT(wires=[2, 1])

        # ── variational layers — Clifford bivector generators + Hodge-dual
        for d in range(depth):
            p = weights[d]
            # bivector generators (XY-Heisenberg-style couplings)
            qml.IsingXX(p[0], wires=[0, 1])
            qml.IsingYY(p[0], wires=[0, 1])
            qml.IsingXX(p[1], wires=[0, 2])
            qml.IsingYY(p[1], wires=[0, 2])
            qml.IsingXX(p[2], wires=[1, 2])
            qml.IsingYY(p[2], wires=[1, 2])
            # vector-axis rotations
            qml.RZ(p[3], wires=0)
            qml.RZ(p[4], wires=1)
            qml.RZ(p[5], wires=2)
            if use_hodge:
                # Hodge-dual generator: G_hodge = X₀ Y₁ X₂
                # exp(-i θ G_hodge) on |ψ⟩ via PauliRot
                qml.PauliRot(p[6], "XYX", wires=[0, 1, 2])

        return [qml.expval(o) for o in observables]

    return circuit, weights_shape


# ─────────────────────────────────────────────────────────────────────────
# Training — binary (MSE) and multi-class (softmax cross-entropy)
# ─────────────────────────────────────────────────────────────────────────

def _train(circuit, weights_shape, X, y, n_classes, optimizer, lr, epochs, seed):
    import pennylane as qml
    from pennylane import numpy as pnp
    import time

    rng = np.random.default_rng(int(seed))
    weights = pnp.array(rng.normal(0, 0.1, weights_shape), requires_grad=True)

    def _circuit_out(x, w):
        out = circuit(x, w)
        return pnp.stack([pnp.asarray(o) for o in out]) if isinstance(out, list) else pnp.asarray(out)

    # Internal helper to run inference in small chunks to prevent CPU plateauing
    def _get_preds_safe(X_data, w_data):
        all_preds = []
        chunk_size = 10 # Small chunks to let the CPU breathe
        for i in range(0, len(X_data), chunk_size):
            chunk = X_data[i : i + chunk_size]
            preds = [_circuit_out(x, w_data) for x in chunk]
            all_preds.extend(preds)
            time.sleep(0.01) # Force release of CPU every 10 samples
        return pnp.stack(all_preds)

    if n_classes == 2:
        def loss_fn(w, batch_X, batch_y):
            # Training uses smaller batches (defined in loop), so this is safe
            preds = pnp.stack([_circuit_out(x, w) for x in batch_X])
            targets = 2 * batch_y - 1
            return pnp.mean((preds - targets) ** 2)
    else:
        def loss_fn(w, batch_X, batch_y):
            logits = pnp.stack([_circuit_out(x, w) for x in batch_X])
            shift = pnp.max(logits, axis=1, keepdims=True)
            e = pnp.exp(logits - shift)
            probs = e / pnp.sum(e, axis=1, keepdims=True)
            idx = batch_y.astype(int)
            picked = pnp.stack([probs[i, idx[i]] for i in range(len(idx))])
            return -pnp.mean(pnp.log(picked + 1e-12))

    # Optimizer selection remains the same
    if optimizer == "Adam":
        opt = qml.AdamOptimizer(stepsize=lr)
    elif optimizer == "Nesterov":
        opt = qml.NesterovMomentumOptimizer(stepsize=lr)
    else:
        opt = qml.SPSAOptimizer(maxiter=epochs)

    history: list[dict] = []
    
    for ep in range(epochs):
        # Keep batch_size small (default 32 is okay, 16 is safer for 1-CPU)
        batch_size = min(16, len(X))
        idx = rng.choice(len(X), size=batch_size, replace=False)
        bX = pnp.array(X[idx], requires_grad=False)
        by = pnp.array(y[idx], requires_grad=False)

        if optimizer == "SPSA":
            weights = opt.step(lambda w: loss_fn(w, bX, by), weights)
            loss_val = loss_fn(weights, bX, by)
        else:
            # Adam/Nesterov use parameter-shift: this is CPU intensive!
            weights, loss_val = opt.step_and_cost(lambda w: loss_fn(w, bX, by), weights)

        loss_item = float(qml.math.to_numpy(loss_val))

        # Accuracy check every 5 epochs using the 'Safe' chunked method
        if ep % 5 == 0 or ep == (epochs - 1):
            preds_full = _get_preds_safe(X, weights)
            if n_classes == 2:
                acc = float(qml.math.to_numpy(((preds_full[:, 0] > 0).astype(int) == y).mean()))
            else:
                preds_cls = pnp.argmax(preds_full, axis=1)
                acc = float(qml.math.to_numpy((preds_cls == y).mean()))
        else:
            acc = history[-1]["accuracy"] if history else 0.0

        history.append({"epoch": ep, "loss": loss_item, "accuracy": acc})
        time.sleep(0.05) # Increased rest time between epochs

    # Final predictions using the safe chunked method
    final_raw = _get_preds_safe(X, weights)
    final = np.asarray(qml.math.to_numpy(final_raw))
    
    return np.asarray(qml.math.to_numpy(weights)), history, final


# ─────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────

def render(embed: bool = False) -> None:
    if not embed:
        st.caption(
            "Clifford-Geometric VQE — hybrid quantum-classical classifier on "
            "LIGO O3 strain. Classical Cl(3,0) encoder + PennyLane VQC with "
            "bivector-generator ansatz and Hodge-dual symmetry breaker. "
            "Target hardware: IBM `ibm_kingston`; this UI runs lightning.qubit."
        )

    # ── pick fixture / class count
    fx_bin = load_binary_fixture()
    fx_4 = load_4class_fixture()
    options = []
    if fx_4 is not None:
        options.append("4-class (BBH / BNS / ECO / Beyond-GR)")
    if fx_bin is not None:
        options.append("2-class (chirp / glitch)")
    if not options:
        st.warning(
            "**No strain fixture present.** Run one of:\n"
            "- `python scripts/fetch_ligo_real.py --output data/ligo_strain_sample.npz` (real O3)\n"
            "- `python scripts/generate_ligo_fixture.py --output data/ligo_strain_sample.npz` (synthetic 2-class)\n"
            "- `python scripts/generate_4class_ligo.py --output data/ligo_4class.npz` (synthetic 4-class)"
        )
        if not embed and st.button("Fetch real LIGO data now (5–10 min)"):
            with st.spinner("Downloading O3 strain windows from GWOSC…"):
                ok, msg = _fetch_real_ligo_data()
            if ok:
                st.success(msg + " — reload the page.")
                load_binary_fixture.clear()
            else:
                st.error(msg)
        return

    fixture_choice = st.radio("Fixture", options, horizontal=True)
    is_4class = fixture_choice.startswith("4-class")
    n_classes = 4 if is_4class else 2
    class_names = CLASS_NAMES_4 if is_4class else CLASS_NAMES_2

    if is_4class:
        windows = fx_4["windows"]
        labels_all = fx_4["labels"]
        src = fx_4["source"]
        n_real = 0
        sr = fx_4["sample_rate"]
    else:
        sigs, glis = fx_bin["signals"], fx_bin["glitches"]
        windows = np.vstack([sigs, glis])
        labels_all = np.concatenate([
            np.ones(len(sigs), dtype=int),
            np.zeros(len(glis), dtype=int),
        ])
        src = fx_bin["source"]
        n_real = fx_bin["n_real_events"]
        sr = fx_bin["sample_rate"]

    cols = st.columns(4)
    src_label = {"gwosc-real-o3": "GWOSC O3 (real)",
                 "synthetic-pycbc": "PyCBC (synthetic)"}.get(src, src)
    cols[0].metric("Data source", src_label)
    cols[1].metric("Real events", n_real)
    cols[2].metric("Sample rate", f"{sr} Hz")
    cols[3].metric("Classes", str(n_classes))

    # ── sidebar
    with st.sidebar:
        st.markdown("### Pipeline")
        encoder = st.radio(
            "Encoder",
            ["Clifford Cl(3,0)", "Standard"],
            help="Clifford = classical geometric encoder feeding a 3-qubit "
                 "equivariant ansatz. Standard = legacy Angle/Amplitude/IQP "
                 "embedding into StronglyEntanglingLayers.",
        )
        if encoder == "Standard":
            n_qubits = st.slider("Qubits", 2, 6, 4)
            encoding = st.radio("Embedding", ["Angle", "Amplitude", "IQP-style"])
            use_hodge = False
        else:
            n_qubits = 3
            encoding = "Clifford"
            use_hodge = st.checkbox(
                "Include Hodge-dual generator (X₀Y₁X₂)",
                value=True,
                help="Adds a learnable PauliRot(XYX) per layer. Without it "
                     "the ansatz stays locked inside the bivector subspace.",
            )

        depth = st.slider("Layers", 1, 6, 3)
        n_train = st.slider(
            "Training samples",
            min(40, len(windows)), min(500, len(windows)),
            min(200, len(windows)),
        )
        optimizer = st.radio("Optimizer", ["Adam", "Nesterov", "SPSA"], horizontal=True)
        lr = st.number_input("Learning rate", 0.001, 0.5, 0.05, 0.001, format="%.3f")
        epochs = st.slider("Epochs", 5, 50, 15)
        seed = st.number_input("Seed", value=42, step=1)
        run = st.button("Train classifier", type="primary", use_container_width=True)

    # ── build balanced training matrix
    rng = np.random.default_rng(int(seed))
    per_class = max(2, n_train // n_classes)
    chosen_idx: list[int] = []
    chosen_y: list[int] = []
    for c in range(n_classes):
        cls_mask = np.where(labels_all == c)[0]
        if len(cls_mask) == 0:
            continue
        pick = rng.choice(cls_mask, size=min(per_class, len(cls_mask)), replace=False)
        chosen_idx.extend(pick.tolist())
        chosen_y.extend([c] * len(pick))
    chosen_idx = np.array(chosen_idx)
    perm = rng.permutation(len(chosen_idx))
    raw = windows[chosen_idx[perm]]
    labels = np.array(chosen_y, dtype=int)[perm]

    # ── feature extraction
    if encoder == "Clifford Cl(3,0)":
        X = np.array([_clifford_encode(s) for s in raw])
    else:
        enc_key = encoding.replace("-style", "")
        X = np.array([_features_from_strain(s, n_qubits, enc_key) for s in raw])

    # ── build circuit
    if encoder == "Clifford Cl(3,0)":
        circuit, weights_shape = _build_clifford_circuit(int(depth), n_classes, use_hodge)
    else:
        circuit, weights_shape = _build_legacy_circuit(
            int(n_qubits), int(depth), encoding.replace("-style", ""), n_classes,
        )

    if run:
        with st.spinner("Compiling and training the variational circuit…"):
            try:
                weights, history, preds = _train(
                    circuit, weights_shape, X, labels,
                    n_classes, optimizer, lr, int(epochs), int(seed),
                )
            except Exception as e:
                st.error(f"Training failed: {e}")
                return
        st.session_state.vqe_results = dict(
            weights=np.asarray(weights), history=history, preds=preds, X=X, y=labels,
            n_classes=n_classes, class_names=class_names, encoder=encoder,
        )

    if "vqe_results" not in st.session_state:
        st.info("Configure pipeline + hyperparameters in the sidebar, then click **Train classifier**.")
        return

    res = st.session_state.vqe_results
    history = res["history"]

    # learning curves
    losses = [h["loss"] for h in history]
    accs = [h["accuracy"] for h in history]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=losses, name="loss", line=dict(color=PALETTE["mint_rim"])))
    fig.add_trace(go.Scatter(y=accs, name="accuracy", line=dict(color=PALETTE["violet_glow"]), yaxis="y2"))
    fig.update_layout(
        title="Training", xaxis_title="epoch",
        yaxis=dict(title="loss"),
        yaxis2=dict(title="accuracy", overlaying="y", side="right", range=[0, 1]),
        height=320,
    )
    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    # ── per-class metrics
    if res["n_classes"] == 2:
        preds_bin = (res["preds"] > 0).astype(int)
        cm_mat = np.zeros((2, 2), dtype=int)
        for actual, pred in zip(res["y"], preds_bin):
            cm_mat[int(actual), int(pred)] += 1
        # ROC needs scores ∈ [0,1]
        scores = (res["preds"] + 1) / 2
    else:
        preds_cls = res["preds"].argmax(axis=1)
        cm_mat = np.zeros((res["n_classes"], res["n_classes"]), dtype=int)
        for actual, pred in zip(res["y"], preds_cls):
            cm_mat[int(actual), int(pred)] += 1
        scores = None

    test_acc = float(np.trace(cm_mat) / max(1, cm_mat.sum()))
    metric_cols = st.columns(4)
    metric_cols[0].metric("Accuracy", f"{test_acc * 100:.1f}%")
    metric_cols[1].metric("Encoder", res["encoder"])
    metric_cols[2].metric("Classes", res["n_classes"])
    metric_cols[3].metric("Samples", int(cm_mat.sum()))

    col1, col2 = st.columns(2)
    with col1:
        cm = go.Figure(data=go.Heatmap(
            z=cm_mat[::-1],
            x=[f"pred · {n}" for n in res["class_names"]],
            y=[f"actual · {n}" for n in res["class_names"][::-1]],
            text=cm_mat[::-1].astype(str),
            texttemplate="%{text}",
            colorscale=[[0, PALETTE["bg_panel"]], [1, PALETTE["violet_glow"]]],
            showscale=False,
        ))
        cm.update_layout(title="Confusion matrix", height=360)
        apply_plotly_theme(cm)
        st.plotly_chart(cm, use_container_width=True)

    with col2:
        if res["n_classes"] == 2 and scores is not None:
            order = np.argsort(-scores)
            ys = res["y"][order]
            tpr = np.cumsum(ys) / max(1, ys.sum())
            fpr = np.cumsum(1 - ys) / max(1, (1 - ys).sum())
            auc = float(np.trapz(tpr, fpr))
            roc = go.Figure()
            roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {auc:.3f}",
                                     line=dict(color=PALETTE["violet_glow"])))
            roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     line=dict(color=PALETTE["violet_deep"], dash="dash"),
                                     showlegend=False))
            roc.update_layout(title="ROC", xaxis_title="FPR", yaxis_title="TPR", height=360)
            apply_plotly_theme(roc)
            st.plotly_chart(roc, use_container_width=True)
        else:
            # per-class precision/recall
            prec, rec = [], []
            for c in range(res["n_classes"]):
                tp = cm_mat[c, c]
                fp = cm_mat[:, c].sum() - tp
                fn = cm_mat[c, :].sum() - tp
                prec.append(tp / max(1, tp + fp))
                rec.append(tp / max(1, tp + fn))
            pr = go.Figure()
            pr.add_trace(go.Bar(x=res["class_names"], y=prec, name="precision",
                                marker_color=PALETTE["violet_glow"]))
            pr.add_trace(go.Bar(x=res["class_names"], y=rec, name="recall",
                                marker_color=PALETTE["mint_rim"]))
            pr.update_layout(title="Per-class precision / recall", height=360,
                             barmode="group", yaxis=dict(range=[0, 1]))
            apply_plotly_theme(pr)
            st.plotly_chart(pr, use_container_width=True)

    # downloadable weights
    buf = BytesIO()
    np.savez(buf, weights=res["weights"], history=np.array(history))
    st.download_button(
        "Download trained weights (.npz)",
        buf.getvalue(),
        file_name=f"vqe_weights_{res['encoder'].replace(' ', '_')}_{res['n_classes']}cls.npz",
        mime="application/octet-stream",
    )

    if not embed:
        with st.expander("Show circuit + encoder code"):
            st.code(
                inspect.getsource(_clifford_encode) + "\n\n"
                + inspect.getsource(_build_clifford_circuit) + "\n\n"
                + inspect.getsource(_train),
                language="python",
            )
