"""Demo 02 — Clifford geometric algebra agent.

Two modes share the same Cl(p,q) machinery:

  1. Equivariance demonstration — pick an input x and a transform g,
     show that ‖f(g·x) − g·f(x)‖ stays at machine precision.
  2. LIGO classifier — train a small Clifford-rotor classifier on the
     LIGO O3 strain fixture (real GWOSC events vs. blip/sine-Gaussian
     glitches injected on real noise) and report identification metrics.

Multivectors are encoded as length-2^n arrays indexed by basis-blade
bitmasks. The geometric product is implemented from the signature
directly — no third-party clifford pkg.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.theme import PALETTE, apply_plotly_theme

Algebra = Literal["Cl(3,0)", "Cl(3,1)"]
Transform = Literal["Rotation", "Boost", "Reflection"]
Mode = Literal["Equivariance check", "LIGO classifier"]

LIGO_FIXTURE = Path(__file__).resolve().parent.parent / "data" / "ligo_strain_sample.npz"


# ----- algebra primitives -----

def _signature(algebra: Algebra) -> tuple[int, int, int]:
    return {"Cl(3,0)": (3, 0, 0), "Cl(3,1)": (3, 1, 0)}[algebra]


def _basis_dim(p: int, q: int, r: int = 0) -> int:
    return 1 << (p + q + r)


def _basis_grade(idx: int) -> int:
    return bin(idx).count("1")


def _blade_product(i: int, j: int, sig: tuple[int, int, int]) -> tuple[int, float]:
    """Return (resulting_basis_index, sign) for product of basis blades i, j."""
    p, q, r = sig
    n = p + q + r
    sign = 1.0
    # count swaps to merge bit-set of i with j (anti-commuting basis vectors)
    a, b = i, j
    swaps = 0
    for k in range(n):
        if not (b >> k) & 1:
            continue
        # number of bits of a above position k
        higher = bin(a >> (k + 1)).count("1")
        swaps += higher
    sign *= (-1.0) ** swaps
    # square contributions for shared basis vectors
    common = i & j
    for k in range(n):
        if (common >> k) & 1:
            if k < p:
                sign *= 1.0  # e_k² = +1
            elif k < p + q:
                sign *= -1.0  # e_k² = -1
            else:
                sign *= 0.0  # null vectors — not used here, kept for future r > 0
    return (i ^ j, sign)


def geometric_product(a: np.ndarray, b: np.ndarray, sig: tuple[int, int, int]) -> np.ndarray:
    """Multivector geometric product. a, b shape (2^n,)."""
    dim = a.shape[0]
    out = np.zeros(dim)
    for i in range(dim):
        if a[i] == 0.0:
            continue
        for j in range(dim):
            if b[j] == 0.0:
                continue
            k, s = _blade_product(i, j, sig)
            if s != 0.0:
                out[k] += a[i] * b[j] * s
    return out


def reverse(a: np.ndarray) -> np.ndarray:
    """Multivector reversal: ã with sign (-1)^(k(k-1)/2) on grade-k blades."""
    out = a.copy()
    for i in range(out.size):
        k = _basis_grade(i)
        if (k * (k - 1) // 2) % 2:
            out[i] *= -1.0
    return out


# ----- rotors -----

def _bivector_basis_index(i: int, j: int, n: int) -> int:
    return (1 << i) | (1 << j)


def rotor_from_bivector(B: np.ndarray, theta: float, sig: tuple[int, int, int]) -> np.ndarray:
    """R = exp(-θ B / 2) for a unit bivector B (rotation) or boost generator.
    Implemented via cos/sinh based on B² sign."""
    dim = B.shape[0]
    BB = geometric_product(B, B, sig)[0]  # scalar part
    R = np.zeros(dim)
    R[0] = 1.0
    if BB < 0:  # rotation in elliptic plane
        c = np.cos(theta / 2)
        s = -np.sin(theta / 2)
        R[0] = c
        R += s * B - (R - R)  # add s*B (component-wise)
        R = R + s * B
        R[0] = c  # ensure scalar precise
    elif BB > 0:  # boost in hyperbolic plane
        c = np.cosh(theta / 2)
        s = -np.sinh(theta / 2)
        R[0] = c
        R = R + s * B
        R[0] = c
    else:
        # null bivector — exponential is 1 + (-θ/2) B
        R = R + (-theta / 2) * B
    return R


def apply_rotor(R: np.ndarray, x: np.ndarray, sig: tuple[int, int, int]) -> np.ndarray:
    return geometric_product(geometric_product(R, x, sig), reverse(R), sig)


# ----- equivariant layer -----

class CliffordLayer:
    """y = R x R̃ with a learned rotor R."""
    def __init__(self, dim: int, sig: tuple[int, int, int], rng: np.random.Generator):
        self.dim = dim
        self.sig = sig
        # parametrize R as exp(-α B/2) for a random unit bivector B
        n = sig[0] + sig[1]
        i, j = rng.integers(0, n), rng.integers(0, n)
        while i == j:
            j = rng.integers(0, n)
        B = np.zeros(dim)
        B[_bivector_basis_index(int(i), int(j), n)] = 1.0
        alpha = float(rng.normal(0, 0.5))
        self.R = rotor_from_bivector(B, alpha, sig)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return apply_rotor(self.R, x, self.sig)


# ----- LIGO classifier (real-data Clifford identifier) -----

def _strain_to_features(strain: np.ndarray, n_features: int) -> np.ndarray:
    """Map a (B, 256) strain batch to (B, n_features) by averaging chunks of |x|.

    Using |strain| means the feature is sensitive to envelope shape —
    chirps concentrate energy into the middle samples; blips spread it.
    """
    B, T = strain.shape
    chunks = np.array_split(np.arange(T), n_features)
    feats = np.stack([np.abs(strain[:, c]).mean(axis=1) for c in chunks], axis=1)
    # standardize column-wise
    mu = feats.mean(axis=0, keepdims=True)
    sd = feats.std(axis=0, keepdims=True) + 1e-9
    return (feats - mu) / sd


def _build_rotor(angles: np.ndarray, sig: tuple[int, int, int]) -> np.ndarray:
    """Compose elementary plane-rotors R = R_{12} · R_{13} · R_{23}.

    Each angles[i] parameterizes one bivector plane. For Cl(3,0), all
    three bivectors square to -1 → all elementary rotors are elliptic.
    """
    n = sig[0] + sig[1]
    dim = 1 << n
    R = np.zeros(dim)
    R[0] = 1.0
    planes = [(0, 1), (0, 2), (1, 2)]
    for k, (i, j) in enumerate(planes[:len(angles)]):
        B = np.zeros(dim)
        B[_bivector_basis_index(i, j, n)] = 1.0
        R = geometric_product(rotor_from_bivector(B, float(angles[k]), sig), R, sig)
    # renormalize R so R R̃ = 1 (numerical drift)
    norm_sq = float(geometric_product(R, reverse(R), sig)[0])
    if norm_sq > 0:
        R = R / np.sqrt(norm_sq)
    return R


def _clifford_forward(
    feats: np.ndarray,
    angles: np.ndarray,
    head: np.ndarray,
    sig: tuple[int, int, int],
) -> np.ndarray:
    """Lift feats to grade-1 multivectors, rotate, project to scalar logits.

    feats: (B, n)  → x: (B, dim) with x[:, 1<<i] = feats[:, i]
    rotor: dim-vector
    head:  dim-vector (linear classifier on multivector components)
    out:   (B,) logits
    """
    n = sig[0] + sig[1]
    dim = 1 << n
    B = feats.shape[0]
    X = np.zeros((B, dim))
    for i in range(n):
        X[:, 1 << i] = feats[:, i]
    R = _build_rotor(angles, sig)
    Rt = reverse(R)
    Y = np.zeros_like(X)
    for b_i in range(B):
        Y[b_i] = geometric_product(geometric_product(R, X[b_i], sig), Rt, sig)
    return Y @ head


def _logistic(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def _train_clifford_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sig: tuple[int, int, int],
    n_features: int,
    epochs: int,
    lr: float,
    seed: int,
):
    """Joint optimization of (rotor angles, linear head) on logistic loss.

    Uses central-difference gradients on the rotor angles (3 params)
    and analytic gradient on the linear head (dim params). Very small
    parameter count → trivial to optimize.
    """
    rng = np.random.default_rng(seed)
    n = sig[0] + sig[1]
    dim = 1 << n

    angles = rng.normal(0, 0.2, size=3)
    head = rng.normal(0, 0.1, size=dim)

    history = []
    eps = 1e-3

    for ep in range(epochs):
        # forward
        logits = _clifford_forward(X_train, angles, head, sig)
        p = _logistic(logits)
        loss = -np.mean(y_train * np.log(p + 1e-12) + (1 - y_train) * np.log(1 - p + 1e-12))

        # gradient on head — analytic
        # d loss / d logits = (p - y) / B
        n_b = X_train.shape[0]
        d_logit = (p - y_train) / n_b

        # rebuild Y for the head gradient (Y @ head = logits)
        # Y = R x R̃ for current rotor
        R = _build_rotor(angles, sig)
        Rt = reverse(R)
        Y = np.zeros((n_b, dim))
        for b in range(n_b):
            xv = np.zeros(dim)
            for i in range(n):
                xv[1 << i] = X_train[b, i]
            Y[b] = geometric_product(geometric_product(R, xv, sig), Rt, sig)
        grad_head = Y.T @ d_logit

        # gradient on angles — finite differences
        grad_angles = np.zeros_like(angles)
        for k in range(len(angles)):
            ap = angles.copy(); ap[k] += eps
            am = angles.copy(); am[k] -= eps
            lp = _clifford_forward(X_train, ap, head, sig)
            lm = _clifford_forward(X_train, am, head, sig)
            pp = _logistic(lp); pm = _logistic(lm)
            loss_p = -np.mean(y_train * np.log(pp + 1e-12) + (1 - y_train) * np.log(1 - pp + 1e-12))
            loss_m = -np.mean(y_train * np.log(pm + 1e-12) + (1 - y_train) * np.log(1 - pm + 1e-12))
            grad_angles[k] = (loss_p - loss_m) / (2 * eps)

        # step
        head = head - lr * grad_head
        angles = angles - lr * grad_angles

        # accuracy
        acc = float(((p > 0.5).astype(int) == y_train).mean())
        history.append({"epoch": ep, "loss": float(loss), "accuracy": acc})

    return angles, head, history


@st.cache_data(ttl=600, show_spinner=False)
def _load_ligo_fixture():
    if not LIGO_FIXTURE.exists():
        return None
    npz = np.load(LIGO_FIXTURE, allow_pickle=False)
    out = {"signals": npz["signals"], "glitches": npz["glitches"]}
    out["source"] = str(npz["source"]) if "source" in npz.files else "unknown"
    out["n_real_events"] = int(npz["n_real_events"]) if "n_real_events" in npz.files else 0
    return out


def _render_ligo_classifier(embed: bool) -> None:
    fixture = _load_ligo_fixture()
    if fixture is None:
        st.warning(
            "**LIGO fixture missing.** Run "
            "`python scripts/fetch_ligo_real.py --output data/ligo_strain_sample.npz` "
            "(real GWOSC O3) or "
            "`python scripts/generate_ligo_fixture.py --output data/ligo_strain_sample.npz` "
            "(synthetic)."
        )
        return

    src = fixture.get("source", "unknown")
    n_real = fixture.get("n_real_events", 0)
    info = st.columns(3)
    if src == "gwosc-real-o3":
        info[0].metric("Data source", "GWOSC O3 (real)")
        info[1].metric("Real events", n_real)
    elif src == "synthetic-pycbc":
        info[0].metric("Data source", "PyCBC (synthetic)")
        info[1].metric("Real events", "0")
    else:
        info[0].metric("Data source", src)
        info[1].metric("Real events", n_real)
    info[2].metric("Algebra", "Cl(3,0)")

    with st.sidebar:
        st.markdown("### Classifier")
        n_each = st.slider(
            "Examples per class",
            50, min(500, len(fixture["signals"]), len(fixture["glitches"])),
            200, step=50,
        )
        epochs = st.slider("Epochs", 5, 80, 30)
        lr = st.number_input("Learning rate", 0.001, 1.0, 0.1, 0.01, format="%.3f")
        seed = int(st.number_input("Seed", value=7, step=1))
        run = st.button("Train Clifford classifier", type="primary", use_container_width=True)

    sig = _signature("Cl(3,0)")
    n_basis = sig[0] + sig[1]

    rng = np.random.default_rng(seed)
    sigs = fixture["signals"]
    glis = fixture["glitches"]
    s_idx = rng.choice(len(sigs), size=n_each, replace=False)
    g_idx = rng.choice(len(glis), size=n_each, replace=False)
    raw = np.vstack([sigs[s_idx], glis[g_idx]])
    y = np.concatenate([np.ones(n_each, dtype=np.float64), np.zeros(n_each, dtype=np.float64)])
    perm = rng.permutation(len(raw))
    raw, y = raw[perm], y[perm]

    # 80/20 split
    cut = int(0.8 * len(raw))
    feats = _strain_to_features(raw, n_basis)
    X_train, X_test = feats[:cut], feats[cut:]
    y_train, y_test = y[:cut], y[cut:]

    if run:
        with st.spinner("Training Clifford-rotor classifier on LIGO strain windows…"):
            try:
                angles, head, history = _train_clifford_classifier(
                    X_train, y_train, sig, n_basis, int(epochs), float(lr), seed,
                )
            except Exception as e:
                st.error(f"Training failed: {e}")
                return
        test_logits = _clifford_forward(X_test, angles, head, sig)
        test_p = _logistic(test_logits)
        st.session_state.clf_results = dict(
            angles=angles, head=head, history=history,
            test_p=test_p, y_test=y_test,
            train_p=_logistic(_clifford_forward(X_train, angles, head, sig)),
            y_train=y_train,
        )

    if "clf_results" not in st.session_state:
        st.info(
            "Configure parameters in the sidebar, then click **Train Clifford "
            "classifier**. The model has 3 rotor angles + 8 linear-head weights."
        )
        return

    res = st.session_state.clf_results
    history = res["history"]

    # learning curves
    losses = [h["loss"] for h in history]
    accs = [h["accuracy"] for h in history]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=losses, name="loss", line=dict(color=PALETTE["mint_rim"])))
    fig.add_trace(go.Scatter(y=accs, name="train acc", line=dict(color=PALETTE["violet_glow"]), yaxis="y2"))
    fig.update_layout(
        title="Training", xaxis_title="epoch",
        yaxis=dict(title="loss"),
        yaxis2=dict(title="accuracy", overlaying="y", side="right", range=[0, 1]),
        height=320,
    )
    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    # test metrics
    test_pred = (res["test_p"] > 0.5).astype(int)
    y_test_int = res["y_test"].astype(int)
    tp = int(((test_pred == 1) & (y_test_int == 1)).sum())
    tn = int(((test_pred == 0) & (y_test_int == 0)).sum())
    fp = int(((test_pred == 1) & (y_test_int == 0)).sum())
    fn = int(((test_pred == 0) & (y_test_int == 1)).sum())
    test_acc = (tp + tn) / max(1, tp + tn + fp + fn)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Test accuracy", f"{test_acc * 100:.1f}%")
    metric_cols[1].metric("True positive", tp)
    metric_cols[2].metric("False positive", fp)
    metric_cols[3].metric("False negative", fn)

    col1, col2 = st.columns(2)
    with col1:
        cm = go.Figure(data=go.Heatmap(
            z=[[tn, fp], [fn, tp]],
            x=["Pred glitch", "Pred chirp"],
            y=["Actual glitch", "Actual chirp"],
            text=[[tn, fp], [fn, tp]],
            texttemplate="%{text}",
            colorscale=[[0, PALETTE["bg_panel"]], [1, PALETTE["violet_glow"]]],
            showscale=False,
        ))
        cm.update_layout(title="Test confusion matrix", height=320)
        apply_plotly_theme(cm)
        st.plotly_chart(cm, use_container_width=True)

    with col2:
        scores = res["test_p"]
        order = np.argsort(-scores)
        ys = y_test_int[order]
        tpr = np.cumsum(ys) / max(1, ys.sum())
        fpr = np.cumsum(1 - ys) / max(1, (1 - ys).sum())
        auc = float(np.trapz(tpr, fpr))
        roc = go.Figure()
        roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {auc:.3f}",
                                 line=dict(color=PALETTE["violet_glow"])))
        roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line=dict(color=PALETTE["violet_deep"], dash="dash"),
                                 showlegend=False))
        roc.update_layout(title="Test ROC", xaxis_title="FPR", yaxis_title="TPR", height=320)
        apply_plotly_theme(roc)
        st.plotly_chart(roc, use_container_width=True)

    # learned rotor / head summary
    st.markdown("**Learned parameters**")
    st.dataframe(
        pd.DataFrame({
            "rotor angle (rad)": [f"{a:.3f}" for a in res["angles"]] + [""] * 5,
            "head weight (per blade)": [f"{w:.3f}" for w in res["head"]],
        }),
        use_container_width=True, hide_index=True,
    )

    if not embed:
        with st.expander("Show classifier code"):
            st.code(
                inspect.getsource(_strain_to_features) + "\n\n"
                + inspect.getsource(_build_rotor) + "\n\n"
                + inspect.getsource(_train_clifford_classifier),
                language="python",
            )


# ----- UI -----

def render(embed: bool = False) -> None:
    if not embed:
        st.caption(
            "Two demos in one: an equivariance check on Clifford layers, "
            "and a small Clifford-rotor classifier trained on real LIGO O3 strain."
        )

    mode_default = "LIGO classifier" if LIGO_FIXTURE.exists() else "Equivariance check"
    mode: Mode = st.radio(
        "Mode", ["Equivariance check", "LIGO classifier"],
        index=["Equivariance check", "LIGO classifier"].index(mode_default),
        horizontal=True,
    )

    if mode == "LIGO classifier":
        _render_ligo_classifier(embed=embed)
        return

    with st.sidebar:
        st.markdown("### Parameters")
        algebra: Algebra = st.radio("Algebra", ["Cl(3,0)", "Cl(3,1)"], horizontal=True)
        sig = _signature(algebra)
        dim = _basis_dim(*sig)

        st.markdown("**Input vector x** (grade 1)")
        n_basis = sig[0] + sig[1]
        names = ["e1", "e2", "e3", "e4"][:n_basis]
        if algebra == "Cl(3,1)":
            names = ["e1", "e2", "e3", "e0"]  # e0 timelike
        cols = st.columns(n_basis)
        x_vec = []
        for i, name in enumerate(names):
            with cols[i]:
                x_vec.append(st.number_input(name, value=1.0 if i == 0 else 0.0, step=0.5))

        transform: Transform = st.radio("Transform", ["Rotation", "Boost", "Reflection"], horizontal=True)
        if transform == "Boost" and algebra != "Cl(3,1)":
            st.info("Boosts require Cl(3,1) — falling back to rotation.")
            transform = "Rotation"
        param = st.slider(
            "θ" if transform == "Rotation" else "η (rapidity)" if transform == "Boost" else "axis index",
            -3.14 if transform != "Reflection" else 0.0,
            3.14 if transform != "Reflection" else float(n_basis - 1),
            0.4,
        )
        seed = st.number_input("Layer seed", value=3, step=1)

    # build x as multivector (grade-1)
    x = np.zeros(dim)
    for i, val in enumerate(x_vec):
        x[1 << i] = val

    # build transformation g
    g_R = np.zeros(dim)
    g_R[0] = 1.0
    if transform == "Rotation":
        B = np.zeros(dim)
        B[_bivector_basis_index(0, 1, n_basis)] = 1.0
        g_R = rotor_from_bivector(B, float(param), sig)
    elif transform == "Boost":
        B = np.zeros(dim)
        B[_bivector_basis_index(0, 3, n_basis)] = 1.0  # e1 ∧ e0
        g_R = rotor_from_bivector(B, float(param), sig)
    else:  # Reflection: g(x) = -e_k x e_k
        k = int(param) % n_basis
        e = np.zeros(dim)
        e[1 << k] = 1.0
        # reflection: x → -e x e
        gx = -geometric_product(geometric_product(e, x, sig), e, sig)
        # implement as a rotor-equivalent: skip — handled below

    layer = CliffordLayer(dim, sig, np.random.default_rng(int(seed)))

    if transform == "Reflection":
        gx = -geometric_product(geometric_product(e, x, sig), e, sig)
        f_gx = layer.forward(gx)
        # for reflection, the equivariance check uses the same operation
        f_x = layer.forward(x)
        g_f_x = -geometric_product(geometric_product(e, f_x, sig), e, sig)
    else:
        gx = apply_rotor(g_R, x, sig)
        f_gx = layer.forward(gx)
        f_x = layer.forward(x)
        g_f_x = apply_rotor(g_R, f_x, sig)

    err = float(np.linalg.norm(f_gx - g_f_x))

    # grade decomposition
    grade_table = pd.DataFrame({
        "grade": list(range(n_basis + 1)),
        "‖x‖": [float(np.linalg.norm([x[i] for i in range(dim) if _basis_grade(i) == g])) for g in range(n_basis + 1)],
        "‖f(x)‖": [float(np.linalg.norm([f_x[i] for i in range(dim) if _basis_grade(i) == g])) for g in range(n_basis + 1)],
        "‖f(g·x) − g·f(x)‖": [
            float(np.linalg.norm([(f_gx - g_f_x)[i] for i in range(dim) if _basis_grade(i) == g]))
            for g in range(n_basis + 1)
        ],
    })

    cols = st.columns(3)
    cols[0].metric("Equivariance error", f"{err:.2e}")
    cols[1].metric("Algebra", algebra)
    cols[2].metric("Multivector dim", dim)

    st.markdown("**Grade decomposition**")
    st.dataframe(grade_table, use_container_width=True, hide_index=True)

    # 3D vector visualization
    if n_basis >= 3:
        x_vec3 = [float(x[1 << i]) for i in range(3)]
        f_vec3 = [float(f_x[1 << i]) for i in range(3)]
        gx_vec3 = [float(gx[1 << i]) for i in range(3)]
        fig = go.Figure()
        for label, v, color in [("x", x_vec3, PALETTE["phosphor"]),
                                 ("g·x", gx_vec3, PALETTE["amber"]),
                                 ("f(x)", f_vec3, PALETTE["bone_dim"])]:
            fig.add_trace(go.Scatter3d(
                x=[0, v[0]], y=[0, v[1]], z=[0, v[2]],
                mode="lines+markers+text",
                text=["", label],
                line=dict(color=color, width=4),
                marker=dict(size=4, color=color),
                name=label,
            ))
        fig.update_layout(scene=dict(
            xaxis=dict(backgroundcolor=PALETTE["ink"], gridcolor=PALETTE["rule"]),
            yaxis=dict(backgroundcolor=PALETTE["ink"], gridcolor=PALETTE["rule"]),
            zaxis=dict(backgroundcolor=PALETTE["ink"], gridcolor=PALETTE["rule"]),
        ), height=400)
        apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    if not embed:
        with st.expander("Show code"):
            st.code(
                inspect.getsource(_blade_product) + "\n\n"
                + inspect.getsource(geometric_product) + "\n\n"
                + inspect.getsource(CliffordLayer),
                language="python",
            )
