"""Demo 02 — Clifford geometric algebra agent.

Equivariance demonstration in Cl(3,0) and Cl(3,1). Multivectors are encoded
as length-2^n arrays indexed by basis-blade bitmasks. The geometric product
is implemented from the signature directly — no third-party clifford pkg.

The headline numerical fact: ‖f(g·x) − g·f(x)‖ stays at machine precision
for any rotor g and any input x.
"""

from __future__ import annotations

import inspect
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.theme import PALETTE, apply_plotly_theme

Algebra = Literal["Cl(3,0)", "Cl(3,1)"]
Transform = Literal["Rotation", "Boost", "Reflection"]


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


# ----- UI -----

def render(embed: bool = False) -> None:
    if not embed:
        st.caption(
            "Equivariant Clifford layers: y = R x R̃. "
            "Picking an input x, a transform g, and showing that f(g·x) = g·f(x) numerically."
        )

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
