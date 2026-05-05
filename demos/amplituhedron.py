"""Demo 03 — Amplituhedron explorer.

Cyclic-polytope realization of the tree-level k=1 amplituhedron in P^{n-2}
for N=4 SYM. Vertices live on the moment curve t ↦ (t, t², …, t^{n-2});
all (n−2)×(n−2) ordered minors are positive — that's the cyclic positivity
that makes this the right toy model for amplituhedron geometry.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.spatial import ConvexHull

from lib.theme import PALETTE, apply_plotly_theme


def _moment_curve_vertices(n: int) -> np.ndarray:
    """n vertices in (n-2)-dim space, on the moment curve t = 1..n."""
    d = n - 2
    t = np.arange(1, n + 1, dtype=float)
    return np.stack([t ** (k + 1) for k in range(d)], axis=1)


def _cyclic_minors_positive(V: np.ndarray) -> tuple[bool, list[float]]:
    """Check all ordered (d × d) minors of [1 | V] are positive."""
    n, d = V.shape
    aug = np.concatenate([np.ones((n, 1)), V], axis=1)
    minors = []
    from itertools import combinations
    for idx in combinations(range(n), d + 1):
        sub = aug[list(idx)]
        m = float(np.linalg.det(sub))
        minors.append(m)
    all_pos = all(m > 0 for m in minors)
    return all_pos, minors


@st.cache_data(ttl=600, show_spinner=False)
def _construct_polytope(n: int):
    V = _moment_curve_vertices(n)
    if V.shape[1] >= 2:
        try:
            hull = ConvexHull(V)
            volume = float(hull.volume)
            simplices = hull.simplices.tolist()
        except Exception:
            volume = 0.0
            simplices = []
    else:
        volume = 0.0
        simplices = []
    all_pos, minors = _cyclic_minors_positive(V)
    return V, simplices, volume, all_pos, minors


def _sample_interior(V: np.ndarray, n_pts: int, rng: np.random.Generator) -> np.ndarray:
    """Sample by mixing convex weights — Dirichlet over vertices."""
    weights = rng.dirichlet(np.ones(V.shape[0]), size=n_pts)
    return weights @ V


def render(embed: bool = False) -> None:
    if not embed:
        st.caption(
            "Cyclic-polytope realization of the tree-level k=1 amplituhedron. "
            "Cyclic positivity is the key property — every ordered minor is positive."
        )

    with st.sidebar:
        st.markdown("### Parameters")
        n = st.slider("Number of particles n", 4, 8, 6)
        view = st.radio("View", ["3D wireframe", "2D projection", "Vertex coordinates"])
        n_pts = st.slider("Sample interior points", 0, 500, 120)
        seed = st.number_input("Seed", value=11, step=1)
        st.button("Resample", type="primary")

    V, simplices, volume, all_pos, minors = _construct_polytope(n)
    rng = np.random.default_rng(int(seed))
    pts = _sample_interior(V, n_pts, rng) if n_pts > 0 else np.zeros((0, V.shape[1]))

    cols = st.columns(3)
    cols[0].metric("Vertices", n)
    cols[1].metric("Dimension", n - 2)
    cols[2].metric("Volume", f"{volume:.4g}")

    fig = go.Figure()
    if view == "3D wireframe" and V.shape[1] >= 3:
        proj = V[:, :3]
        fig.add_trace(go.Scatter3d(
            x=proj[:, 0], y=proj[:, 1], z=proj[:, 2],
            mode="markers+lines+text",
            text=[str(i + 1) for i in range(n)],
            marker=dict(size=6, color=PALETTE["phosphor"]),
            line=dict(color=PALETTE["phosphor_dark"]),
            name="vertices",
        ))
        if pts.shape[0] > 0 and pts.shape[1] >= 3:
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers",
                marker=dict(size=2, color=PALETTE["amber"], opacity=0.5),
                name="interior",
            ))
        fig.update_layout(scene=dict(
            xaxis=dict(backgroundcolor=PALETTE["ink"], gridcolor=PALETTE["rule"], color=PALETTE["bone"]),
            yaxis=dict(backgroundcolor=PALETTE["ink"], gridcolor=PALETTE["rule"], color=PALETTE["bone"]),
            zaxis=dict(backgroundcolor=PALETTE["ink"], gridcolor=PALETTE["rule"], color=PALETTE["bone"]),
        ), height=480)
    elif view == "2D projection":
        proj = V[:, :2]
        # close polygon
        closed = np.vstack([proj, proj[:1]])
        fig.add_trace(go.Scatter(
            x=closed[:, 0], y=closed[:, 1],
            mode="markers+lines+text",
            text=[str(i + 1) for i in range(n)] + [""],
            marker=dict(size=8, color=PALETTE["phosphor"]),
            line=dict(color=PALETTE["phosphor_dark"]),
            name="vertices",
        ))
        if pts.shape[0] > 0:
            fig.add_trace(go.Scatter(
                x=pts[:, 0], y=pts[:, 1],
                mode="markers",
                marker=dict(size=4, color=PALETTE["amber"], opacity=0.6),
                name="interior",
            ))
        fig.update_layout(height=480, xaxis_title="t", yaxis_title="t²")
    else:
        df = pd.DataFrame(V, columns=[f"t^{k+1}" for k in range(V.shape[1])])
        df.insert(0, "vertex", [f"v_{i + 1}" for i in range(n)])
        st.dataframe(df, use_container_width=True)

    if view != "Vertex coordinates":
        apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # cyclic positivity readout
    cls = "phosphor" if all_pos else "rust"
    color = PALETTE["phosphor"] if all_pos else PALETTE["rust"]
    label = "ALL ORDERED MINORS > 0" if all_pos else "VIOLATION DETECTED"
    st.markdown(
        f"<div style='font-family: monospace; font-size: 0.85rem; letter-spacing: 0.16em; "
        f"color: {color}; margin-top: 1rem;'>● {label}</div>",
        unsafe_allow_html=True,
    )
    with st.expander(f"Minor table ({len(minors)} ordered (n−1)×(n−1) minors)"):
        st.dataframe(pd.DataFrame({"minor": minors}), use_container_width=True)

    if not embed:
        with st.expander("Show code"):
            st.code(
                inspect.getsource(_moment_curve_vertices) + "\n\n"
                + inspect.getsource(_cyclic_minors_positive),
                language="python",
            )
