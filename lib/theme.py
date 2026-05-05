"""Theme injection — the CSS that aligns Streamlit's chrome with the
Astro site's tokens, plus an embed-mode override that strips the chrome
entirely for iframe embedding."""

from __future__ import annotations

import streamlit as st

# Token mirror of Astro's src/styles/global.css :root values.
PALETTE = {
    "ink": "#0a0d0c",
    "ink_2": "#111614",
    "ink_3": "#1a201d",
    "bone": "#e8e4d8",
    "bone_dim": "#b8b3a4",
    "bone_faint": "#6a685e",
    "phosphor": "#7eff9f",
    "phosphor_dark": "#4cb86b",
    "rust": "#c9543a",
    "amber": "#e8b04c",
    "rule": "#2a2f2c",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor=PALETTE["ink"],
    plot_bgcolor=PALETTE["ink_2"],
    font=dict(family="JetBrains Mono, monospace", color=PALETTE["bone"]),
    colorway=[
        PALETTE["phosphor"],
        PALETTE["amber"],
        PALETTE["bone_dim"],
        PALETTE["rust"],
        PALETTE["phosphor_dark"],
    ],
    xaxis=dict(gridcolor=PALETTE["rule"], zerolinecolor=PALETTE["rule"]),
    yaxis=dict(gridcolor=PALETTE["rule"], zerolinecolor=PALETTE["rule"]),
    legend=dict(font=dict(family="JetBrains Mono, monospace")),
    margin=dict(l=40, r=20, t=40, b=40),
)


_BASE_CSS = """
<style>
  /* phosphor-tinted scrollbars */
  ::-webkit-scrollbar { width: 8px; height: 8px; }
  ::-webkit-scrollbar-track { background: #111614; }
  ::-webkit-scrollbar-thumb { background: #2a2f2c; }
  ::-webkit-scrollbar-thumb:hover { background: #7eff9f; }

  /* selection */
  ::selection { background: #7eff9f; color: #0a0d0c; }

  /* body baseline */
  .stApp { background: #0a0d0c !important; color: #e8e4d8; }

  /* code blocks */
  pre, code { font-family: "JetBrains Mono", ui-monospace, monospace !important; }

  /* metrics — phosphor numerals on ink */
  div[data-testid="stMetricValue"] {
    font-family: "JetBrains Mono", monospace !important;
    color: #e8e4d8 !important;
  }
  div[data-testid="stMetricLabel"] {
    font-family: "JetBrains Mono", monospace !important;
    color: #6a685e !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.72rem !important;
  }

  /* primary button = phosphor */
  button[kind="primary"] {
    background: #7eff9f !important;
    color: #0a0d0c !important;
    border-color: #7eff9f !important;
    font-family: "JetBrains Mono", monospace !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  button[kind="primary"]:hover {
    background: transparent !important;
    color: #7eff9f !important;
  }

  /* tab labels */
  .stTabs [data-baseweb="tab-list"] button {
    font-family: "JetBrains Mono", monospace !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-size: 0.78rem !important;
  }
</style>
"""

_EMBED_CSS = """
<style>
  /* hide all streamlit chrome when embedded in the Astro iframe */
  #MainMenu, footer, header[data-testid="stHeader"] { display: none !important; }
  div[data-testid="stToolbar"] { display: none !important; }
  div[data-testid="stDecoration"] { display: none !important; }
  .block-container {
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
    padding-left: 1.25rem !important;
    padding-right: 1.25rem !important;
    max-width: 100% !important;
  }
</style>
"""


def inject_css(embed: bool = False) -> None:
    st.markdown(_BASE_CSS, unsafe_allow_html=True)
    if embed:
        st.markdown(_EMBED_CSS, unsafe_allow_html=True)


def apply_plotly_theme(fig) -> None:
    """Mutate a Plotly figure in-place to match the lab palette."""
    fig.update_layout(**PLOTLY_LAYOUT)
