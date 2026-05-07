"""Theme injection — the CSS that aligns Streamlit's chrome with the
Astro site's cybergothic palette, plus an embed-mode override that strips
the chrome entirely for iframe embedding."""

from __future__ import annotations

import streamlit as st

# Token mirror of Astro's src/styles/global.css :root values.
# The legacy keys (ink, bone, phosphor, rust, amber, rule) are kept so
# every existing demo keeps rendering — they now resolve to cybergothic
# values rather than the old ink-and-phosphor palette.
PALETTE = {
    # Backgrounds
    "ink":           "#0A0014",  # bg-deep
    "ink_2":         "#160028",  # bg-panel
    "ink_3":         "#240046",  # bg-elevated
    "bg_deep":       "#0A0014",
    "bg_panel":      "#160028",
    "bg_elevated":   "#240046",

    # Brand violets
    "violet_deep":   "#3C096C",
    "violet_mid":    "#6B2FBF",
    "violet_bright": "#9D4EDD",
    "violet_glow":   "#C77DFF",
    "violet_pale":   "#E0AAFF",

    # Iridescent accents
    "magenta_glow":  "#FF77E6",
    "pink_soft":     "#FF8FD8",
    "mint_rim":      "#A7F3D0",
    "mint_bright":   "#86EFAC",

    # Text
    "ink_primary":   "#F2E6FF",
    "ink_secondary": "#C77DFF",
    "ink_muted":     "#9D4EDD",
    "ink_dim":       "#6B2FBF",

    # Legacy aliases (re-mapped onto cybergothic)
    "bone":          "#F2E6FF",
    "bone_dim":      "#C77DFF",
    "bone_faint":    "#9D4EDD",
    "phosphor":      "#C77DFF",
    "phosphor_dark": "#6B2FBF",
    "rust":          "#FF77E6",
    "amber":         "#A7F3D0",
    "rule":          "#3C096C",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor=PALETTE["bg_deep"],
    plot_bgcolor=PALETTE["bg_panel"],
    font=dict(family="JetBrains Mono, monospace", color=PALETTE["ink_primary"]),
    colorway=[
        PALETTE["violet_glow"],
        PALETTE["mint_rim"],
        PALETTE["magenta_glow"],
        PALETTE["violet_pale"],
        PALETTE["violet_bright"],
    ],
    xaxis=dict(gridcolor="rgba(199,125,255,0.18)", zerolinecolor=PALETTE["violet_deep"]),
    yaxis=dict(gridcolor="rgba(199,125,255,0.18)", zerolinecolor=PALETTE["violet_deep"]),
    legend=dict(font=dict(family="JetBrains Mono, monospace")),
    margin=dict(l=40, r=20, t=40, b=40),
)


_BASE_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=UnifrakturCook:wght@700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

  /* violet-tinted scrollbars */
  ::-webkit-scrollbar { width: 8px; height: 8px; }
  ::-webkit-scrollbar-track { background: #160028; }
  ::-webkit-scrollbar-thumb { background: #3C096C; }
  ::-webkit-scrollbar-thumb:hover { background: #C77DFF; }

  /* selection */
  ::selection { background: #C77DFF; color: #0A0014; }

  /* body baseline */
  .stApp {
    background: #0A0014 !important;
    color: #F2E6FF !important;
    font-family: "Inter", system-ui, sans-serif;
  }

  /* H1/H2 in blackletter — display font for the lab */
  .stApp h1, .stApp h2 {
    font-family: "UnifrakturCook", serif !important;
    font-weight: 700 !important;
    color: #F2E6FF !important;
    letter-spacing: 0 !important;
  }
  .stApp h3, .stApp h4 {
    font-family: "Inter", system-ui, sans-serif !important;
    color: #C77DFF !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.95rem !important;
  }

  /* code blocks */
  pre, code {
    font-family: "JetBrains Mono", ui-monospace, monospace !important;
    color: #A7F3D0 !important;
  }
  pre { background: #160028 !important; border: 1px solid rgba(199,125,255,0.18) !important; }

  /* metrics — violet labels, pale-violet numerals on deep bg */
  div[data-testid="stMetricValue"] {
    font-family: "JetBrains Mono", monospace !important;
    color: #F2E6FF !important;
  }
  div[data-testid="stMetricLabel"] {
    font-family: "JetBrains Mono", monospace !important;
    color: #9D4EDD !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.72rem !important;
  }

  /* primary button = violet-bright with glow */
  button[kind="primary"] {
    background: #9D4EDD !important;
    color: #0A0014 !important;
    border-color: #9D4EDD !important;
    font-family: "JetBrains Mono", monospace !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    box-shadow: 0 0 24px -8px #C77DFF;
  }
  button[kind="primary"]:hover {
    background: #FF77E6 !important;
    color: #0A0014 !important;
    border-color: #FF77E6 !important;
    box-shadow: 0 0 28px -6px #FF77E6;
  }

  /* secondary buttons */
  button[kind="secondary"] {
    background: transparent !important;
    color: #C77DFF !important;
    border: 1px solid rgba(199,125,255,0.35) !important;
    font-family: "JetBrains Mono", monospace !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  button[kind="secondary"]:hover {
    border-color: #C77DFF !important;
    color: #FF77E6 !important;
  }

  /* tab labels */
  .stTabs [data-baseweb="tab-list"] button {
    font-family: "JetBrains Mono", monospace !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-size: 0.78rem !important;
    color: #C77DFF !important;
  }
  .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    color: #F2E6FF !important;
    border-bottom-color: #C77DFF !important;
  }

  /* sidebar */
  section[data-testid="stSidebar"] {
    background: #160028 !important;
    border-right: 1px solid rgba(199,125,255,0.15) !important;
  }

  /* alerts */
  div[data-testid="stAlert"] {
    background: #160028 !important;
    border: 1px solid rgba(199,125,255,0.25) !important;
    color: #F2E6FF !important;
  }

  /* dataframe */
  div[data-testid="stDataFrame"] {
    background: #160028 !important;
  }

  /* chat messages */
  div[data-testid="stChatMessage"] {
    background: #160028 !important;
    border: 1px solid rgba(199,125,255,0.15) !important;
    border-radius: 0 !important;
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
