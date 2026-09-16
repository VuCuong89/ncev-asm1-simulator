from __future__ import annotations

import html
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from asm1_engine import ASM1Parameters, ProcessConfig, STATE_NAMES, simulate
from chemistry import (
    CarbonSettings,
    NaOHSettings,
    PhosphorusSettings,
    add_external_cod,
    calculate_naoh,
    optimize_external_carbon,
    phosphorus_chemical,
    solve_was_for_target_srt,
    total_sludge,
)
from fractionation import FractionationAssumptions, InfluentInput, InitialConditions, fractionate
from i18n import LANG_OPTIONS, tr

APP_VERSION = "Cloud v3.1"
NCEV_BLUE = "#075DA8"
NCEV_CYAN = "#19A9C7"
NCEV_NAVY = "#0A365D"
NCEV_TEXT = "#17324A"
NCEV_BORDER = "#AFC7D6"
NCEV_SOFT = "#F6FBFE"
BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "ncev_logo.png"

st.set_page_config(
    page_title="NCEV AO ASM1 Simulator",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# v3.0 deliberately uses one compact CSS layer. The calculation engine is unchanged.
st.markdown(
    f"""
<style>
:root {{ color-scheme: light !important; }}
html, body, .stApp, [data-testid="stAppViewContainer"] {{
    background: #FFFFFF !important;
    color: {NCEV_TEXT} !important;
}}
[data-testid="stHeader"] {{ background: rgba(255,255,255,.98) !important; z-index: 1000 !important; }}
[data-testid="stSidebar"], [data-testid="stSidebar"] > div {{
    background: {NCEV_SOFT} !important;
    color: {NCEV_TEXT} !important;
}}
.block-container {{
    max-width: 1500px;
    padding-top: 4.75rem !important;
    padding-bottom: 3rem;
}}

/* Typography */
.stMarkdown, .stMarkdown p, .stMarkdown li,
[data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label {{
    color: {NCEV_TEXT} !important;
}}
h1, h2, h3, h4, h5, h6 {{ color: {NCEV_NAVY} !important; }}

/* Number input: strip BaseWeb/native inner borders, draw ONE uniform outline on the outer control. */
[data-testid="stNumberInput"] div[data-baseweb="input"] {{
    background: #FFFFFF !important;
    border: 0 !important;
    border-radius: 9px !important;
    box-shadow: inset 0 0 0 1px {NCEV_BORDER} !important;
    outline: none !important;
    overflow: hidden !important;
}}
[data-testid="stNumberInput"] div[data-baseweb="input"] *,
[data-testid="stNumberInput"] div[data-baseweb="base-input"],
[data-testid="stNumberInput"] div[data-baseweb="base-input"] *,
[data-testid="stNumberInput"] input {{
    box-shadow: none !important;
    outline: none !important;
}}
[data-testid="stNumberInput"] div[data-baseweb="base-input"],
[data-testid="stNumberInput"] input {{
    background: #FFFFFF !important;
    color: {NCEV_TEXT} !important;
    -webkit-text-fill-color: {NCEV_TEXT} !important;
    border: 0 !important;
}}
[data-testid="stNumberInput"] div[data-baseweb="input"]::before,
[data-testid="stNumberInput"] div[data-baseweb="input"]::after,
[data-testid="stNumberInput"] div[data-baseweb="base-input"]::before,
[data-testid="stNumberInput"] div[data-baseweb="base-input"]::after {{
    border: 0 !important;
    box-shadow: none !important;
}}
[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within {{
    border: 0 !important;
    box-shadow: inset 0 0 0 1px {NCEV_BLUE} !important;
}}
[data-testid="stNumberInput"] button {{
    background: #EAF5FB !important;
    color: {NCEV_BLUE} !important;
    border: 0 !important;
    border-left: 1px solid #C9DCE8 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}}
[data-testid="stNumberInput"] button:hover {{ background: #DCEFF9 !important; }}
[data-testid="stNumberInput"] button svg {{
    color: {NCEV_BLUE} !important;
    fill: {NCEV_BLUE} !important;
    stroke: {NCEV_BLUE} !important;
}}

/* Select boxes: use broad BaseWeb selectors so language + chemical selects stay light. */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
div[data-baseweb="select"] > div {{
    background: #FFFFFF !important;
    color: {NCEV_TEXT} !important;
    border: 0 !important;
    border-radius: 9px !important;
    box-shadow: inset 0 0 0 1px {NCEV_BORDER} !important;
    outline: none !important;
}}
[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within {{
    border: 0 !important;
    box-shadow: inset 0 0 0 1px {NCEV_BLUE} !important;
}}
[data-testid="stSelectbox"] div[data-baseweb="select"] *,
div[data-baseweb="select"] * {{
    color: {NCEV_TEXT} !important;
    -webkit-text-fill-color: {NCEV_TEXT} !important;
}}
[data-testid="stSelectbox"] svg,
div[data-baseweb="select"] svg {{
    color: {NCEV_NAVY} !important;
    fill: {NCEV_NAVY} !important;
    stroke: {NCEV_NAVY} !important;
}}
div[data-baseweb="popover"], div[data-baseweb="popover"] > div,
[role="listbox"] {{ background: #FFFFFF !important; color: {NCEV_TEXT} !important; }}
[role="option"] {{ background: #FFFFFF !important; color: {NCEV_TEXT} !important; }}
[role="option"] * {{ color: {NCEV_TEXT} !important; }}
[role="option"]:hover {{ background: #EAF5FB !important; }}

/* Radios / navigation */
[data-testid="stRadio"] label, [data-testid="stRadio"] label p {{ color: {NCEV_TEXT} !important; }}
[data-testid="stRadio"] [role="radiogroup"] {{ gap: .8rem; }}

/* Buttons */
.stButton > button {{
    border-radius: 9px !important;
    font-weight: 650 !important;
    min-height: 2.7rem;
}}
.stButton > button[kind="primary"] {{
    background: {NCEV_BLUE} !important;
    color: #FFFFFF !important;
    border: 1px solid {NCEV_BLUE} !important;
}}
.stButton > button[kind="primary"] * {{ color: #FFFFFF !important; }}
.stButton > button[kind="secondary"] {{
    background: #FFFFFF !important;
    color: {NCEV_NAVY} !important;
    border: 1px solid {NCEV_BORDER} !important;
}}
.stButton > button[kind="secondary"] * {{ color: {NCEV_NAVY} !important; }}
.stDownloadButton > button {{
    background: #EAF5FB !important;
    color: {NCEV_NAVY} !important;
    border: 1px solid #BFD6E4 !important;
    border-radius: 9px !important;
    font-weight: 650 !important;
}}
.stDownloadButton > button * {{ color: {NCEV_NAVY} !important; }}

/* Expanders / alerts: force the collapsed header itself to light background. */
[data-testid="stExpander"], details[data-testid="stExpander"] {{
    background: #FFFFFF !important;
    border: 1px solid #CFE0E9 !important;
    border-radius: 9px !important;
    overflow: hidden !important;
}}
[data-testid="stExpander"] summary,
details[data-testid="stExpander"] > summary {{
    background: #FFFFFF !important;
    color: {NCEV_TEXT} !important;
    border: 0 !important;
}}
[data-testid="stExpander"] summary *,
details[data-testid="stExpander"] > summary * {{ color: {NCEV_TEXT} !important; }}
[data-testid="stAlert"] {{ color: {NCEV_TEXT} !important; }}

/* Header */
.ncev-title {{
    color: {NCEV_NAVY};
    font-size: 2rem;
    font-weight: 800;
    line-height: 1.15;
    margin-top: .1rem;
}}
.ncev-sub {{ color: #536B7C; font-size: .98rem; margin-top: .25rem; }}
.ncev-rule {{ height: 3px; background: {NCEV_CYAN}; border-radius: 3px; margin: .65rem 0 1rem; }}
.section-title {{
    color: {NCEV_NAVY};
    font-size: 1.12rem;
    font-weight: 800;
    margin: .8rem 0 .55rem;
}}
.group-title {{ color: {NCEV_NAVY}; font-size: 1rem; font-weight: 750; margin: .2rem 0 .45rem; }}

/* KPI cards */
.kpi-card {{
    background: #FFFFFF;
    border: 1px solid #CFE0EA;
    border-radius: 11px;
    padding: .8rem 1rem .85rem;
    min-height: 112px;
    overflow: visible;
}}
.kpi-label {{ font-size: .9rem; color: #36566D; margin-bottom: .4rem; }}
.kpi-value {{
    font-size: 1.9rem;
    font-weight: 650;
    line-height: 1.08;
    color: {NCEV_NAVY};
    white-space: nowrap;
}}
.kpi-unit {{ font-size: .8rem; color: #617C8E; margin-left: .25rem; white-space: nowrap; }}
.kpi-note {{ font-size: .76rem; color: #617C8E; margin-top: .35rem; }}

/* Simple result group cards */
.result-box {{
    background: #FAFDFE;
    border: 1px solid #D5E5EE;
    border-radius: 11px;
    padding: .8rem .9rem .35rem;
    margin-bottom: .6rem;
}}
.result-box-title {{ color: {NCEV_NAVY}; font-weight: 800; font-size: 1rem; margin-bottom: .2rem; }}

/* Custom tables: never depend on dataframe theme */
.result-table {{ width: 100%; border-collapse: collapse; background: #FFFFFF; color: {NCEV_TEXT}; }}
.result-table th {{
    background: #DDEFF8;
    color: {NCEV_NAVY};
    border: 1px solid #AFC7D6;
    padding: .65rem .7rem;
    text-align: left;
    font-weight: 800;
}}
.result-table td {{
    background: #FFFFFF;
    color: {NCEV_TEXT};
    border: 1px solid #D1E0E9;
    padding: .6rem .7rem;
}}

/* Streamlit toolbar: preserve Share, suppress the extra cloud buttons. */
[data-testid="stToolbarActions"] > div > *:not(:first-child),
[data-testid="stToolbarActions"] > *:not(:first-child) {{ display: none !important; }}
[data-testid="stToolbar"] button, [data-testid="stToolbarActions"] button {{
    background: #F7FBFD !important;
    color: {NCEV_NAVY} !important;
    border-color: #C8DCE8 !important;
}}
[data-testid="stToolbar"] svg, [data-testid="stToolbarActions"] svg {{
    color: {NCEV_NAVY} !important;
    fill: {NCEV_NAVY} !important;
    stroke: {NCEV_NAVY} !important;
}}
</style>
""",
    unsafe_allow_html=True,
)


def ui_text(lang: str, vi: str, en: str, zh: str) -> str:
    return {"vi": vi, "en": en, "zh": zh}.get(lang, en)


def num(label, value, min_value=0.0, step=1.0, key=None, help=None, fmt="%.2f"):
    return st.number_input(
        label,
        min_value=float(min_value),
        value=float(value),
        step=float(step),
        key=key,
        help=help,
        format=fmt,
    )


def fmtv(x, n=2):
    if x is None or not np.isfinite(x):
        return "–"
    return f"{x:,.{n}f}"


def kpi(label: str, value: str, unit: str = "", note: str | None = None):
    unit_html = f'<span class="kpi-unit">{html.escape(unit)}</span>' if unit else ""
    note_html = f'<div class="kpi-note">{html.escape(note)}</div>' if note else ""
    st.markdown(
        '<div class="kpi-card">'
        f'<div class="kpi-label">{html.escape(str(label))}</div>'
        f'<div><span class="kpi-value">{html.escape(str(value))}</span>{unit_html}</div>'
        f'{note_html}'
        '</div>',
        unsafe_allow_html=True,
    )


def plot_theme(fig, x_title: str, y_title: str, height=390):
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color=NCEV_TEXT, size=13),
        margin=dict(l=45, r=20, t=20, b=45),
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend=dict(
            bgcolor="rgba(255,255,255,.95)",
            bordercolor="#C6D9E5",
            borderwidth=1,
            font=dict(color=NCEV_TEXT, size=12),
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#E3EDF3",
        linecolor="#87A8BC",
        tickfont=dict(color="#27475D", size=13),
        title_font=dict(color=NCEV_TEXT, size=14),
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#E3EDF3",
        linecolor="#87A8BC",
        tickfont=dict(color="#27475D", size=13),
        title_font=dict(color=NCEV_TEXT, size=14),
        zeroline=False,
    )


def html_table(df: pd.DataFrame, decimals=3):
    def format_value(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:,.{decimals}f}"
        return str(v)

    headers = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{html.escape(format_value(v))}</td>" for v in row.tolist())
        rows.append(f"<tr>{cells}</tr>")
    st.markdown(
        f'<table class="result-table"><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table>',
        unsafe_allow_html=True,
    )


def make_inputs(lang: str):
    st.markdown(f'<div class="section-title">{tr(lang,"influent")}</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    with cols[0]:
        q = num(tr(lang, "flow"), 500, 0, 10, key="q")
        cod = num(tr(lang, "cod"), 450, 0, 10, key="cod")
    with cols[1]:
        bod5 = num(tr(lang, "bod5"), 250, 0, 10, key="bod5")
        tss = num(tr(lang, "tss"), 220, 0, 10, key="tss")
    with cols[2]:
        nh4 = num(tr(lang, "nh4"), 35, 0, 1, key="nh4")
        tn = num(tr(lang, "tn"), 50, 0, 1, key="tn")
    with cols[3]:
        nox = num(tr(lang, "nox"), 0.5, 0, 0.5, key="nox")
        alk = num(tr(lang, "alk"), 250, 0, 10, key="alk")
    with cols[4]:
        tp = num(tr(lang, "tp"), 7, 0, 0.5, key="tp")
        temp = num(tr(lang, "temp"), 25, 0, 1, key="temp")
    do_in = num(tr(lang, "do_in"), 0.5, 0, 0.1, key="do_in")

    st.markdown(f'<div class="section-title">{tr(lang,"process")}</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    with cols[0]:
        v_an = num(tr(lang, "v_an"), 150, 0, 10, key="v_an")
        v_ae = num(tr(lang, "v_ae"), 350, 0, 10, key="v_ae")
    with cols[1]:
        ir = num(tr(lang, "ir"), 200, 0, 10, key="ir")
        ras = num(tr(lang, "ras"), 50, 0, 5, key="ras")
    with cols[2]:
        kla = num(tr(lang, "kla"), 75, 0, 5, key="kla")
        capture = num(tr(lang, "capture"), 99.5, 80, 0.1, key="capture")
    with cols[3]:
        srt_mode = st.radio(
            tr(lang, "srt_mode"),
            [tr(lang, "qwas_mode"), tr(lang, "srt_target_mode")],
            key="srt_mode",
        )
        if srt_mode == tr(lang, "qwas_mode"):
            was = num(tr(lang, "was"), 12, 0, 1, key="was")
            target_srt = None
        else:
            target_srt = num(tr(lang, "target_srt"), 15, 1, 1, key="target_srt")
            was = 12.0
    with cols[4]:
        days = num(tr(lang, "days"), 60, 5, 5, key="days")
        do_sat = num("DO saturation (mgO₂/L)", 8, 1, 0.5, key="do_sat")

    st.markdown(f'<div class="section-title">{tr(lang,"chemicals")}</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="group-title">{tr(lang,"carbon")}</div>', unsafe_allow_html=True)
        carbon_source = st.selectbox(tr(lang, "carbon_source"), ["Methanol", "Ethanol"], key="carbon_source")
        default_purity = 25.0 if carbon_source == "Methanol" else 95.0
        default_density = 0.90 if carbon_source == "Methanol" else 0.81
        carbon_purity = num(tr(lang, "carbon_purity"), default_purity, 1, 1, key="carbon_purity")
        carbon_density = num(tr(lang, "carbon_density"), default_density, 0.1, 0.01, key="carbon_density")
        tn_target = num(tr(lang, "tn_target"), 20, 0, 1, key="tn_target")
    with c2:
        st.markdown(f'<div class="group-title">{tr(lang,"naoh")}</div>', unsafe_allow_html=True)
        naoh_conc = num(tr(lang, "naoh_conc"), 10, 1, 1, key="naoh_conc")
        naoh_density = num(tr(lang, "naoh_density"), 1.11, 0.1, 0.01, key="naoh_density")
        alk_target = num(tr(lang, "alk_target"), 80, 0, 10, key="alk_target")
    with c3:
        st.markdown(f'<div class="group-title">{tr(lang,"pchemical")}</div>', unsafe_allow_html=True)
        p_type = st.selectbox(tr(lang, "p_type"), ["PAC", "Polytetsu"], key="p_type")
        bio_p = num(tr(lang, "bio_p"), 15, 0, 1, key="bio_p")
        tp_target = num(tr(lang, "tp_target"), 1, 0, 0.1, key="tp_target")
        p_density = num(tr(lang, "product_density"), 0, 0, 0.05, key="p_density")
        wet_ts = num(tr(lang, "ts_sludge"), 1, 0.1, 0.1, key="wet_ts")

    with st.expander(tr(lang, "initial")):
        a, b, c = st.columns(3)
        with a:
            mlss_an = num(tr(lang, "mlss_an0"), 3000, 0, 100, key="mlss_an")
            mlss_ae = num(tr(lang, "mlss_ae0"), 3000, 0, 100, key="mlss_ae")
        with b:
            do_an = num(tr(lang, "do_an0"), 0.2, 0, 0.1, key="do_an")
            do_ae = num(tr(lang, "do_ae0"), 2, 0, 0.1, key="do_ae")
        with c:
            nh4_0 = num(tr(lang, "nh4_0"), 5, 0, 1, key="nh4_0")
            nox_0 = num(tr(lang, "nox_0"), 10, 0, 1, key="nox_0")

    raw = InfluentInput(
        q=q, cod=cod, bod5=bod5, tss=tss, nh4=nh4, tn=tn, nox=nox,
        alkalinity=alk, dissolved_oxygen=do_in, temperature=temp, tp=tp,
    )
    init = InitialConditions(
        mlss_anoxic=mlss_an, mlss_aerobic=mlss_ae, do_anoxic=do_an,
        do_aerobic=do_ae, nh4=nh4_0, nox=nox_0,
    )
    cfg = ProcessConfig(
        q=q, v_anoxic=v_an, v_aerobic=v_ae, internal_recycle_ratio=ir / 100,
        ras_ratio=ras / 100, was_flow=was, clarifier_capture=capture / 100,
        kla_anoxic=0, kla_aerobic=kla, do_saturation=do_sat,
        simulation_days=min(days, 120), output_dt=0.2,
    )
    settings = {
        "target_srt": target_srt,
        "carbon": CarbonSettings(source=carbon_source, purity_pct=carbon_purity, density_kg_l=carbon_density),
        "tn_target": tn_target,
        "naoh": NaOHSettings(
            concentration_pct=naoh_conc,
            density_kg_l=naoh_density,
            target_alkalinity_mg_l_as_caco3=alk_target,
        ),
        "phos": PhosphorusSettings(
            tp_in_mg_l=tp,
            biological_removal_pct=bio_p,
            tp_target_mg_l=tp_target,
            chemical=p_type,
            pac_al2o3_pct=31,
            pac_al_to_p_molar=1.5,
            polytetsu_fe_pct=21,
            polytetsu_fe_to_p_molar=1.0,
            product_density_kg_l=p_density,
            wet_sludge_ts_pct=wet_ts,
        ),
    }
    return raw, init, cfg, settings


def run_pipeline(raw, init, cfg, settings, params, frac):
    fx = fractionate(raw, frac, init, i_xb=params.i_XB, i_xp=params.i_XP)
    inf = fx["influent_state"]
    an0 = fx["anoxic_initial"]
    ae0 = fx["aerobic_initial"]

    work_cfg = ProcessConfig(**asdict(cfg))
    target_srt = settings["target_srt"]

    if target_srt is not None:
        work_cfg, base = solve_was_for_target_srt(inf, an0, ae0, work_cfg, params, target_srt)
    else:
        base = simulate(inf, an0, ae0, work_cfg, params)

    carbon = optimize_external_carbon(
        inf, an0, ae0, work_cfg, params, settings["tn_target"], settings["carbon"], base_result=base
    )
    final_cfg = work_cfg
    final = carbon["result"]

    if target_srt is not None and carbon["dose"]["added_cod_mg_l"] > 0:
        inf_c = add_external_cod(inf, carbon["dose"]["added_cod_mg_l"])
        final_cfg, final = solve_was_for_target_srt(inf_c, an0, ae0, final_cfg, params, target_srt, max_iter=5)
        if final["final"]["tn_eff"] > settings["tn_target"]:
            carbon = optimize_external_carbon(
                inf, an0, ae0, final_cfg, params, settings["tn_target"], settings["carbon"], base_result=base
            )
            inf_c = add_external_cod(inf, carbon["dose"]["added_cod_mg_l"])
            final_cfg, final = solve_was_for_target_srt(inf_c, an0, ae0, final_cfg, params, target_srt, max_iter=4)

    carbon["result"] = final
    carbon["target_met"] = final["final"]["tn_eff"] <= settings["tn_target"]

    naoh = calculate_naoh(raw.q, final["final"]["alk_eff"], settings["naoh"])
    phos = phosphorus_chemical(raw.q, settings["phos"])
    sludge = total_sludge(
        final["final"]["was_ds_kg_d"],
        phos["chemical_sludge_ds_kg_d"],
        settings["phos"].wet_sludge_ts_pct,
    )

    return {
        "fractionation": fx,
        "base": base,
        "final": final,
        "carbon": carbon,
        "naoh": naoh,
        "phosphorus": phos,
        "sludge": sludge,
        "config": asdict(final_cfg),
        "settings": {
            "tn_target": settings["tn_target"],
            "target_srt": target_srt,
            "carbon": asdict(settings["carbon"]),
            "naoh": asdict(settings["naoh"]),
            "phos": asdict(settings["phos"]),
        },
    }


def show_results(case, lang):
    base = case["base"]["final"]
    final = case["final"]["final"]
    carbon = case["carbon"]["dose"]
    naoh = case["naoh"]
    p = case["phosphorus"]
    sl = case["sludge"]

    st.markdown(f'<div class="section-title">{tr(lang,"bio_result")}</div>', unsafe_allow_html=True)
    cols = st.columns(6)
    bio_vals = [
        ("COD", final["cod_eff"], "mg/L"),
        ("NH₄-N", final["nh4_eff"], "mgN/L"),
        ("NOₓ-N", final["nox_eff"], "mgN/L"),
        ("TN", final["tn_eff"], "mgN/L"),
        ("DO", final["do_eff"], "mg/L"),
        ("SRT", final["srt_d"], "d"),
    ]
    for col, (label, value, unit) in zip(cols, bio_vals):
        with col:
            kpi(label, fmtv(value, 2), unit)

    cols = st.columns(4)
    with cols[0]:
        kpi("MLSS Anoxic", fmtv(final["mlss_anoxic"], 0), "mg/L")
    with cols[1]:
        kpi("MLSS Aerobic", fmtv(final["mlss_aerobic"], 0), "mg/L")
    with cols[2]:
        kpi(tr(lang, "solver"), tr(lang, "pass") if final["solver_ok"] else tr(lang, "check"))
    with cols[3]:
        kpi(tr(lang, "steady"), tr(lang, "pass") if final["steady_ok"] else tr(lang, "check"))

    st.markdown(f'<div class="section-title">{tr(lang,"chemical_result")}</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="result-box"><div class="result-box-title">{tr(lang,"carbon")}: {html.escape(carbon["source"])}</div></div>',
            unsafe_allow_html=True,
        )
        kpi(tr(lang, "base_tn"), f'{base["tn_eff"]:.2f}', "mgN/L")
        kpi(tr(lang, "final_tn"), f'{final["tn_eff"]:.2f}', "mgN/L")
        kpi(tr(lang, "carbon_cod"), f'{carbon["added_cod_mg_l"]:.1f}', "mgCOD/L")
        kpi(tr(lang, "carbon_l_d"), f'{carbon["solution_l_d"]:.1f}', "L/d")
        kpi(tr(lang, "carbon_l_h"), f'{carbon["solution_l_h"]:.2f}', "L/h")
        if case["carbon"]["target_met"]:
            st.success(tr(lang, "no_carbon_needed") if carbon["added_cod_mg_l"] <= 1e-9 else tr(lang, "pass"))
        else:
            st.warning(tr(lang, "carbon_target_fail"))
    with c2:
        st.markdown(
            f'<div class="result-box"><div class="result-box-title">{tr(lang,"naoh")}</div></div>',
            unsafe_allow_html=True,
        )
        kpi("Alkalinity final", f'{final["alk_eff"]:.1f}', "mg/L as CaCO₃")
        kpi("Alkalinity deficit", f'{naoh["alkalinity_deficit_mg_l_as_caco3"]:.1f}', "mg/L as CaCO₃")
        kpi(tr(lang, "naoh_l_d"), f'{naoh["solution_l_d"]:.1f}', "L/d")
        kpi(tr(lang, "naoh_l_h"), f'{naoh["solution_l_h"]:.2f}', "L/h")
        st.caption(ui_text(
            lang,
            "NaOH được tính theo thiếu hụt độ kiềm của ASM1; chưa mô phỏng pH động trực tiếp.",
            "NaOH is calculated from ASM1 alkalinity deficit; direct dynamic pH is not modeled.",
            "NaOH 按 ASM1 碱度缺口计算；暂不直接模拟动态 pH。",
        ))
    with c3:
        st.markdown(
            f'<div class="result-box"><div class="result-box-title">{tr(lang,"pchemical")}: {html.escape(p["chemical"])}</div></div>',
            unsafe_allow_html=True,
        )
        kpi(tr(lang, "p_before"), f'{p["tp_after_bio_mg_l"]:.2f}', "mgP/L")
        kpi(tr(lang, "p_remove"), f'{p["p_remove_mg_l"]:.2f}', "mgP/L")
        kpi(tr(lang, "p_dose_mgl"), f'{p["product_dose_mg_l"]:.1f}', "mg/L")
        kpi(tr(lang, "p_dose"), f'{p["product_kg_d"]:.1f}', "kg/d")
        if p["product_l_d"] is not None:
            st.caption(f'{p["product_l_d"]:.1f} L/d · {p["product_l_h"]:.2f} L/h')
        st.caption(p["basis"])

    st.markdown(f'<div class="section-title">{tr(lang,"sludge")}</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    sludge_vals = [
        (tr(lang, "bio_sludge"), sl["biological_was_ds_kg_d"], "kgDS/d", 1),
        (tr(lang, "chem_sludge"), sl["chemical_sludge_ds_kg_d"], "kgDS/d", 1),
        (tr(lang, "total_sludge"), sl["total_ds_kg_d"], "kgDS/d", 1),
        (tr(lang, "wet_sludge"), sl["wet_sludge_m3_d"], "m³/d", 2),
        ("WAS flow", case["config"]["was_flow"], "m³/d", 2),
    ]
    for col, (label, value, unit, dec) in zip(cols, sludge_vals):
        with col:
            kpi(label, fmtv(value, dec), unit)

    st.markdown(f'<div class="section-title">{tr(lang,"trends")}</div>', unsafe_allow_html=True)
    df = pd.DataFrame(case["final"]["time_series"])
    trend = st.radio(
        ui_text(lang, "Nhóm biểu đồ", "Chart group", "图表组"),
        ["N", "COD / DO", "MLSS / SRT"],
        horizontal=True,
        label_visibility="collapsed",
        key="trend_v3",
    )
    if trend == "N":
        fig = go.Figure()
        for y, name in [("nh4_eff", "NH4-N"), ("nox_eff", "NOx-N"), ("tn_eff", "TN")]:
            fig.add_trace(go.Scatter(x=df.time_d, y=df[y], name=name, line=dict(width=2.5)))
        plot_theme(fig, "Time (d)", "mgN/L")
        st.plotly_chart(fig, use_container_width=True)
    elif trend == "COD / DO":
        fig = go.Figure()
        for y, name in [("cod_eff", "COD"), ("bod5_eff", "BOD5"), ("do_eff", "DO")]:
            fig.add_trace(go.Scatter(x=df.time_d, y=df[y], name=name, line=dict(width=2.5)))
        plot_theme(fig, "Time (d)", "mg/L")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = go.Figure()
        for y, name in [("mlss_anoxic", "MLSS Anoxic"), ("mlss_aerobic", "MLSS Aerobic")]:
            fig.add_trace(go.Scatter(x=df.time_d, y=df[y], name=name, line=dict(width=2.5)))
        plot_theme(fig, "Time (d)", "mg/L", 350)
        st.plotly_chart(fig, use_container_width=True)
        fig2 = go.Figure(go.Scatter(x=df.time_d, y=df.srt_d, name="SRT", line=dict(width=2.5)))
        plot_theme(fig2, "Time (d)", "d", 280)
        st.plotly_chart(fig2, use_container_width=True)

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            tr(lang, "download_csv"),
            df.to_csv(index=False).encode("utf-8"),
            "asm1_v3_timeseries.csv",
            "text/csv",
            use_container_width=True,
        )
    with d2:
        export = {
            "version": APP_VERSION,
            "final": case["final"]["final"],
            "carbon": case["carbon"]["dose"],
            "naoh": case["naoh"],
            "phosphorus": case["phosphorus"],
            "sludge": case["sludge"],
            "config": case["config"],
            "fractionation_qa": case["fractionation"]["qa"],
        }
        st.download_button(
            tr(lang, "download_json"),
            json.dumps(export, ensure_ascii=False, indent=2).encode("utf-8"),
            "asm1_v3_result.json",
            "application/json",
            use_container_width=True,
        )

    with st.expander(tr(lang, "advanced")):
        st.write("Fractionation QA", case["fractionation"]["qa"])
        state_df = pd.DataFrame({
            "State": STATE_NAMES,
            "Anoxic": [case["final"]["states"]["anoxic_final"][s] for s in STATE_NAMES],
            "Aerobic": [case["final"]["states"]["aerobic_final"][s] for s in STATE_NAMES],
            "Effluent": [case["final"]["states"]["effluent_final"][s] for s in STATE_NAMES],
        })
        html_table(state_df, decimals=4)


# ----- Header -----
header_left, header_right = st.columns([4.8, 1.4], vertical_alignment="center")
with header_left:
    st.image(str(LOGO_PATH), width=300)
with header_right:
    lang_label = st.selectbox(
        "Language / Ngôn ngữ / 语言",
        list(LANG_OPTIONS.keys()),
        index=0,
        key="lang_select_v3",
    )
lang = LANG_OPTIONS[lang_label]
st.markdown(
    f'<div class="ncev-rule"></div><div class="ncev-title">{tr(lang,"app_title")}</div>'
    f'<div class="ncev-sub">{tr(lang,"subtitle")} · {APP_VERSION}</div>',
    unsafe_allow_html=True,
)

# ----- Advanced sidebar -----
with st.sidebar:
    st.markdown(f"### {tr(lang, 'advanced')}")
    st.caption("ASM1 / fractionation calibration")
    params = ASM1Parameters()
    frac = FractionationAssumptions()
    params.mu_H = num("μH (1/d)", params.mu_H, 0.01, 0.1, key="muH_v3")
    params.mu_A = num("μA (1/d)", params.mu_A, 0.01, 0.05, key="muA_v3")
    params.Y_H = num("YH", params.Y_H, 0.01, 0.01, key="YH_v3")
    params.Y_A = num("YA", params.Y_A, 0.01, 0.01, key="YA_v3")
    frac.bod5_to_bcod = num("BOD5 / bCOD", frac.bod5_to_bcod, 0.05, 0.05, key="fbod_v3")
    frac.readily_biodegradable_fraction = num(
        "S_S / bCOD", frac.readily_biodegradable_fraction, 0, 0.05, key="fss_v3"
    )
    frac.soluble_inert_fraction = num("S_I / inert COD", frac.soluble_inert_fraction, 0, 0.05, key="fsi_v3")
    st.warning(tr(lang, "warning_model"))
    st.caption(ui_text(
        lang,
        "Tính NaOH hiện chưa cộng thêm độ kiềm tiêu hao do thủy phân PAC/Polytetsu.",
        "NaOH calculation currently excludes extra alkalinity consumption from PAC/Polytetsu hydrolysis.",
        "NaOH 计算目前尚未计入 PAC/Polytetsu 水解造成的额外碱度消耗。",
    ))

# ----- Main navigation -----
view = st.radio(
    ui_text(lang, "Chức năng", "Mode", "功能"),
    [tr(lang, "run"), tr(lang, "compare")],
    horizontal=True,
    label_visibility="collapsed",
    key="nav_v3",
)

if view == tr(lang, "run"):
    raw, init, cfg, settings = make_inputs(lang)
    if st.button("▶ " + tr(lang, "run"), type="primary", use_container_width=True, key="run_v3"):
        try:
            with st.spinner("ASM1 / BDF ..."):
                st.session_state["case_v3"] = run_pipeline(raw, init, cfg, settings, params, frac)
        except Exception as exc:
            st.error(str(exc))
    if "case_v3" in st.session_state:
        show_results(st.session_state["case_v3"], lang)
else:
    if "case_v3" not in st.session_state:
        st.info(ui_text(lang, "Hãy chạy trường hợp cơ sở trước.", "Run the base case first.", "请先运行基准工况。"))
    else:
        st.caption(ui_text(
            lang,
            "Thay đổi tuần hoàn nội, thể tích hiếu khí, WAS hoặc KLa rồi so sánh với trường hợp cơ sở.",
            "Change internal recycle, aerobic volume, WAS or KLa and compare with the base case.",
            "修改内回流、好氧池容积、WAS 或 KLa，然后与基准工况比较。",
        ))
        base_case = st.session_state["case_v3"]
        bcfg = ProcessConfig(**base_case["config"])
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            sc_ir = num(tr(lang, "ir"), bcfg.internal_recycle_ratio * 100, 0, 10, key="sc_ir_v3")
        with c2:
            sc_vae = num(tr(lang, "v_ae"), bcfg.v_aerobic, 1, 10, key="sc_vae_v3")
        with c3:
            sc_kla = num(tr(lang, "kla"), bcfg.kla_aerobic, 0, 5, key="sc_kla_v3")
        with c4:
            sc_was = num(tr(lang, "was"), bcfg.was_flow, 0, 1, key="sc_was_v3")
        if st.button(
            ui_text(lang, "▶ Chạy kịch bản", "▶ Run scenario", "▶ 运行情景"),
            type="primary",
            use_container_width=True,
            key="scenario_v3_btn",
        ):
            try:
                raw2 = InfluentInput(**base_case["fractionation"]["input"])
                init2 = InitialConditions(**base_case["fractionation"]["initial_conditions"])
                cfg2 = ProcessConfig(**base_case["config"])
                cfg2.internal_recycle_ratio = sc_ir / 100
                cfg2.v_aerobic = sc_vae
                cfg2.kla_aerobic = sc_kla
                cfg2.was_flow = sc_was
                settings2 = deepcopy(base_case["settings"])
                settings_obj = {
                    "target_srt": None,
                    "carbon": CarbonSettings(**settings2["carbon"]),
                    "tn_target": settings2["tn_target"],
                    "naoh": NaOHSettings(**settings2["naoh"]),
                    "phos": PhosphorusSettings(**settings2["phos"]),
                }
                st.session_state["scenario_v3"] = run_pipeline(raw2, init2, cfg2, settings_obj, params, frac)
            except Exception as exc:
                st.error(str(exc))

        if "scenario_v3" in st.session_state:
            base_final = base_case["final"]["final"]
            scenario_final = st.session_state["scenario_v3"]["final"]["final"]
            rows = []
            for label, key, unit in [
                ("COD", "cod_eff", "mg/L"),
                ("NH4-N", "nh4_eff", "mgN/L"),
                ("NOx-N", "nox_eff", "mgN/L"),
                ("TN", "tn_eff", "mgN/L"),
                ("SRT", "srt_d", "d"),
                ("MLSS aerobic", "mlss_aerobic", "mg/L"),
            ]:
                rows.append([label, base_final[key], scenario_final[key], scenario_final[key] - base_final[key], unit])
            comp_df = pd.DataFrame(rows, columns=["KPI", "Base", "Scenario", "Δ", "Unit"])
            html_table(comp_df, decimals=3)

st.divider()
st.caption(
    f"NCEV {APP_VERSION} · ASM1 + SciPy BDF · Engineering screening / scenario analysis. "
    "Calibrate against plant data before design guarantees."
)
