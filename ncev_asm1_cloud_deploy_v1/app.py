
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from asm1_engine import ASM1Parameters, ProcessConfig, simulate, STATE_NAMES
from fractionation import (
    InfluentInput,
    FractionationAssumptions,
    InitialConditions,
    fractionate,
)


st.set_page_config(
    page_title="NCEV AO ASM1 Simulator Cloud",
    page_icon="💧",
    layout="wide",
)

APP_VERSION = "Cloud MVP v1.0"

# Keep only current simulation objects in session to reduce long-running memory growth.
if "_app_version" not in st.session_state:
    st.session_state["_app_version"] = APP_VERSION


st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px;}
div[data-testid="stMetricValue"] {font-size: 1.65rem;}
.small-note {font-size: 0.86rem; color: #666;}
.kpi-pass {font-weight: 700;}
</style>
""", unsafe_allow_html=True)

DEFAULT_TARGETS = {
    "COD": 100.0,
    "BOD5": 30.0,
    "TSS": 50.0,
    "NH4-N": 5.0,
    "TN": 20.0,
    "DO": 1.0,
    "Alkalinity": 50.0,
}


def number(label, value, min_value=0.0, step=1.0, help=None, key=None, fmt="%.2f"):
    return st.number_input(
        label, min_value=float(min_value), value=float(value), step=float(step),
        help=help, key=key, format=fmt
    )


def build_case(prefix: str = ""):
    st.subheader("1. Influent")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        q = number("Flow Q (m³/d)", 500, 0, 10, key=f"{prefix}q")
        cod = number("COD (mg/L)", 450, 0, 10, key=f"{prefix}cod")
    with c2:
        bod5 = number("BOD₅ (mg/L)", 250, 0, 10, key=f"{prefix}bod5")
        tss = number("TSS (mg/L)", 220, 0, 10, key=f"{prefix}tss")
    with c3:
        nh4 = number("NH₄-N (mgN/L)", 35, 0, 1, key=f"{prefix}nh4")
        tn = number("TN (mgN/L)", 50, 0, 1, key=f"{prefix}tn")
    with c4:
        nox = number("NOₓ-N influent (mgN/L)", 0.5, 0, 0.5, key=f"{prefix}nox")
        alk = number("Alkalinity (mg/L as CaCO₃)", 250, 0, 10, key=f"{prefix}alk")
    with c5:
        do_in = number("Influent DO (mgO₂/L)", 0.5, 0, 0.1, key=f"{prefix}do_in")
        temp = number("Temperature (°C)", 25, 0, 1, key=f"{prefix}temp")

    st.subheader("2. Process")
    p1, p2, p3, p4, p5 = st.columns(5)
    with p1:
        v_an = number("Anoxic volume (m³)", 150, 0, 10, key=f"{prefix}v_an")
        v_ae = number("Aerobic volume (m³)", 350, 0, 10, key=f"{prefix}v_ae")
    with p2:
        ir_pct = number("Internal recycle (%Q)", 200, 0, 10, key=f"{prefix}ir")
        ras_pct = number("RAS (%Q)", 50, 0, 5, key=f"{prefix}ras")
    with p3:
        was = number("WAS flow (m³/d)", 12, 0, 1, key=f"{prefix}was")
        capture_pct = number("Clarifier capture (%)", 99.5, 0, 0.1, key=f"{prefix}cap")
    with p4:
        kla_an = number("KLa anoxic (1/d)", 0, 0, 1, key=f"{prefix}kla_an")
        kla_ae = number("KLa aerobic (1/d)", 75, 0, 5, key=f"{prefix}kla_ae")
    with p5:
        do_sat = number("DO saturation (mgO₂/L)", 8, 0, 0.5, key=f"{prefix}do_sat")
        days = number("Simulation duration (d)", 45, 1, 5, key=f"{prefix}days")

    st.subheader("3. Initial condition")
    i1, i2, i3 = st.columns(3)
    with i1:
        mlss_an = number("Initial anoxic MLSS (mg/L)", 3000, 0, 100, key=f"{prefix}mlss_an")
        mlss_ae = number("Initial aerobic MLSS (mg/L)", 3000, 0, 100, key=f"{prefix}mlss_ae")
    with i2:
        do_an = number("Initial anoxic DO (mg/L)", 0.2, 0, 0.1, key=f"{prefix}do_an")
        do_ae = number("Initial aerobic DO (mg/L)", 2.0, 0, 0.1, key=f"{prefix}do_ae")
    with i3:
        nh4_0 = number("Initial NH₄-N (mgN/L)", 5, 0, 1, key=f"{prefix}nh4_0")
        nox_0 = number("Initial NOₓ-N (mgN/L)", 10, 0, 1, key=f"{prefix}nox_0")

    raw = InfluentInput(
        q=q, cod=cod, bod5=bod5, tss=tss, nh4=nh4, tn=tn, nox=nox,
        alkalinity=alk, dissolved_oxygen=do_in, temperature=temp
    )
    init = InitialConditions(
        mlss_anoxic=mlss_an, mlss_aerobic=mlss_ae,
        do_anoxic=do_an, do_aerobic=do_ae, nh4=nh4_0, nox=nox_0
    )
    cfg = ProcessConfig(
        q=q,
        v_anoxic=v_an,
        v_aerobic=v_ae,
        internal_recycle_ratio=ir_pct / 100.0,
        ras_ratio=ras_pct / 100.0,
        was_flow=was,
        clarifier_capture=capture_pct / 100.0,
        kla_anoxic=kla_an,
        kla_aerobic=kla_ae,
        do_saturation=do_sat,
        simulation_days=days,
        output_dt=0.1,
    )
    return raw, init, cfg


def run_case(raw, init, cfg, frac, params):
    # Cloud safety guard: avoid unnecessarily large output arrays.
    cfg.output_dt = max(float(cfg.output_dt), 0.05)
    cfg.simulation_days = min(float(cfg.simulation_days), 120.0)

    fx = fractionate(raw, frac, init, i_xb=params.i_XB, i_xp=params.i_XP)
    sim = simulate(
        fx["influent_state"],
        fx["anoxic_initial"],
        fx["aerobic_initial"],
        cfg,
        params
    )
    return {"fractionation": fx, "simulation": sim}


def status(value, target, direction):
    if direction == "max":
        return "PASS" if value <= target else "CHECK"
    return "PASS" if value >= target else "CHECK"


def show_results(case, targets):
    sim = case["simulation"]
    fx = case["fractionation"]
    f = sim["final"]

    st.subheader("Result summary")
    cols = st.columns(4)
    metrics = [
        ("COD", f["cod_eff"], targets["COD"], "max", "mg/L"),
        ("NH₄-N", f["nh4_eff"], targets["NH4-N"], "max", "mgN/L"),
        ("TN", f["tn_eff"], targets["TN"], "max", "mgN/L"),
        ("TSS", f["tss_eff"], targets["TSS"], "max", "mg/L"),
    ]
    for col, (name, val, tar, direction, unit) in zip(cols, metrics):
        with col:
            st.metric(name, f"{val:.2f} {unit}", f"Target {tar:g}")
            st.caption(status(val, tar, direction))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("SRT", f"{f['srt_d']:.2f} d")
    with c2:
        st.metric("MLSS aerobic", f"{f['mlss_aerobic']:.0f} mg/L")
    with c3:
        st.metric("DO effluent", f"{f['do_eff']:.2f} mg/L")
        st.caption(status(f["do_eff"], targets["DO"], "min"))
    with c4:
        st.metric("Alkalinity", f"{f['alk_eff']:.1f} mg/L")
        st.caption(status(f["alk_eff"], targets["Alkalinity"], "min"))

    solver_col, steady_col, frac_col = st.columns(3)
    with solver_col:
        st.metric("Solver", "PASS" if f["solver_ok"] else "CHECK")
    with steady_col:
        st.metric("Steady state", "PASS" if f["steady_ok"] else "CHECK")
        st.caption(f"Index = {f['steady_index_per_d']:.2e} /d")
    with frac_col:
        qa = fx["qa"]
        st.metric("Fractionation QA", "PASS" if qa["tss_ok"] else "CHECK")
        if qa["tss_ratio"] is not None:
            st.caption(f"Predicted/measured TSS = {qa['tss_ratio']:.2f}")

    ts = sim["time_series"]
    df = pd.DataFrame(ts)

    st.subheader("Dynamic trends")
    tab1, tab2, tab3 = st.tabs(["Nitrogen", "DO & organic matter", "Biomass / SRT"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["time_d"], y=df["nh4_eff"], name="NH4-N"))
        fig.add_trace(go.Scatter(x=df["time_d"], y=df["nox_eff"], name="NOx-N"))
        fig.add_trace(go.Scatter(x=df["time_d"], y=df["tn_eff"], name="TN"))
        fig.update_layout(xaxis_title="Time (d)", yaxis_title="mgN/L", height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["time_d"], y=df["do_eff"], name="DO"))
        fig.add_trace(go.Scatter(x=df["time_d"], y=df["cod_eff"], name="COD"))
        fig.add_trace(go.Scatter(x=df["time_d"], y=df["bod5_eff"], name="BOD5"))
        fig.update_layout(xaxis_title="Time (d)", yaxis_title="mg/L", height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["time_d"], y=df["mlss_anoxic"], name="MLSS Anoxic"))
        fig.add_trace(go.Scatter(x=df["time_d"], y=df["mlss_aerobic"], name="MLSS Aerobic"))
        fig.update_layout(xaxis_title="Time (d)", yaxis_title="mg/L", height=420)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df["time_d"], y=df["srt_d"], name="SRT"))
        fig2.update_layout(xaxis_title="Time (d)", yaxis_title="SRT (d)", height=320)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Downloads")
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Download time series CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="asm1_time_series.csv",
            mime="text/csv",
        )
    with d2:
        export = {
            "final": sim["final"],
            "fractionation_qa": fx["qa"],
            "metadata": sim["metadata"],
        }
        st.download_button(
            "Download result JSON",
            json.dumps(export, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="asm1_result.json",
            mime="application/json",
        )

    with st.expander("Advanced: final ASM1 states"):
        state_df = pd.DataFrame({
            "State": STATE_NAMES,
            "Anoxic": [sim["states"]["anoxic_final"][s] for s in STATE_NAMES],
            "Aerobic": [sim["states"]["aerobic_final"][s] for s in STATE_NAMES],
            "Effluent": [sim["states"]["effluent_final"][s] for s in STATE_NAMES],
        })
        st.dataframe(state_df, use_container_width=True, hide_index=True)


st.title("💧 NCEV AO ASM1 Simulator – Cloud MVP")
st.caption(
    "MVP engineering simulator: ASM1 (13 states / 8 processes) + Anoxic/Aerobic + recycle + simple clarifier + SciPy BDF."
)

with st.sidebar:
    st.header("Targets")
    targets = {}
    targets["COD"] = number("COD target", DEFAULT_TARGETS["COD"], 0, 5, key="tar_cod")
    targets["BOD5"] = number("BOD5 target", DEFAULT_TARGETS["BOD5"], 0, 5, key="tar_bod")
    targets["TSS"] = number("TSS target", DEFAULT_TARGETS["TSS"], 0, 5, key="tar_tss")
    targets["NH4-N"] = number("NH4-N target", DEFAULT_TARGETS["NH4-N"], 0, 1, key="tar_nh4")
    targets["TN"] = number("TN target", DEFAULT_TARGETS["TN"], 0, 1, key="tar_tn")
    targets["DO"] = number("Min DO", DEFAULT_TARGETS["DO"], 0, 0.2, key="tar_do")
    targets["Alkalinity"] = number("Min alkalinity", DEFAULT_TARGETS["Alkalinity"], 0, 10, key="tar_alk")

    st.divider()
    st.warning(
        "MVP limitation: clarifier is a simple separator and kinetic temperature correction is not automatic. "
        "Calibrate with plant data before design guarantees."
    )

    st.divider()
    if st.button("Reset cached results", use_container_width=True):
        st.session_state.pop("base_case", None)
        st.session_state.pop("scenario_case", None)
        st.success("Results cleared.")

main_tab, compare_tab, advanced_tab = st.tabs(["Run simulation", "Compare scenarios", "Advanced model"])

with advanced_tab:
    st.subheader("ASM1 parameters")
    st.caption("Only change these if you understand the model or are calibrating against real plant data.")
    params = ASM1Parameters()
    frac = FractionationAssumptions()

    a, b, c = st.columns(3)
    with a:
        params.mu_H = number("μH (1/d)", params.mu_H, 0.01, 0.1, key="adv_muH")
        params.b_H = number("bH (1/d)", params.b_H, 0.0, 0.01, key="adv_bH")
        params.Y_H = number("YH", params.Y_H, 0.01, 0.01, key="adv_YH")
    with b:
        params.mu_A = number("μA (1/d)", params.mu_A, 0.01, 0.05, key="adv_muA")
        params.b_A = number("bA (1/d)", params.b_A, 0.0, 0.01, key="adv_bA")
        params.Y_A = number("YA", params.Y_A, 0.01, 0.01, key="adv_YA")
    with c:
        params.k_h = number("kh (1/d)", params.k_h, 0.01, 0.1, key="adv_kh")
        params.k_a = number("ka", params.k_a, 0.0, 0.01, key="adv_ka")
        params.eta_g = number("ηg", params.eta_g, 0.0, 0.05, key="adv_etag")

    st.subheader("Influent fractionation")
    f1, f2 = st.columns(2)
    with f1:
        frac.bod5_to_bcod = number("BOD5 / bCOD", frac.bod5_to_bcod, 0.05, 0.05, key="adv_bcod")
        frac.readily_biodegradable_fraction = number("S_S / bCOD", frac.readily_biodegradable_fraction, 0.0, 0.05, key="adv_ss")
    with f2:
        frac.soluble_inert_fraction = number("S_I / inert COD", frac.soluble_inert_fraction, 0.0, 0.05, key="adv_si")
        frac.soluble_org_n_fraction = number("S_ND / free organic N", frac.soluble_org_n_fraction, 0.0, 0.05, key="adv_snd")

# Use parameters even if user has not clicked Advanced tab.
if "params" not in locals():
    params = ASM1Parameters()
    frac = FractionationAssumptions()

with main_tab:
    raw, init, cfg = build_case("base_")
    run = st.button("▶ Run simulation", type="primary", use_container_width=True)
    if run:
        try:
            with st.spinner("Solving ASM1 ODE system..."):
                st.session_state["base_case"] = run_case(raw, init, cfg, frac, params)
        except Exception as exc:
            st.error(f"Simulation failed: {exc}")
            st.info("Check COD/BOD/TN consistency, WAS < Q, tank volumes > 0, and try again.")

    if "base_case" in st.session_state:
        show_results(st.session_state["base_case"], targets)

with compare_tab:
    st.subheader("Scenario comparison")
    st.caption(
        "Run the base case first. Then change only the scenario controls below to see the engineering effect."
    )

    if "base_case" not in st.session_state:
        st.info("Run a base simulation in the first tab before comparing scenarios.")
    else:
        base_case = st.session_state["base_case"]
        base_cfg_data = base_case["simulation"]["metadata"]["config"]
        base_input = base_case["fractionation"]["input"]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            sc_ir = number("Scenario internal recycle (%Q)", base_cfg_data["internal_recycle_ratio"]*100, 0, 10, key="sc_ir")
        with c2:
            sc_ras = number("Scenario RAS (%Q)", base_cfg_data["ras_ratio"]*100, 0, 5, key="sc_ras")
        with c3:
            sc_van = number("Scenario anoxic volume (m³)", base_cfg_data["v_anoxic"], 0, 10, key="sc_van")
        with c4:
            sc_vae = number("Scenario aerobic volume (m³)", base_cfg_data["v_aerobic"], 0, 10, key="sc_vae")

        c5, c6, c7 = st.columns(3)
        with c5:
            sc_was = number("Scenario WAS (m³/d)", base_cfg_data["was_flow"], 0, 1, key="sc_was")
        with c6:
            sc_kla = number("Scenario KLa aerobic (1/d)", base_cfg_data["kla_aerobic"], 0, 5, key="sc_kla")
        with c7:
            sc_days = number("Scenario duration (d)", base_cfg_data["simulation_days"], 1, 5, key="sc_days")

        if st.button("Run scenario", use_container_width=True):
            try:
                raw2 = InfluentInput(**base_input)
                init2 = InitialConditions(**base_case["fractionation"]["initial_conditions"])
                cfg2 = ProcessConfig(**base_cfg_data)
                cfg2.internal_recycle_ratio = sc_ir / 100
                cfg2.ras_ratio = sc_ras / 100
                cfg2.v_anoxic = sc_van
                cfg2.v_aerobic = sc_vae
                cfg2.was_flow = sc_was
                cfg2.kla_aerobic = sc_kla
                cfg2.simulation_days = sc_days

                base_frac = FractionationAssumptions(**base_case["fractionation"]["assumptions"])
                base_params = ASM1Parameters(**base_case["simulation"]["metadata"]["parameters"])
                st.session_state["scenario_case"] = run_case(raw2, init2, cfg2, base_frac, base_params)
            except Exception as exc:
                st.error(f"Scenario failed: {exc}")
                st.info("Try returning the changed scenario values closer to the base case.")

        if "scenario_case" in st.session_state:
            b = base_case["simulation"]["final"]
            s = st.session_state["scenario_case"]["simulation"]["final"]
            comparison = pd.DataFrame([
                ["NH4-N", b["nh4_eff"], s["nh4_eff"], s["nh4_eff"]-b["nh4_eff"], "mgN/L"],
                ["NOx-N", b["nox_eff"], s["nox_eff"], s["nox_eff"]-b["nox_eff"], "mgN/L"],
                ["TN", b["tn_eff"], s["tn_eff"], s["tn_eff"]-b["tn_eff"], "mgN/L"],
                ["COD", b["cod_eff"], s["cod_eff"], s["cod_eff"]-b["cod_eff"], "mg/L"],
                ["DO", b["do_eff"], s["do_eff"], s["do_eff"]-b["do_eff"], "mg/L"],
                ["SRT", b["srt_d"], s["srt_d"], s["srt_d"]-b["srt_d"], "d"],
                ["MLSS aerobic", b["mlss_aerobic"], s["mlss_aerobic"], s["mlss_aerobic"]-b["mlss_aerobic"], "mg/L"],
            ], columns=["KPI", "Base", "Scenario", "Δ Scenario-Base", "Unit"])
            st.dataframe(comparison, use_container_width=True, hide_index=True)

            bdf = pd.DataFrame(base_case["simulation"]["time_series"])
            sdf = pd.DataFrame(st.session_state["scenario_case"]["simulation"]["time_series"])

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=bdf["time_d"], y=bdf["tn_eff"], name="TN Base"))
            fig.add_trace(go.Scatter(x=sdf["time_d"], y=sdf["tn_eff"], name="TN Scenario"))
            fig.update_layout(xaxis_title="Time (d)", yaxis_title="TN (mgN/L)", height=420)
            st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption(
    "Model scope: ASM1 biological engine + simple secondary clarifier. "
    "Use for engineering screening / learning / scenario analysis, not as the sole basis for compliance or equipment guarantees."
)

st.caption(f"{APP_VERSION} · Python/SciPy ASM1 engine · Results are stored only in the current browser session.")
