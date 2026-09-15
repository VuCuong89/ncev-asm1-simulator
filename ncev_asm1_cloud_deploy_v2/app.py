from __future__ import annotations

import json
from dataclasses import asdict
from copy import deepcopy

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from asm1_engine import ASM1Parameters, ProcessConfig, simulate, STATE_NAMES
from fractionation import InfluentInput, FractionationAssumptions, InitialConditions, fractionate
from chemistry import (
    CarbonSettings,
    NaOHSettings,
    PhosphorusSettings,
    optimize_external_carbon,
    calculate_naoh,
    phosphorus_chemical,
    total_sludge,
    solve_was_for_target_srt,
    add_external_cod,
)
from i18n import LANG_OPTIONS, tr

APP_VERSION = "Cloud v2.0"
NCEV_BLUE = "#075DA8"
NCEV_CYAN = "#19A9C7"
NCEV_NAVY = "#0A365D"
LOGO_URL = "https://agjob.vn/storage/nakagawa-logo.webp"

st.set_page_config(
    page_title="NCEV AO ASM1 Simulator",
    page_icon="💧",
    layout="wide",
)

st.markdown(
    f"""
<style>
.block-container {{padding-top: 1rem; padding-bottom: 2.5rem; max-width: 1550px;}}
[data-testid="stMetricValue"] {{font-size: 1.5rem;}}
.ncev-header {{
  display:flex; align-items:center; gap:18px; border-bottom:3px solid {NCEV_CYAN};
  padding:4px 0 10px 0; margin-bottom:8px;
}}
.ncev-logo {{height:58px; max-width:250px; object-fit:contain;}}
.ncev-wordmark {{font-weight:800; color:{NCEV_BLUE}; font-size:2rem; letter-spacing:0.04em;}}
.ncev-title {{color:{NCEV_NAVY}; font-weight:800; font-size:1.65rem; line-height:1.15;}}
.ncev-sub {{color:#536878; font-size:0.92rem;}}
.section-title {{color:{NCEV_NAVY}; font-size:1.05rem; font-weight:800; margin:0.3rem 0 0.5rem;}}
.info-card {{border-left:4px solid {NCEV_CYAN}; padding:0.55rem 0.8rem; background:#F7FBFD; border-radius:6px;}}
.chem-card {{border:1px solid #D6E6EF; padding:0.75rem 0.9rem; border-radius:8px; background:#FBFDFE;}}
.small {{font-size:0.84rem;color:#637786;}}
</style>
""",
    unsafe_allow_html=True,
)


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


def metric_card(label, value, unit="", delta=None):
    st.metric(label, f"{value} {unit}".strip(), delta=delta)


def make_inputs(lang: str):
    st.markdown(f'<div class="section-title">{tr(lang,"influent")}</div>', unsafe_allow_html=True)
    a,b,c,d,e = st.columns(5)
    with a:
        q=num(tr(lang,"flow"),500,0,10,key="q")
        cod=num(tr(lang,"cod"),450,0,10,key="cod")
    with b:
        bod5=num(tr(lang,"bod5"),250,0,10,key="bod5")
        tss=num(tr(lang,"tss"),220,0,10,key="tss")
    with c:
        nh4=num(tr(lang,"nh4"),35,0,1,key="nh4")
        tn=num(tr(lang,"tn"),50,0,1,key="tn")
    with d:
        nox=num(tr(lang,"nox"),0.5,0,0.5,key="nox")
        alk=num(tr(lang,"alk"),250,0,10,key="alk")
    with e:
        tp=num(tr(lang,"tp"),7,0,0.5,key="tp")
        temp=num(tr(lang,"temp"),25,0,1,key="temp")
    do_in=num(tr(lang,"do_in"),0.5,0,0.1,key="do_in")

    st.markdown(f'<div class="section-title">{tr(lang,"process")}</div>', unsafe_allow_html=True)
    a,b,c,d,e = st.columns(5)
    with a:
        v_an=num(tr(lang,"v_an"),150,0,10,key="v_an")
        v_ae=num(tr(lang,"v_ae"),350,0,10,key="v_ae")
    with b:
        ir=num(tr(lang,"ir"),200,0,10,key="ir")
        ras=num(tr(lang,"ras"),50,0,5,key="ras")
    with c:
        kla=num(tr(lang,"kla"),75,0,5,key="kla")
        capture=num(tr(lang,"capture"),99.5,80,0.1,key="capture")
    with d:
        srt_mode=st.radio(
            tr(lang,"srt_mode"),
            [tr(lang,"qwas_mode"),tr(lang,"srt_target_mode")],
            horizontal=False,
            key="srt_mode"
        )
        if srt_mode==tr(lang,"qwas_mode"):
            was=num(tr(lang,"was"),12,0,1,key="was")
            target_srt=None
        else:
            target_srt=num(tr(lang,"target_srt"),15,1,1,key="target_srt")
            was=12.0
    with e:
        days=num(tr(lang,"days"),60,5,5,key="days")
        do_sat=num("DO saturation (mgO₂/L)",8,1,0.5,key="do_sat")

    st.markdown(f'<div class="section-title">{tr(lang,"chemicals")}</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown(f"**{tr(lang,'carbon')}**")
        carbon_source=st.selectbox(tr(lang,"carbon_source"),["Methanol","Ethanol"],key="carbon_source")
        default_purity=25.0 if carbon_source=="Methanol" else 95.0
        default_density=0.90 if carbon_source=="Methanol" else 0.81
        carbon_purity=num(tr(lang,"carbon_purity"),default_purity,1,1,key="carbon_purity")
        carbon_density=num(tr(lang,"carbon_density"),default_density,0.1,0.01,key="carbon_density")
        tn_target=num(tr(lang,"tn_target"),20,0,1,key="tn_target")
    with c2:
        st.markdown(f"**{tr(lang,'naoh')}**")
        naoh_conc=num(tr(lang,"naoh_conc"),10,1,1,key="naoh_conc")
        naoh_density=num(tr(lang,"naoh_density"),1.11,0.1,0.01,key="naoh_density")
        alk_target=num(tr(lang,"alk_target"),80,0,10,key="alk_target")
    with c3:
        st.markdown(f"**{tr(lang,'pchemical')}**")
        p_type=st.selectbox(tr(lang,"p_type"),["PAC","Polytetsu"],key="p_type")
        bio_p=num(tr(lang,"bio_p"),15,0,1,key="bio_p")
        tp_target=num(tr(lang,"tp_target"),1,0,0.1,key="tp_target")
        p_density=num(tr(lang,"product_density"),0,0,0.05,key="p_density")
        wet_ts=num(tr(lang,"ts_sludge"),1,0.1,0.1,key="wet_ts")

    with st.expander(tr(lang,"initial")):
        a,b,c = st.columns(3)
        with a:
            mlss_an=num(tr(lang,"mlss_an0"),3000,0,100,key="mlss_an")
            mlss_ae=num(tr(lang,"mlss_ae0"),3000,0,100,key="mlss_ae")
        with b:
            do_an=num(tr(lang,"do_an0"),0.2,0,0.1,key="do_an")
            do_ae=num(tr(lang,"do_ae0"),2,0,0.1,key="do_ae")
        with c:
            nh4_0=num(tr(lang,"nh4_0"),5,0,1,key="nh4_0")
            nox_0=num(tr(lang,"nox_0"),10,0,1,key="nox_0")

    raw=InfluentInput(q=q,cod=cod,bod5=bod5,tss=tss,nh4=nh4,tn=tn,nox=nox,
                      alkalinity=alk,dissolved_oxygen=do_in,temperature=temp,tp=tp)
    init=InitialConditions(mlss_anoxic=mlss_an,mlss_aerobic=mlss_ae,do_anoxic=do_an,
                           do_aerobic=do_ae,nh4=nh4_0,nox=nox_0)
    cfg=ProcessConfig(q=q,v_anoxic=v_an,v_aerobic=v_ae,internal_recycle_ratio=ir/100,
                      ras_ratio=ras/100,was_flow=was,clarifier_capture=capture/100,
                      kla_anoxic=0,kla_aerobic=kla,do_saturation=do_sat,
                      simulation_days=min(days,120),output_dt=0.2)
    settings={
        "target_srt":target_srt,
        "carbon":CarbonSettings(source=carbon_source,purity_pct=carbon_purity,density_kg_l=carbon_density),
        "tn_target":tn_target,
        "naoh":NaOHSettings(concentration_pct=naoh_conc,density_kg_l=naoh_density,
                             target_alkalinity_mg_l_as_caco3=alk_target),
        "phos":PhosphorusSettings(tp_in_mg_l=tp,biological_removal_pct=bio_p,tp_target_mg_l=tp_target,
                                   chemical=p_type,pac_al2o3_pct=31,pac_al_to_p_molar=1.5,
                                   polytetsu_fe_pct=21,polytetsu_fe_to_p_molar=1.0,
                                   product_density_kg_l=p_density,wet_sludge_ts_pct=wet_ts),
    }
    return raw,init,cfg,settings


def run_pipeline(raw,init,cfg,settings,params,frac):
    fx=fractionate(raw,frac,init,i_xb=params.i_XB,i_xp=params.i_XP)
    inf=fx["influent_state"]
    an0=fx["anoxic_initial"]
    ae0=fx["aerobic_initial"]

    work_cfg=ProcessConfig(**asdict(cfg))
    target_srt=settings["target_srt"]

    # Base biological run / optional target-SRT calculation.
    if target_srt is not None:
        work_cfg,base=solve_was_for_target_srt(inf,an0,ae0,work_cfg,params,target_srt)
    else:
        base=simulate(inf,an0,ae0,work_cfg,params)

    # Carbon optimization is tied back to the deterministic ASM1 model.
    carbon=optimize_external_carbon(
        inf,an0,ae0,work_cfg,params,settings["tn_target"],settings["carbon"],base_result=base
    )

    final_cfg=work_cfg
    final=carbon["result"]

    # If target SRT is selected, re-balance WAS after carbon addition and re-check TN once.
    if target_srt is not None and carbon["dose"]["added_cod_mg_l"]>0:
        inf_c=add_external_cod(inf,carbon["dose"]["added_cod_mg_l"])
        final_cfg,final=solve_was_for_target_srt(inf_c,an0,ae0,final_cfg,params,target_srt,max_iter=5)
        if final["final"]["tn_eff"]>settings["tn_target"]:
            carbon2=optimize_external_carbon(
                inf,an0,ae0,final_cfg,params,settings["tn_target"],settings["carbon"],base_result=base
            )
            carbon=carbon2
            inf_c=add_external_cod(inf,carbon["dose"]["added_cod_mg_l"])
            final_cfg,final=solve_was_for_target_srt(inf_c,an0,ae0,final_cfg,params,target_srt,max_iter=4)

    # Ensure carbon result carries the final coupled simulation.
    carbon["result"]=final
    carbon["target_met"]=final["final"]["tn_eff"]<=settings["tn_target"]

    naoh=calculate_naoh(raw.q,final["final"]["alk_eff"],settings["naoh"])
    phos=phosphorus_chemical(raw.q,settings["phos"])
    sludge=total_sludge(final["final"]["was_ds_kg_d"],phos["chemical_sludge_ds_kg_d"],settings["phos"].wet_sludge_ts_pct)

    return {
        "fractionation":fx,
        "base":base,
        "final":final,
        "carbon":carbon,
        "naoh":naoh,
        "phosphorus":phos,
        "sludge":sludge,
        "config":asdict(final_cfg),
        "settings":{
            "tn_target":settings["tn_target"],
            "target_srt":target_srt,
            "carbon":asdict(settings["carbon"]),
            "naoh":asdict(settings["naoh"]),
            "phos":asdict(settings["phos"]),
        }
    }


def show_results(case,lang):
    base=case["base"]["final"]
    final=case["final"]["final"]
    carbon=case["carbon"]["dose"]
    naoh=case["naoh"]
    p=case["phosphorus"]
    sl=case["sludge"]

    st.markdown(f'<div class="section-title">{tr(lang,"bio_result")}</div>',unsafe_allow_html=True)
    cols=st.columns(6)
    vals=[
        ("COD",final["cod_eff"],"mg/L"),
        ("NH₄-N",final["nh4_eff"],"mgN/L"),
        ("NOₓ-N",final["nox_eff"],"mgN/L"),
        ("TN",final["tn_eff"],"mgN/L"),
        ("DO",final["do_eff"],"mg/L"),
        ("SRT",final["srt_d"],"d"),
    ]
    for c,(label,v,u) in zip(cols,vals):
        with c: metric_card(label,fmtv(v,2),u)

    a,b,c,d=st.columns(4)
    with a: metric_card("MLSS Anoxic",fmtv(final["mlss_anoxic"],0),"mg/L")
    with b: metric_card("MLSS Aerobic",fmtv(final["mlss_aerobic"],0),"mg/L")
    with c: metric_card(tr(lang,"solver"),tr(lang,"pass") if final["solver_ok"] else tr(lang,"check"))
    with d: metric_card(tr(lang,"steady"),tr(lang,"pass") if final["steady_ok"] else tr(lang,"check"))

    st.markdown(f'<div class="section-title">{tr(lang,"chemical_result")}</div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1:
        st.markdown(f'<div class="chem-card"><b>{tr(lang,"carbon")}: {carbon["source"]}</b></div>',unsafe_allow_html=True)
        st.metric(tr(lang,"base_tn"),f'{base["tn_eff"]:.2f} mgN/L')
        st.metric(tr(lang,"final_tn"),f'{final["tn_eff"]:.2f} mgN/L')
        st.metric(tr(lang,"carbon_cod"),f'{carbon["added_cod_mg_l"]:.1f} mgCOD/L')
        st.metric(tr(lang,"carbon_l_d"),f'{carbon["solution_l_d"]:.1f} L/d')
        st.metric(tr(lang,"carbon_l_h"),f'{carbon["solution_l_h"]:.2f} L/h')
        if case["carbon"]["target_met"]:
            st.success(tr(lang,"no_carbon_needed") if carbon["added_cod_mg_l"]<=1e-9 else tr(lang,"pass"))
        else:
            st.warning(tr(lang,"carbon_target_fail"))

    with c2:
        st.markdown(f'<div class="chem-card"><b>{tr(lang,"naoh")}</b></div>',unsafe_allow_html=True)
        st.metric("Alkalinity final",f'{final["alk_eff"]:.1f} mg/L as CaCO₃')
        st.metric("Alkalinity deficit",f'{naoh["alkalinity_deficit_mg_l_as_caco3"]:.1f} mg/L as CaCO₃')
        st.metric(tr(lang,"naoh_l_d"),f'{naoh["solution_l_d"]:.1f} L/d')
        st.metric(tr(lang,"naoh_l_h"),f'{naoh["solution_l_h"]:.2f} L/h')
        st.caption("NaOH is calculated from ASM1 alkalinity deficit; direct dynamic pH prediction is not included.")

    with c3:
        st.markdown(f'<div class="chem-card"><b>{tr(lang,"pchemical")}: {p["chemical"]}</b></div>',unsafe_allow_html=True)
        st.metric(tr(lang,"p_before"),f'{p["tp_after_bio_mg_l"]:.2f} mgP/L')
        st.metric(tr(lang,"p_remove"),f'{p["p_remove_mg_l"]:.2f} mgP/L')
        st.metric(tr(lang,"p_dose_mgl"),f'{p["product_dose_mg_l"]:.1f} mg/L')
        st.metric(tr(lang,"p_dose"),f'{p["product_kg_d"]:.1f} kg/d')
        if p["product_l_d"] is not None:
            st.caption(f'{p["product_l_d"]:.1f} L/d · {p["product_l_h"]:.2f} L/h')
        st.caption(p["basis"])

    st.markdown(f'<div class="section-title">{tr(lang,"sludge")}</div>',unsafe_allow_html=True)
    a,b,c,d,e=st.columns(5)
    with a: metric_card(tr(lang,"bio_sludge"),fmtv(sl["biological_was_ds_kg_d"],1),"kgDS/d")
    with b: metric_card(tr(lang,"chem_sludge"),fmtv(sl["chemical_sludge_ds_kg_d"],1),"kgDS/d")
    with c: metric_card(tr(lang,"total_sludge"),fmtv(sl["total_ds_kg_d"],1),"kgDS/d")
    with d: metric_card(tr(lang,"wet_sludge"),fmtv(sl["wet_sludge_m3_d"],2),"m³/d")
    with e: metric_card("WAS flow",fmtv(case["config"]["was_flow"],2),"m³/d")
    st.caption("Chemical sludge is added as a separate dry-solids calculation; it is not yet fed back into ASM1/clarifier solids hydraulics.")

    st.markdown(f'<div class="section-title">{tr(lang,"trends")}</div>',unsafe_allow_html=True)
    df=pd.DataFrame(case["final"]["time_series"])
    tab1,tab2,tab3=st.tabs(["N", "COD / DO", "MLSS / SRT"])
    with tab1:
        fig=go.Figure()
        for y,name in [("nh4_eff","NH4-N"),("nox_eff","NOx-N"),("tn_eff","TN")]:
            fig.add_trace(go.Scatter(x=df.time_d,y=df[y],name=name))
        fig.update_layout(height=400,xaxis_title="Time (d)",yaxis_title="mgN/L",margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig,use_container_width=True)
    with tab2:
        fig=go.Figure()
        for y,name in [("cod_eff","COD"),("bod5_eff","BOD5"),("do_eff","DO")]:
            fig.add_trace(go.Scatter(x=df.time_d,y=df[y],name=name))
        fig.update_layout(height=400,xaxis_title="Time (d)",yaxis_title="mg/L",margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig,use_container_width=True)
    with tab3:
        fig=go.Figure()
        for y,name in [("mlss_anoxic","MLSS Anoxic"),("mlss_aerobic","MLSS Aerobic")]:
            fig.add_trace(go.Scatter(x=df.time_d,y=df[y],name=name))
        fig.update_layout(height=350,xaxis_title="Time (d)",yaxis_title="mg/L",margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig,use_container_width=True)
        fig2=go.Figure(go.Scatter(x=df.time_d,y=df.srt_d,name="SRT"))
        fig2.update_layout(height=280,xaxis_title="Time (d)",yaxis_title="d",margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig2,use_container_width=True)

    c1,c2=st.columns(2)
    with c1:
        st.download_button(tr(lang,"download_csv"),df.to_csv(index=False).encode("utf-8"),"asm1_v2_timeseries.csv","text/csv",use_container_width=True)
    with c2:
        export={
            "version":APP_VERSION,
            "final":case["final"]["final"],
            "carbon":case["carbon"]["dose"],
            "naoh":case["naoh"],
            "phosphorus":case["phosphorus"],
            "sludge":case["sludge"],
            "config":case["config"],
            "fractionation_qa":case["fractionation"]["qa"],
        }
        st.download_button(tr(lang,"download_json"),json.dumps(export,ensure_ascii=False,indent=2).encode("utf-8"),"asm1_v2_result.json","application/json",use_container_width=True)

    with st.expander(tr(lang,"advanced")):
        st.write("Fractionation QA",case["fractionation"]["qa"])
        state_df=pd.DataFrame({
            "State":STATE_NAMES,
            "Anoxic":[case["final"]["states"]["anoxic_final"][s] for s in STATE_NAMES],
            "Aerobic":[case["final"]["states"]["aerobic_final"][s] for s in STATE_NAMES],
            "Effluent":[case["final"]["states"]["effluent_final"][s] for s in STATE_NAMES],
        })
        st.dataframe(state_df,use_container_width=True,hide_index=True)


# Header and language
h1,h2=st.columns([1.1,4.2])
with h1:
    st.markdown(
        f'<div class="ncev-header"><img class="ncev-logo" src="{LOGO_URL}" onerror="this.style.display=\'none\'"><span class="ncev-wordmark">NCEV</span></div>',
        unsafe_allow_html=True,
    )
with h2:
    lang_label=st.selectbox("Language / Ngôn ngữ / 语言",list(LANG_OPTIONS.keys()),index=0,key="lang_select")
lang=LANG_OPTIONS[lang_label]
st.markdown(f'<div class="ncev-title">{tr(lang,"app_title")}</div><div class="ncev-sub">{tr(lang,"subtitle")} · {APP_VERSION}</div>',unsafe_allow_html=True)
st.caption(tr(lang,"brand_tag"))

with st.sidebar:
    st.markdown(f"### {tr(lang,'advanced')}")
    st.caption("ASM1 / fractionation calibration")
    params=ASM1Parameters()
    frac=FractionationAssumptions()
    params.mu_H=num("μH (1/d)",params.mu_H,0.01,0.1,key="muH")
    params.mu_A=num("μA (1/d)",params.mu_A,0.01,0.05,key="muA")
    params.Y_H=num("YH",params.Y_H,0.01,0.01,key="YH")
    params.Y_A=num("YA",params.Y_A,0.01,0.01,key="YA")
    frac.bod5_to_bcod=num("BOD5 / bCOD",frac.bod5_to_bcod,0.05,0.05,key="fbod")
    frac.readily_biodegradable_fraction=num("S_S / bCOD",frac.readily_biodegradable_fraction,0,0.05,key="fss")
    frac.soluble_inert_fraction=num("S_I / inert COD",frac.soluble_inert_fraction,0,0.05,key="fsi")
    st.warning(tr(lang,"warning_model"))
    st.caption("NaOH calculation currently excludes additional alkalinity consumption from PAC/Polytetsu hydrolysis.")

main_tab,compare_tab=st.tabs([tr(lang,"run"),tr(lang,"compare")])

with main_tab:
    raw,init,cfg,settings=make_inputs(lang)
    if st.button("▶ "+tr(lang,"run"),type="primary",use_container_width=True,key="run_main"):
        try:
            with st.spinner("ASM1 / BDF ..."):
                st.session_state["case_v2"]=run_pipeline(raw,init,cfg,settings,params,frac)
        except Exception as exc:
            st.error(str(exc))
    if "case_v2" in st.session_state:
        show_results(st.session_state["case_v2"],lang)

with compare_tab:
    if "case_v2" not in st.session_state:
        st.info("Run the base case first.")
    else:
        st.caption("Quick scenario: change internal recycle, aerobic volume, WAS/SRT, or KLa, then compare final KPIs.")
        base_case=st.session_state["case_v2"]
        bcfg=ProcessConfig(**base_case["config"])
        c1,c2,c3,c4=st.columns(4)
        with c1: sc_ir=num(tr(lang,"ir"),bcfg.internal_recycle_ratio*100,0,10,key="sc_ir")
        with c2: sc_vae=num(tr(lang,"v_ae"),bcfg.v_aerobic,1,10,key="sc_vae")
        with c3: sc_kla=num(tr(lang,"kla"),bcfg.kla_aerobic,0,5,key="sc_kla")
        with c4: sc_was=num(tr(lang,"was"),bcfg.was_flow,0,1,key="sc_was")
        if st.button("Run scenario",use_container_width=True,key="run_scenario"):
            try:
                # Reuse the saved base raw data and current chemical assumptions.
                raw2=InfluentInput(**base_case["fractionation"]["input"])
                init2=InitialConditions(**base_case["fractionation"]["initial_conditions"])
                cfg2=ProcessConfig(**base_case["config"])
                cfg2.internal_recycle_ratio=sc_ir/100
                cfg2.v_aerobic=sc_vae
                cfg2.kla_aerobic=sc_kla
                cfg2.was_flow=sc_was
                # Compare with Qw-input mode for deterministic quick comparison.
                settings2=deepcopy(base_case["settings"])
                cs=CarbonSettings(**settings2["carbon"])
                ns=NaOHSettings(**settings2["naoh"])
                ps=PhosphorusSettings(**settings2["phos"])
                settings_obj={"target_srt":None,"carbon":cs,"tn_target":settings2["tn_target"],"naoh":ns,"phos":ps}
                st.session_state["scenario_v2"]=run_pipeline(raw2,init2,cfg2,settings_obj,params,frac)
            except Exception as exc:
                st.error(str(exc))
        if "scenario_v2" in st.session_state:
            B=base_case["final"]["final"]
            S=st.session_state["scenario_v2"]["final"]["final"]
            rows=[]
            for label,key,unit in [("COD","cod_eff","mg/L"),("NH4-N","nh4_eff","mgN/L"),("NOx-N","nox_eff","mgN/L"),("TN","tn_eff","mgN/L"),("SRT","srt_d","d"),("MLSS aerobic","mlss_aerobic","mg/L")]:
                rows.append([label,B[key],S[key],S[key]-B[key],unit])
            st.dataframe(pd.DataFrame(rows,columns=["KPI","Base","Scenario","Δ","Unit"]),use_container_width=True,hide_index=True)

st.divider()
st.caption("NCEV Cloud v2 · ASM1 + SciPy BDF · Engineering screening / scenario analysis. Calibrate against plant data before design guarantees.")
