
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any
import numpy as np

@dataclass
class InfluentInput:
    q: float = 500.0
    cod: float = 450.0
    bod5: float = 250.0
    tss: float = 220.0
    nh4: float = 35.0
    tn: float = 50.0
    nox: float = 0.5
    alkalinity: float = 250.0
    dissolved_oxygen: float = 0.5
    temperature: float = 25.0
    tp: float = 7.0

@dataclass
class FractionationAssumptions:
    bod5_to_bcod: float = 0.75
    readily_biodegradable_fraction: float = 0.25
    soluble_inert_fraction: float = 0.25
    soluble_org_n_fraction: float = 0.40
    cod_to_ss: float = 0.75
    tss_tolerance: float = 0.30

@dataclass
class InitialConditions:
    mlss_anoxic: float = 3000.0
    mlss_aerobic: float = 3000.0
    do_anoxic: float = 0.2
    do_aerobic: float = 2.0
    nh4: float = 5.0
    nox: float = 10.0
    f_xi: float = 0.20
    f_xs: float = 0.04
    f_xbh: float = 0.62
    f_xba: float = 0.04
    f_xp: float = 0.10
    xnd_per_particulate_cod: float = 0.002
    initial_ss: float = 5.0
    initial_snd: float = 1.0

def fractionate(raw:InfluentInput, a:FractionationAssumptions, init:InitialConditions,
                i_xb:float=0.08, i_xp:float=0.06)->Dict[str,Any]:
    if raw.cod<=0 or raw.bod5<=0: raise ValueError("COD and BOD5 must be > 0.")
    if raw.bod5>raw.cod: raise ValueError("BOD5 should not exceed COD.")
    if raw.tn<raw.nh4+raw.nox: raise ValueError("TN must be >= NH4-N + NOx-N.")
    if raw.tss<0 or raw.alkalinity<0 or raw.tp<0: raise ValueError("TSS, alkalinity and TP cannot be negative.")

    bcod=min(raw.cod, raw.bod5/a.bod5_to_bcod)
    inert=max(0.0,raw.cod-bcod)
    s_i=inert*a.soluble_inert_fraction
    x_i=inert-s_i
    s_s=bcod*a.readily_biodegradable_fraction
    x_s=bcod-s_s
    x_bh=x_ba=x_p=0.0

    embedded_n=i_xb*(x_bh+x_ba)+i_xp*(x_i+x_p)
    free_org_n=raw.tn-raw.nox-raw.nh4-embedded_n
    if free_org_n < -1e-9:
        raise ValueError("TN is too low relative to NH4/NOx and organic-N assumptions.")
    free_org_n=max(0.0,free_org_n)
    s_nd=free_org_n*a.soluble_org_n_fraction
    x_nd=free_org_n-s_nd

    influent=np.array([s_i,s_s,x_i,x_s,x_bh,x_ba,x_p,raw.dissolved_oxygen,
                       raw.nox,raw.nh4,s_nd,x_nd,raw.alkalinity/50.0],float)

    pred_tss=a.cod_to_ss*(x_i+x_s+x_bh+x_ba+x_p)
    tss_ratio=pred_tss/raw.tss if raw.tss>0 else np.nan
    tss_ok=True if not np.isfinite(tss_ratio) else abs(tss_ratio-1)<=a.tss_tolerance

    fsum=init.f_xi+init.f_xs+init.f_xbh+init.f_xba+init.f_xp
    if abs(fsum-1)>1e-6: raise ValueError("Initial particulate fractions must sum to 1.0.")

    def initial_state(mlss,do):
        pcod=mlss/a.cod_to_ss
        return np.array([
            s_i,init.initial_ss,pcod*init.f_xi,pcod*init.f_xs,pcod*init.f_xbh,
            pcod*init.f_xba,pcod*init.f_xp,do,init.nox,init.nh4,init.initial_snd,
            pcod*init.xnd_per_particulate_cod,raw.alkalinity/50.0
        ],float)

    return {
        "influent_state":influent,
        "anoxic_initial":initial_state(init.mlss_anoxic,init.do_anoxic),
        "aerobic_initial":initial_state(init.mlss_aerobic,init.do_aerobic),
        "qa":{
            "bcod":float(bcod),"inert_cod":float(inert),"predicted_tss":float(pred_tss),
            "measured_tss":float(raw.tss),
            "tss_ratio":float(tss_ratio) if np.isfinite(tss_ratio) else None,
            "tss_ok":bool(tss_ok),"bod5_cod_ratio":float(raw.bod5/raw.cod),
        },
        "input":asdict(raw),"assumptions":asdict(a),"initial_conditions":asdict(init)
    }
