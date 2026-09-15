
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Dict, Any, Optional, Tuple
import numpy as np

from asm1_engine import ASM1Parameters, ProcessConfig, simulate

CARBON_COD_EQ = {
    "Methanol": 1.50,   # kg COD equivalent / kg pure chemical
    "Ethanol": 2.087,
}

@dataclass
class CarbonSettings:
    enabled: bool = True
    source: str = "Methanol"
    purity_pct: float = 25.0
    density_kg_l: float = 0.90
    max_added_cod_mg_l: float = 1000.0
    search_iterations: int = 8

@dataclass
class NaOHSettings:
    concentration_pct: float = 10.0
    density_kg_l: float = 1.11
    target_alkalinity_mg_l_as_caco3: float = 80.0

@dataclass
class PhosphorusSettings:
    tp_in_mg_l: float = 7.0
    biological_removal_pct: float = 15.0
    tp_target_mg_l: float = 1.0
    chemical: str = "PAC"  # PAC or Polytetsu
    pac_al2o3_pct: float = 31.0
    pac_al_to_p_molar: float = 1.5
    polytetsu_fe_pct: float = 21.0
    polytetsu_fe_to_p_molar: float = 1.0
    product_density_kg_l: float = 0.0  # optional; 0 = unknown
    chemical_sludge_factor: float = 1.0
    wet_sludge_ts_pct: float = 1.0

def _copy_cfg(cfg:ProcessConfig)->ProcessConfig:
    return ProcessConfig(**asdict(cfg))

def solve_was_for_target_srt(
    influent:np.ndarray, anoxic0:np.ndarray, aerobic0:np.ndarray,
    cfg:ProcessConfig, p:ASM1Parameters, target_srt_d:float,
    max_iter:int=7
)->Tuple[ProcessConfig,Dict[str,Any]]:
    if target_srt_d<=0:
        raise ValueError("Target SRT must be > 0.")
    work=_copy_cfg(cfg)
    work.was_flow=max(0.001, min(work.was_flow, 0.25*work.q))
    result=None
    for _ in range(max_iter):
        result=simulate(influent,anoxic0,aerobic0,work,p)
        f=result["final"]
        inv=f["particulate_inventory_g"]
        E=f["eff_particulate_cod_mg_l"]
        R=f["ras_particulate_cod_mg_l"]
        denom=R-E
        if denom<=1e-12:
            break
        qcalc=(inv/target_srt_d - work.q*E)/denom
        qcalc=max(0.001,min(float(qcalc),0.30*work.q))
        if abs(qcalc-work.was_flow)/max(work.was_flow,1e-6)<0.02:
            work.was_flow=qcalc
            result=simulate(influent,anoxic0,aerobic0,work,p)
            break
        work.was_flow=0.5*work.was_flow+0.5*qcalc
    if result is None:
        result=simulate(influent,anoxic0,aerobic0,work,p)
    return work,result

def add_external_cod(influent:np.ndarray, added_cod_mg_l:float)->np.ndarray:
    x=np.asarray(influent,float).copy()
    x[1]+=max(0.0,float(added_cod_mg_l))  # added COD treated as readily biodegradable S_S
    return x

def carbon_product_dose(
    q_m3_d:float, added_cod_mg_l:float, settings:CarbonSettings
)->Dict[str,float]:
    source=settings.source if settings.source in CARBON_COD_EQ else "Methanol"
    cod_eq=CARBON_COD_EQ[source]
    cod_kg_d=q_m3_d*added_cod_mg_l/1000.0
    pure_kg_d=cod_kg_d/cod_eq if cod_eq>0 else 0.0
    purity=max(settings.purity_pct/100.0,1e-9)
    solution_kg_d=pure_kg_d/purity
    density=max(settings.density_kg_l,1e-9)
    solution_l_d=solution_kg_d/density
    return {
        "source":source,
        "added_cod_mg_l":float(added_cod_mg_l),
        "cod_kg_d":float(cod_kg_d),
        "pure_chemical_kg_d":float(pure_kg_d),
        "solution_kg_d":float(solution_kg_d),
        "solution_l_d":float(solution_l_d),
        "solution_l_h":float(solution_l_d/24.0),
        "cod_equivalent_kgcod_per_kg":float(cod_eq),
    }

def optimize_external_carbon(
    influent:np.ndarray, anoxic0:np.ndarray, aerobic0:np.ndarray,
    cfg:ProcessConfig, p:ASM1Parameters, tn_target_mg_l:float,
    settings:CarbonSettings, base_result:Optional[Dict[str,Any]]=None
)->Dict[str,Any]:
    if base_result is None:
        base_result=simulate(influent,anoxic0,aerobic0,cfg,p)

    base_tn=base_result["final"]["tn_eff"]
    if (not settings.enabled) or base_tn<=tn_target_mg_l:
        dose=carbon_product_dose(cfg.q,0.0,settings)
        return {"dose":dose,"result":base_result,"target_met":base_tn<=tn_target_mg_l,"base_tn":base_tn}

    # If ammonium alone is already above the TN target, carbon cannot solve it.
    if base_result["final"]["nh4_eff"]>=tn_target_mg_l:
        dose=carbon_product_dose(cfg.q,0.0,settings)
        return {
            "dose":dose,"result":base_result,"target_met":False,"base_tn":base_tn,
            "warning":"TN target is below final NH4-N; external carbon cannot solve nitrification limitation."
        }

    # Build a bracket. Start from denitrification stoichiometry (~2.86 COD/NOx-N) with margin.
    initial=max(10.0, 2.86*max(base_tn-tn_target_mg_l,0.0)*1.25)
    low=0.0
    high=min(initial,settings.max_added_cod_mg_l)

    # Use final base reactor states as optimization initial guess to accelerate settling to the new steady state.
    opt_an=np.array([base_result["states"]["anoxic_final"][k] for k in
                     ["S_I","S_S","X_I","X_S","X_BH","X_BA","X_P","S_O","S_NO","S_NH","S_ND","X_ND","S_ALK"]],float)
    opt_ae=np.array([base_result["states"]["aerobic_final"][k] for k in
                     ["S_I","S_S","X_I","X_S","X_BH","X_BA","X_P","S_O","S_NO","S_NH","S_ND","X_ND","S_ALK"]],float)

    fast_cfg=_copy_cfg(cfg)
    fast_cfg.simulation_days=min(max(20.0,cfg.simulation_days*0.5),35.0)
    fast_cfg.output_dt=max(0.5,cfg.output_dt)

    def run_at(delta):
        return simulate(add_external_cod(influent,delta),opt_an,opt_ae,fast_cfg,p)

    high_res=run_at(high)
    while high_res["final"]["tn_eff"]>tn_target_mg_l and high<settings.max_added_cod_mg_l-1e-9:
        low=high
        high=min(high*2.0,settings.max_added_cod_mg_l)
        high_res=run_at(high)
        if high>=settings.max_added_cod_mg_l:
            break

    if high_res["final"]["tn_eff"]>tn_target_mg_l:
        # Return max-dose case using normal horizon for transparency.
        final_inf=add_external_cod(influent,high)
        final_res=simulate(final_inf,anoxic0,aerobic0,cfg,p)
        return {
            "dose":carbon_product_dose(cfg.q,high,settings),
            "result":final_res,"target_met":False,"base_tn":base_tn,
            "warning":"Maximum external carbon limit reached before TN target."
        }

    # Bisection
    best=high
    for _ in range(max(3,int(settings.search_iterations))):
        mid=0.5*(low+high)
        r=run_at(mid)
        if r["final"]["tn_eff"]<=tn_target_mg_l:
            best=mid
            high=mid
        else:
            low=mid

    # Final result uses original initial condition and full simulation horizon.
    final_inf=add_external_cod(influent,best)
    final_res=simulate(final_inf,anoxic0,aerobic0,cfg,p)
    # Small safety correction if the full-horizon result is just above target.
    if final_res["final"]["tn_eff"]>tn_target_mg_l and best<settings.max_added_cod_mg_l:
        best=min(best*1.05+1.0,settings.max_added_cod_mg_l)
        final_inf=add_external_cod(influent,best)
        final_res=simulate(final_inf,anoxic0,aerobic0,cfg,p)

    return {
        "dose":carbon_product_dose(cfg.q,best,settings),
        "result":final_res,
        "target_met":final_res["final"]["tn_eff"]<=tn_target_mg_l,
        "base_tn":base_tn,
    }

def calculate_naoh(q_m3_d:float, final_alk_mg_l_as_caco3:float, s:NaOHSettings)->Dict[str,float]:
    deficit=max(0.0,s.target_alkalinity_mg_l_as_caco3-final_alk_mg_l_as_caco3)
    pure_mg_l=deficit*40.0/50.0  # equivalent-weight conversion CaCO3 -> NaOH
    pure_kg_d=q_m3_d*pure_mg_l/1000.0
    frac=max(s.concentration_pct/100.0,1e-9)
    sol_kg_d=pure_kg_d/frac
    density=max(s.density_kg_l,1e-9)
    sol_l_d=sol_kg_d/density
    return {
        "alkalinity_deficit_mg_l_as_caco3":float(deficit),
        "pure_naoh_mg_l":float(pure_mg_l),
        "pure_naoh_kg_d":float(pure_kg_d),
        "solution_kg_d":float(sol_kg_d),
        "solution_l_d":float(sol_l_d),
        "solution_l_h":float(sol_l_d/24.0),
        "target_alkalinity_mg_l_as_caco3":float(s.target_alkalinity_mg_l_as_caco3),
    }

def phosphorus_chemical(q_m3_d:float, s:PhosphorusSettings)->Dict[str,Any]:
    tp_after_bio=s.tp_in_mg_l*(1.0-s.biological_removal_pct/100.0)
    remove_mg_l=max(0.0,tp_after_bio-s.tp_target_mg_l)
    p_kg_d=q_m3_d*remove_mg_l/1000.0

    if s.chemical=="Polytetsu":
        fe_per_p=s.polytetsu_fe_to_p_molar*56.0/31.0
        product_fraction=max(s.polytetsu_fe_pct/100.0,1e-9)
        product_kg_per_kgp=fe_per_p/product_fraction
        product_kg_d=p_kg_d*product_kg_per_kgp
        # FePO4: (56 + 31 + 64) / 31 kg dry precipitate per kg P
        theoretical_sludge_kg_per_kgp=(56.0+31.0+64.0)/31.0
        basis="FePO4, Fe:P = 1:1 mol"
    else:
        # PAC basis using supplied molecular masses Al2O3=102, Al2=54, P=31
        al_fraction=(s.pac_al2o3_pct/100.0)*(54.0/102.0)
        al_per_p=s.pac_al_to_p_molar*27.0/31.0
        product_kg_per_kgp=al_per_p/max(al_fraction,1e-9)
        product_kg_d=p_kg_d*product_kg_per_kgp
        # 1 mol P -> 1 mol AlPO4 + excess (Al/P-1) mol Al(OH)3.
        excess=max(0.0,s.pac_al_to_p_molar-1.0)
        theoretical_sludge_kg_per_kgp=((27.0+31.0+64.0)+excess*(27.0+51.0))/31.0
        basis=f"Al:P = {s.pac_al_to_p_molar:g}:1 mol"

    chem_sludge_ds=p_kg_d*theoretical_sludge_kg_per_kgp*s.chemical_sludge_factor
    vol_l_d=None
    vol_l_h=None
    if s.product_density_kg_l>0:
        vol_l_d=product_kg_d/s.product_density_kg_l
        vol_l_h=vol_l_d/24.0

    return {
        "chemical":s.chemical,
        "tp_in_mg_l":float(s.tp_in_mg_l),
        "biological_removal_pct":float(s.biological_removal_pct),
        "tp_after_bio_mg_l":float(tp_after_bio),
        "tp_target_mg_l":float(s.tp_target_mg_l),
        "p_remove_mg_l":float(remove_mg_l),
        "p_remove_kg_d":float(p_kg_d),
        "product_dose_mg_l":float(product_kg_per_kgp*remove_mg_l),
        "product_kg_per_kgp":float(product_kg_per_kgp),
        "product_kg_d":float(product_kg_d),
        "product_l_d":None if vol_l_d is None else float(vol_l_d),
        "product_l_h":None if vol_l_h is None else float(vol_l_h),
        "chemical_sludge_ds_kg_d":float(chem_sludge_ds),
        "theoretical_sludge_kg_per_kgp":float(theoretical_sludge_kg_per_kgp),
        "basis":basis,
    }

def total_sludge(bio_was_ds_kg_d:float, chem_sludge_ds_kg_d:float, wet_ts_pct:float)->Dict[str,float]:
    total=max(0.0,bio_was_ds_kg_d)+max(0.0,chem_sludge_ds_kg_d)
    ts=max(wet_ts_pct/100.0,1e-9)
    wet_volume=total/(1000.0*ts)
    return {
        "biological_was_ds_kg_d":float(bio_was_ds_kg_d),
        "chemical_sludge_ds_kg_d":float(chem_sludge_ds_kg_d),
        "total_ds_kg_d":float(total),
        "wet_ts_pct":float(wet_ts_pct),
        "wet_sludge_m3_d":float(wet_volume),
    }
