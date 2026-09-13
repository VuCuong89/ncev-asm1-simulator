
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple
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
    alkalinity: float = 250.0  # mg/L as CaCO3
    dissolved_oxygen: float = 0.5
    temperature: float = 25.0


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

    # COD fractions inside starting MLSS particulate COD
    f_xi: float = 0.20
    f_xs: float = 0.04
    f_xbh: float = 0.62
    f_xba: float = 0.04
    f_xp: float = 0.10
    xnd_per_particulate_cod: float = 0.002
    initial_ss: float = 5.0
    initial_snd: float = 1.0


def fractionate(
    raw: InfluentInput,
    assumptions: FractionationAssumptions,
    initial: InitialConditions,
    i_xb: float = 0.08,
    i_xp: float = 0.06,
) -> Dict[str, Any]:
    """
    Engineering 80/20 fractionation from ordinary wastewater analytics to ASM1 states.

    This is a starting assumption layer, not a substitute for wastewater-specific
    respirometry/fractionation measurements.
    """
    if raw.cod <= 0 or raw.bod5 <= 0:
        raise ValueError("COD and BOD5 must be > 0.")
    if raw.bod5 > raw.cod:
        raise ValueError("BOD5 should not exceed COD for normal input screening.")
    if raw.tn < raw.nh4 + raw.nox:
        raise ValueError("TN must be >= NH4-N + NOx-N.")
    if raw.tss < 0 or raw.alkalinity < 0:
        raise ValueError("TSS and alkalinity cannot be negative.")

    bcod = min(raw.cod, raw.bod5 / assumptions.bod5_to_bcod)
    inert_cod = max(0.0, raw.cod - bcod)

    s_i = inert_cod * assumptions.soluble_inert_fraction
    x_i = inert_cod - s_i
    s_s = bcod * assumptions.readily_biodegradable_fraction
    x_s = bcod - s_s

    # No active biomass assumed in raw influent by default.
    x_bh = 0.0
    x_ba = 0.0
    x_p = 0.0

    # Nitrogen embedded in particulate COD, using ASM1 composition assumptions.
    embedded_n = i_xb * (x_bh + x_ba) + i_xp * (x_i + x_p)
    free_org_n = raw.tn - raw.nox - raw.nh4 - embedded_n
    if free_org_n < -1e-9:
        raise ValueError(
            "Input TN is too low relative to NH4/NOx and particulate organic-N assumptions."
        )
    free_org_n = max(0.0, free_org_n)
    s_nd = free_org_n * assumptions.soluble_org_n_fraction
    x_nd = free_org_n - s_nd

    influent = np.array([
        s_i, s_s, x_i, x_s, x_bh, x_ba, x_p,
        raw.dissolved_oxygen, raw.nox, raw.nh4, s_nd, x_nd,
        raw.alkalinity / 50.0,
    ], dtype=float)

    predicted_tss = assumptions.cod_to_ss * (x_i + x_s + x_bh + x_ba + x_p)
    tss_ratio = predicted_tss / raw.tss if raw.tss > 0 else np.nan
    tss_ok = (
        True if not np.isfinite(tss_ratio)
        else abs(tss_ratio - 1.0) <= assumptions.tss_tolerance
    )

    reconstructed_tn = (
        raw.nox + raw.nh4 + s_nd + x_nd
        + i_xb * (x_bh + x_ba)
        + i_xp * (x_i + x_p)
    )

    # Initial reactor states from MLSS.
    fsum = initial.f_xi + initial.f_xs + initial.f_xbh + initial.f_xba + initial.f_xp
    if abs(fsum - 1.0) > 1e-6:
        raise ValueError("Initial particulate COD fractions must sum to 1.0.")

    def make_initial(mlss: float, do: float) -> np.ndarray:
        particulate_cod = mlss / assumptions.cod_to_ss
        return np.array([
            s_i,
            initial.initial_ss,
            particulate_cod * initial.f_xi,
            particulate_cod * initial.f_xs,
            particulate_cod * initial.f_xbh,
            particulate_cod * initial.f_xba,
            particulate_cod * initial.f_xp,
            do,
            initial.nox,
            initial.nh4,
            initial.initial_snd,
            particulate_cod * initial.xnd_per_particulate_cod,
            raw.alkalinity / 50.0,
        ], dtype=float)

    anoxic0 = make_initial(initial.mlss_anoxic, initial.do_anoxic)
    aerobic0 = make_initial(initial.mlss_aerobic, initial.do_aerobic)

    return {
        "influent_state": influent,
        "anoxic_initial": anoxic0,
        "aerobic_initial": aerobic0,
        "qa": {
            "bcod": float(bcod),
            "inert_cod": float(inert_cod),
            "predicted_tss": float(predicted_tss),
            "measured_tss": float(raw.tss),
            "tss_ratio": float(tss_ratio) if np.isfinite(tss_ratio) else None,
            "tss_ok": bool(tss_ok),
            "reconstructed_tn": float(reconstructed_tn),
            "tn_error": float(reconstructed_tn - raw.tn),
            "bod5_cod_ratio": float(raw.bod5 / raw.cod),
        },
        "input": asdict(raw),
        "assumptions": asdict(assumptions),
        "initial_conditions": asdict(initial),
    }
