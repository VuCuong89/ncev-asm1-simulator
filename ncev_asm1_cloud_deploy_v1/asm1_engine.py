
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple
import numpy as np
from scipy.integrate import solve_ivp


STATE_NAMES = [
    "S_I", "S_S", "X_I", "X_S", "X_BH", "X_BA", "X_P",
    "S_O", "S_NO", "S_NH", "S_ND", "X_ND", "S_ALK"
]

PARTICULATE_IDX = np.array([2, 3, 4, 5, 6, 11], dtype=int)
PARTICULATE_COD_IDX = np.array([2, 3, 4, 5, 6], dtype=int)


@dataclass
class ASM1Parameters:
    # Stoichiometry
    Y_H: float = 0.67
    Y_A: float = 0.24
    f_p: float = 0.08
    i_XB: float = 0.08
    i_XP: float = 0.06

    # Kinetics
    mu_H: float = 4.0
    K_S: float = 10.0
    K_OH: float = 0.2
    K_NO: float = 0.5
    b_H: float = 0.3
    eta_g: float = 0.8
    eta_h: float = 0.8
    k_h: float = 3.0
    K_X: float = 0.1
    mu_A: float = 0.5
    K_NH: float = 1.0
    K_OA: float = 0.4
    b_A: float = 0.05
    k_a: float = 0.05

    # Reporting only
    cod_to_ss: float = 0.75
    bod5_factor: float = 0.25

    epsilon: float = 1e-9


@dataclass
class ProcessConfig:
    q: float = 500.0
    v_anoxic: float = 150.0
    v_aerobic: float = 350.0
    internal_recycle_ratio: float = 2.0
    ras_ratio: float = 0.5
    was_flow: float = 12.0
    clarifier_capture: float = 0.995
    kla_anoxic: float = 0.0
    kla_aerobic: float = 75.0
    do_saturation: float = 8.0
    simulation_days: float = 60.0
    output_dt: float = 0.1
    rtol: float = 1e-6
    atol: float = 1e-8


def gujer_matrix(p: ASM1Parameters) -> np.ndarray:
    """Return 8-process x 13-state Peterson/Gujer matrix."""
    N = np.zeros((8, 13), dtype=float)

    YH, YA, fp, iXB, iXP = p.Y_H, p.Y_A, p.f_p, p.i_XB, p.i_XP

    # R1 Aerobic heterotrophic growth
    N[0, 1] = -1 / YH
    N[0, 4] = 1
    N[0, 7] = -(1 - YH) / YH
    N[0, 9] = -iXB
    N[0, 12] = -iXB / 14

    # R2 Anoxic heterotrophic growth
    N[1, 1] = -1 / YH
    N[1, 4] = 1
    N[1, 8] = -(1 - YH) / (2.86 * YH)
    N[1, 9] = -iXB
    N[1, 12] = ((1 - YH) / (2.86 * YH) - iXB) / 14

    # R3 Aerobic autotrophic growth
    N[2, 5] = 1
    N[2, 7] = -(4.57 - YA) / YA
    N[2, 8] = 1 / YA
    N[2, 9] = -iXB - 1 / YA
    N[2, 12] = (-iXB - 2 / YA) / 14

    # R4 Heterotroph decay
    N[3, 3] = 1 - fp
    N[3, 4] = -1
    N[3, 6] = fp
    N[3, 11] = iXB - fp * iXP

    # R5 Autotroph decay
    N[4, 3] = 1 - fp
    N[4, 5] = -1
    N[4, 6] = fp
    N[4, 11] = iXB - fp * iXP

    # R6 Ammonification
    N[5, 9] = 1
    N[5, 10] = -1
    N[5, 12] = 1 / 14

    # R7 Hydrolysis - organics
    N[6, 1] = 1
    N[6, 3] = -1

    # R8 Hydrolysis - organic N
    N[7, 10] = 1
    N[7, 11] = -1

    return N


def process_rates(x: np.ndarray, p: ASM1Parameters) -> np.ndarray:
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    _, SS, _, XS, XBH, XBA, _, SO, SNO, SNH, SND, XND, _ = x
    e = p.epsilon

    r1 = p.mu_H * SS / (p.K_S + SS + e) * SO / (p.K_OH + SO + e) * XBH
    r2 = (
        p.mu_H
        * SS / (p.K_S + SS + e)
        * p.K_OH / (p.K_OH + SO + e)
        * SNO / (p.K_NO + SNO + e)
        * p.eta_g
        * XBH
    )
    r3 = p.mu_A * SNH / (p.K_NH + SNH + e) * SO / (p.K_OA + SO + e) * XBA
    r4 = p.b_H * XBH
    r5 = p.b_A * XBA
    r6 = p.k_a * SND * XBH
    r7 = (
        p.k_h
        * XS / (p.K_X * XBH + XS + e)
        * (
            SO / (p.K_OH + SO + e)
            + p.eta_h
            * p.K_OH / (p.K_OH + SO + e)
            * SNO / (p.K_NO + SNO + e)
        )
        * XBH
    )
    r8 = r7 * XND / (XS + e)
    return np.array([r1, r2, r3, r4, r5, r6, r7, r8], dtype=float)


def _validate(cfg: ProcessConfig, influent: np.ndarray, anoxic0: np.ndarray, aerobic0: np.ndarray) -> None:
    if cfg.q <= 0:
        raise ValueError("Influent flow Q must be > 0.")
    if cfg.v_anoxic <= 0 or cfg.v_aerobic <= 0:
        raise ValueError("Both bioreactor volumes must be > 0.")
    if cfg.was_flow < 0 or cfg.was_flow >= cfg.q:
        raise ValueError("WAS flow must satisfy 0 <= Qw < Q.")
    if not (0 < cfg.clarifier_capture <= 1):
        raise ValueError("Clarifier capture must be in (0, 1].")
    if cfg.output_dt <= 0 or cfg.simulation_days <= 0:
        raise ValueError("Simulation duration and output interval must be > 0.")
    for name, arr in [("influent", influent), ("anoxic0", anoxic0), ("aerobic0", aerobic0)]:
        if np.asarray(arr).shape != (13,):
            raise ValueError(f"{name} must contain 13 ASM1 state values.")
        if np.any(~np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values.")
        if np.any(np.asarray(arr) < 0):
            raise ValueError(f"{name} contains negative values.")


def simulate(
    influent: np.ndarray,
    anoxic0: np.ndarray,
    aerobic0: np.ndarray,
    cfg: ProcessConfig,
    p: ASM1Parameters,
) -> Dict[str, Any]:
    """
    Dynamic two-CSTR ASM1 simulation:
    influent -> anoxic -> aerobic -> simple clarifier
                       ^             |
                       |---- RAS ----|
              aerobic -- internal recycle --> anoxic
    """
    influent = np.asarray(influent, dtype=float)
    anoxic0 = np.asarray(anoxic0, dtype=float)
    aerobic0 = np.asarray(aerobic0, dtype=float)
    _validate(cfg, influent, anoxic0, aerobic0)

    N = gujer_matrix(p)
    Q = cfg.q
    Qi = cfg.internal_recycle_ratio * Q
    Qr = cfg.ras_ratio * Q
    Qw = cfg.was_flow
    Qbio = Q + Qi + Qr
    Qc = Q + Qr
    Qe = Q - Qw
    Qu = Qr + Qw
    cap = cfg.clarifier_capture

    def settler(aerobic: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        aerobic = np.maximum(aerobic, 0.0)
        eff = aerobic.copy()
        ras = aerobic.copy()
        eff[PARTICULATE_IDX] = (1 - cap) * aerobic[PARTICULATE_IDX]
        ras[PARTICULATE_IDX] = (
            Qc * aerobic[PARTICULATE_IDX] - Qe * eff[PARTICULATE_IDX]
        ) / max(Qu, p.epsilon)
        return np.maximum(eff, 0.0), np.maximum(ras, 0.0)

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        an = np.maximum(y[:13], 0.0)
        ae = np.maximum(y[13:], 0.0)
        _eff, ras = settler(ae)

        dan = (Q * influent + Qr * ras + Qi * ae - Qbio * an) / cfg.v_anoxic
        dan += process_rates(an, p) @ N

        dae = Qbio * (an - ae) / cfg.v_aerobic
        dae += process_rates(ae, p) @ N

        # Oxygen transfer
        dan[7] += cfg.kla_anoxic * (cfg.do_saturation - an[7])
        dae[7] += cfg.kla_aerobic * (cfg.do_saturation - ae[7])

        return np.r_[dan, dae]

    n_points = max(2, int(round(cfg.simulation_days / cfg.output_dt)) + 1)
    n_points = min(n_points, 5001)
    times = np.linspace(0.0, cfg.simulation_days, n_points)

    sol = solve_ivp(
        rhs,
        (0.0, cfg.simulation_days),
        np.r_[anoxic0, aerobic0],
        method="BDF",
        t_eval=times,
        rtol=cfg.rtol,
        atol=cfg.atol,
    )

    if sol.y.shape[1] == 0:
        raise RuntimeError("ODE solver returned no result.")

    an = np.maximum(sol.y[:13].T, 0.0)
    ae = np.maximum(sol.y[13:].T, 0.0)
    eff = np.array([settler(x)[0] for x in ae])
    ras = np.array([settler(x)[1] for x in ae])

    cod = eff[:, :7].sum(axis=1)
    bod5 = p.bod5_factor * (
        eff[:, 1] + eff[:, 3] + (1 - p.f_p) * (eff[:, 4] + eff[:, 5])
    )
    tss = p.cod_to_ss * (
        eff[:, 2] + eff[:, 3] + eff[:, 4] + eff[:, 5] + eff[:, 6]
    )
    tkn = (
        eff[:, 9] + eff[:, 10] + eff[:, 11]
        + p.i_XB * (eff[:, 4] + eff[:, 5])
        + p.i_XP * (eff[:, 6] + eff[:, 2])
    )
    tn = tkn + eff[:, 8]
    alkalinity = eff[:, 12] * 50.0

    inventory = (
        (an[:, PARTICULATE_COD_IDX] * cfg.v_anoxic).sum(axis=1)
        + (ae[:, PARTICULATE_COD_IDX] * cfg.v_aerobic).sum(axis=1)
    )
    loss = (
        Qe * eff[:, PARTICULATE_COD_IDX].sum(axis=1)
        + Qw * ras[:, PARTICULATE_COD_IDX].sum(axis=1)
    )
    srt = np.divide(
        inventory,
        loss,
        out=np.full_like(inventory, np.nan),
        where=loss > p.epsilon,
    )

    mlss_anoxic = p.cod_to_ss * (
        an[:, 2] + an[:, 3] + an[:, 4] + an[:, 5] + an[:, 6]
    )
    mlss_aerobic = p.cod_to_ss * (
        ae[:, 2] + ae[:, 3] + ae[:, 4] + ae[:, 5] + ae[:, 6]
    )

    final_dy = rhs(float(times[-1]), sol.y[:, -1])
    scale = np.maximum(np.abs(sol.y[:, -1]), 1.0)
    steady_index = float(np.max(np.abs(final_dy) / scale))
    steady_ok = bool(steady_index < 1e-3)

    time_series = {
        "time_d": times,
        "nh4_eff": eff[:, 9],
        "nox_eff": eff[:, 8],
        "tn_eff": tn,
        "do_eff": eff[:, 7],
        "cod_eff": cod,
        "bod5_eff": bod5,
        "tss_eff": tss,
        "alk_eff": alkalinity,
        "srt_d": srt,
        "mlss_anoxic": mlss_anoxic,
        "mlss_aerobic": mlss_aerobic,
    }

    final = {key: float(val[-1]) for key, val in time_series.items() if key != "time_d"}
    final["solver_ok"] = bool(sol.success)
    final["solver_message"] = str(sol.message)
    final["steady_ok"] = steady_ok
    final["steady_index_per_d"] = steady_index
    final["time_d"] = float(times[-1])

    return {
        "final": final,
        "time_series": time_series,
        "states": {
            "anoxic_final": dict(zip(STATE_NAMES, map(float, an[-1]))),
            "aerobic_final": dict(zip(STATE_NAMES, map(float, ae[-1]))),
            "effluent_final": dict(zip(STATE_NAMES, map(float, eff[-1]))),
        },
        "metadata": {
            "engine": "NCEV ASM1 MVP",
            "solver": "scipy.solve_ivp/BDF",
            "n_states": 13,
            "n_processes": 8,
            "config": asdict(cfg),
            "parameters": asdict(p),
        },
    }
