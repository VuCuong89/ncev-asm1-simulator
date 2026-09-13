
from asm1_engine import ASM1Parameters, ProcessConfig, simulate
from fractionation import InfluentInput, FractionationAssumptions, InitialConditions, fractionate

raw = InfluentInput()
frac = FractionationAssumptions()
init = InitialConditions()
p = ASM1Parameters()
cfg = ProcessConfig()

fx = fractionate(raw, frac, init, i_xb=p.i_XB, i_xp=p.i_XP)
result = simulate(
    fx["influent_state"],
    fx["anoxic_initial"],
    fx["aerobic_initial"],
    cfg,
    p
)

f = result["final"]
assert f["solver_ok"]
assert f["cod_eff"] >= 0
assert f["nh4_eff"] >= 0
assert f["tn_eff"] >= 0
print("SMOKE TEST PASS")
print({k: round(v, 4) if isinstance(v, float) else v for k, v in f.items() if k in [
    "cod_eff","nh4_eff","nox_eff","tn_eff","do_eff","srt_d","mlss_aerobic","solver_ok","steady_ok"
]})
