from asm1_engine import ASM1Parameters, ProcessConfig, simulate
from fractionation import InfluentInput, FractionationAssumptions, InitialConditions, fractionate
from chemistry import (
    CarbonSettings, NaOHSettings, PhosphorusSettings,
    optimize_external_carbon, calculate_naoh, phosphorus_chemical,
    total_sludge, solve_was_for_target_srt, add_external_cod
)

p=ASM1Parameters()
raw=InfluentInput()
frac=FractionationAssumptions()
init=InitialConditions()
fx=fractionate(raw,frac,init,i_xb=p.i_XB,i_xp=p.i_XP)
cfg=ProcessConfig(simulation_days=60,output_dt=0.5)

base=simulate(fx['influent_state'],fx['anoxic_initial'],fx['aerobic_initial'],cfg,p)
assert base['final']['solver_ok']

cs=CarbonSettings(source='Methanol',purity_pct=25,density_kg_l=0.90,search_iterations=6)
carb=optimize_external_carbon(fx['influent_state'],fx['anoxic_initial'],fx['aerobic_initial'],cfg,p,10.0,cs,base_result=base)
final=carb['result']
assert final['final']['solver_ok']
assert carb['dose']['solution_l_d'] >= 0

na=calculate_naoh(raw.q,final['final']['alk_eff'],NaOHSettings())
assert na['solution_l_d'] >= 0

for chem in ['PAC','Polytetsu']:
    ps=PhosphorusSettings(chemical=chem,tp_in_mg_l=7,biological_removal_pct=15,tp_target_mg_l=1,wet_sludge_ts_pct=1)
    ph=phosphorus_chemical(raw.q,ps)
    assert ph['product_kg_d'] > 0
    sludge=total_sludge(final['final']['was_ds_kg_d'],ph['chemical_sludge_ds_kg_d'],1)
    assert sludge['total_ds_kg_d'] > 0
    assert sludge['wet_sludge_m3_d'] > 0

cfg_srt,result_srt=solve_was_for_target_srt(fx['influent_state'],fx['anoxic_initial'],fx['aerobic_initial'],cfg,p,15,max_iter=5)
assert result_srt['final']['solver_ok']
assert cfg_srt.was_flow > 0

print('CLOUD V3 SMOKE TEST PASS')
print('base TN',round(base['final']['tn_eff'],3),'NH4',round(base['final']['nh4_eff'],3))
print('carbon COD mg/L',round(carb['dose']['added_cod_mg_l'],3),'solution L/d',round(carb['dose']['solution_l_d'],3),'final TN',round(final['final']['tn_eff'],3))
print('NaOH L/d',round(na['solution_l_d'],3))
print('target SRT WAS',round(cfg_srt.was_flow,3),'SRT',round(result_srt['final']['srt_d'],3))
