# NCEV AO ASM1 Simulator — Cloud MVP v1

A browser-based AO wastewater process simulator built with Streamlit + Python + SciPy.

## End-user workflow

1. Open the deployed URL in Chrome.
2. Enter influent and process data.
3. Click **Run simulation**.
4. Review KPI results and dynamic trends.
5. Use **Compare scenarios** to assess changes in recycle, volume, WAS and KLa.

End users do **not** need Python or any local installation after cloud deployment.

## Model core

- ASM1: 13 state variables
- 8 biological processes
- Peterson/Gujer matrix
- Anoxic CSTR
- Aerobic CSTR
- Internal recycle
- RAS / WAS
- Simple particulate clarifier
- SciPy `solve_ivp(method="BDF")`

## Current limitations

- Simple clarifier, not a Takács multilayer model
- No automatic temperature correction
- Influent fractionation is an engineering starting assumption
- Requires calibration/validation against real plant data before design guarantees

See `DEPLOY_TO_STREAMLIT_CLOUD.md` for deployment steps.
