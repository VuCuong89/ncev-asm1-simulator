# NCEV AO ASM1 Simulator — Cloud v2.1

Web simulator for AO wastewater treatment using ASM1 + SciPy BDF.

## Main additions in v2

- 3 languages: Vietnamese / English / Simplified Chinese.
- External carbon: Methanol or Ethanol.
- Automatic external-carbon optimization to meet TN target.
- External carbon is converted to readily biodegradable COD and fed back into ASM1.
- NaOH 10% default dosing calculated from final alkalinity deficit.
- Chemical phosphorus removal:
  - PAC: 31% Al2O3, 1.5 mol Al3+ / mol P.
  - Polytetsu: 21% Fe, 1.0 mol Fe3+ / mol P.
  - Default biological P removal = 15%.
  - Default influent T-P = 7 mgP/L.
- Daily sludge:
  - biological WAS dry solids,
  - chemical P sludge dry solids,
  - total dry solids,
  - wet sludge volume at default TS = 1%.
- Two sludge-control modes:
  - input WAS flow,
  - input target SRT and calculate WAS.
- Scenario comparison retained.
- NCEV-inspired blue/cyan branding and logo reference.

## Core model

- ASM1: 13 states, 8 processes.
- Anoxic CSTR + Aerobic CSTR.
- Internal recycle + RAS + WAS.
- Simple secondary clarifier.
- `scipy.integrate.solve_ivp(method="BDF")`.

## Files

- `app.py` — Streamlit UI and orchestration.
- `asm1_engine.py` — ASM1 dynamic engine.
- `fractionation.py` — ordinary wastewater inputs -> ASM1 states.
- `chemistry.py` — carbon / NaOH / P / sludge / target-SRT calculations.
- `i18n.py` — VI / EN / Simplified Chinese translations.
- `requirements.txt` — cloud dependencies.
- `.streamlit/config.toml` — Streamlit theme/runtime.
- `smoke_test.py` — deterministic model checks.

## Important engineering limitations

1. Influent ASM1 fractionation is a starting engineering assumption and needs calibration.
2. Methanol/Ethanol are represented as COD-equivalent `S_S`; source-specific kinetics are not yet separated.
3. NaOH is calculated from alkalinity deficit; the app does not calculate dynamic pH.
4. NaOH calculation currently does **not** include additional alkalinity consumption due to PAC/Polytetsu hydrolysis.
5. Biological phosphorus removal is an external assumption (default 15%); ASM1 itself does not model phosphorus.
6. Chemical sludge is calculated separately and is not fed back into ASM1/clarifier solids hydraulics.
7. The clarifier is a simple capture model, not a Takács multilayer settler.

Use for engineering screening, learning, scenario comparison and model development. Calibrate against plant data before design guarantees.


## v2.1 giao diện
- Giữ nguyên vị trí Target và Advanced của v2.
- Nền sáng, chữ tối được ép bằng CSS để không phụ thuộc theme trình duyệt.
- Sửa độ tương phản phần Tối ưu hóa hóa chất.
- Phóng lớn logo NCEV; bỏ chữ NCEV lặp trong tiêu đề và bỏ slogan bên dưới.
- Không thay đổi engine ASM1 / carbon / NaOH / P / sludge của v2.


## Cloud v2.6 UI strategy

- Pins Streamlit 1.63.0.
- Uses native Streamlit Light Theme for number inputs, select boxes, buttons and download controls.
- Uses `showWidgetBorder=true` + `borderColor` for even widget borders.
- Uses white `secondaryBackgroundColor` so input/selection regions stay light.
- Custom CSS is intentionally limited to logo/layout, KPI cards, scenario table and hiding secondary Cloud toolbar actions.
- Do not re-add CSS overrides for BaseWeb number/select internals unless a specific Streamlit version is tested.
