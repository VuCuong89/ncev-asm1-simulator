# Cloud v2.1 calculation assumptions

## External carbon

- Methanol COD equivalent: 1.50 kgCOD/kg pure methanol.
- Ethanol COD equivalent: 2.087 kgCOD/kg pure ethanol.
- Added carbon is represented as readily biodegradable substrate `S_S`.
- The app uses the ASM1 solver iteratively to search for the minimum external COD dose that reaches the TN target within the configured search range.

## NaOH

NaOH is based on alkalinity deficit:

`pure NaOH = alkalinity deficit (as CaCO3) × 40 / 50`

Default solution:
- NaOH = 10 wt%
- density = 1.11 kg/L

## PAC phosphorus removal

User-defined basis agreed for v2:
- PAC = 31 wt% Al2O3.
- Al content in Al2O3 = 54 / 102.
- Al:P = 1.5:1 mol.

Therefore theoretical PAC product demand is approximately:

`7.96 kg PAC / kg P removed`

Chemical-sludge approximation:
- 1 mol P forms 1 mol AlPO4.
- excess 0.5 mol Al forms Al(OH)3.
- theoretical dry precipitate ≈ 5.19 kgDS/kgP removed.

## Polytetsu phosphorus removal

User-defined basis agreed for v2:
- Fe = 21 wt%.
- Fe:P = 1:1 mol through FePO4.

Theoretical product demand:

`~8.60 kg Polytetsu / kg P removed`

Theoretical FePO4 dry precipitate:

`~4.87 kgDS/kgP removed`

## Phosphorus load

Defaults:
- Influent T-P = 7 mgP/L.
- Biological P removal = 15%.
- Chemical dose is calculated from T-P after assumed biological removal to the selected T-P target.

## Sludge

- Biological WAS DS = WAS flow × RAS TSS.
- Chemical P sludge is added separately.
- Total wet sludge volume assumes selected TS; default TS = 1%.
- Water density is approximated as 1000 kg/m3.
