# Lab 3: Tasks 2, 3 and 4

Run these standalone scripts from the activated lab Conda environment:

```powershell
python lab3_task2_cost_index.py
python lab3_task3_horizontal.py
python lab3_task4_vertical.py
```

Dependencies: Pyomo, IPOPT and Matplotlib. No input TXT or shared Python module is required. Each script contains the supplied A320 coefficients and editable settings. Set `make_plots = False` to solve without Matplotlib, or `show_plots = True` to open the generated figures. Results are saved in `results_task2`, `results_task3` and `results_task4`.

## Numerical formulation

All three scripts use Hermite-Simpson collocation with linear interpolation of the controls between endpoint nodes. Even indices are endpoints; odd indices are midpoint stages. `N = 100` means 100 endpoints and 99 midpoints. Both Simpson integration and Hermite midpoint constraints are imposed for every state. Flight-envelope constraints are checked at all stages. Each constraint and objective has its own named rule; no `pyo.Expression` components are used.

All physical state equations use SI units. BADA speed and altitude arguments are converted to knots and feet where required. The 0.95 cruise thrust factor and the C_fcr fuel correction are omitted, following the lab's stated equations. Variable/constraint scaling is numerical only and does not alter the objective or physical equations. The ISA atmosphere uses a smooth minimum at the 11 km tropopause, with an approximately 2 m transition, so IPOPT can differentiate it.

IPOPT solves a nonconvex nonlinear program: successful termination establishes a numerical local solution, not a proof of global optimality. CSV files include all endpoint and midpoint stages. The summary includes a feasibility check with the exported constraint scaling; between-stage feasibility and mesh independence are separate questions. Increase `N` to check resolution before interpreting small differences in fuel or control profiles.

## Task 2: free-time cruise

The flight covers 1000 km at FL380, starting at 420 kt TAS and 51000 kg. Arrival time is optimized. Final speed is free in the baseline. The script runs CI = 0, 10, 50, 200 and 999, plus heavier initial mass (55000 kg) and fixed-final-speed (420 kt) sensitivity cases.

The printed objective is implemented as `CI * tf + fuel_kg / 100`, with `tf` in seconds by default. This is the literal algebraic convention used with the lab's SI dynamics; it is not a fully reconciled monetary cost because the stated CI units refer to hours and pounds. Consequently, all nonzero specified CI values put very high weight on time, and their trajectories nearly coincide at minimum flight time. This is a consequence of the given weighting, not an omitted CI loop.

`cost_time_unit = 'hours'` is provided as an explicit alternative interpretation, changing the time term to `CI * tf / 3600`. It changes the optimization problem; it is not an additional BADA correction or an automatic conversion to dollars. Do not mix results from the two conventions. Both retain the printed fuel term in kg/100.

The numerical time bounds are 2500--10000 s; active bounds are flagged in the summary. The default runs should be limited by aircraft performance rather than these bounds.

Expected interpretation: CI=0 selects a fuel-efficient cruise; positive CI increases speed and fuel consumption while reducing time. Increased mass raises induced drag. Requiring a final speed changes the terminal maneuver and fuel/time trade-off.

Outputs: trajectory CSV for each case/CI, `summary.json`, `profiles.png`, and `tradeoff.png`.

## Task 3: horizontal trajectory

The chosen endpoints approximate Madrid-Barajas (40.472 N, 3.561 W) and Paris-CDG (49.010 N, 2.548 E); change the coordinate settings for another pair. These are horizontal boundary locations at FL380, not a model of takeoff or landing. The spherical great-circle distance is about 1064.1 km. A common fixed arrival time of about 5001.3 s is used for a controlled fuel comparison. Initial TAS is 420 kt; final speed, initial/final heading and bank are free.

The heading convention follows the printed longitude/latitude equations: chi = 0 is east and chi = pi/2 is north. The given heading-rate equation is retained without adding a spherical transport term. Bank increases lift demand to mg/cos(bank), and therefore increases induced drag. The minimum-speed and buffet constraints account for bank load factor.

Cases include zero wind, eastward wind of 30 m/s, northward wind of 20 m/s, both wind components, heavier mass, and a tighter bank limit of 15 degrees instead of 35 degrees. Constant wind components refer to the local east/north frame.

The great-circle reference is generated analytically by spherical interpolation. The summary reports spherical route length and maximum cross-track distance to that great-circle plane. The zero-wind optimized path closely matches it; wind changes the air-relative trajectory and can make a small lateral detour fuel-optimal. Tailwind reduces the airspeed required to meet the fixed arrival time, lowering fuel consumption. The tighter bank limit can remain inactive on this gentle route; an unchanged result is a valid sensitivity outcome.

Outputs: trajectory CSV for every case, `great_circle.csv`, `summary.json`, and `horizontal_profiles.png`.

## Task 4: vertical trajectory

The five states are distance, altitude, TAS, flight-path angle and mass; controls are thrust and lift coefficient. All five differential equations in the lab are transcribed. Lift is a control-dependent quantity, not forced equal to weight during climb or descent.

The statement does not specify vertical boundary conditions, so these explicit defaults are used: 1000 km, start/end at 10000 ft with TAS 300 kt and zero flight-path angle, initial mass 51000 kg. The aircraft remains in the clean configuration. The altitude floor is the lower boundary altitude. It can climb, cruise continuously and descend without prescribed phase durations. This is a clean flight segment, not a runway-to-runway mission with flap/gear schedules.

The final time is free, using the same DOC convention and five CI values as Task 2. Sensitivities lower the altitude ceiling from 41000 to 35000 ft and increase initial mass to 55000 kg. The mass-dependent BADA ceiling is additionally enforced at every stage. Assumed control bounds CL = 0.05--1.5 and flight-path angle +/-10 degrees are editable modelling assumptions, not extracted A320 coefficients; active CL bounds are flagged. Time bounds are 2500--12000 s. Near the 15000 ft threshold, buffet-limit activation is smoothed over roughly 100 ft; the clean stall-speed constraint remains active at all altitudes.

With CI=0, reduced density and induced/parasite-drag balance favor a high cruise altitude in this particular mission. With strongly time-weighted positive CI, the aircraft favors the altitude near the CAS/Mach crossover, where the largest TAS is available, and uses more fuel to shorten the trip. Lowering the ceiling affects the fuel-optimal case more than the time-optimal cases if those already fly below 35000 ft. These are results for the stated boundary/control assumptions, not a universal rule for A320 operations.

The summary's mean cruise altitude/speed averages the central 20--80% of route distance at stages where abs(gamma) < 0.3 degrees; it is a diagnostic, not a separately imposed cruise phase.

Outputs: trajectory CSV for each case/CI, `summary.json`, one profile figure per sensitivity case, and `tradeoff.png`.

## Verified default results

All default cases below were solved with IPOPT at N=100 endpoint nodes (199 total stages). Solver termination was optimal for each case. These numbers are numerical local solutions for the settings above.

### Task 2: baseline Cost Index study

| CI | Time [s] | Fuel [kg] |
|---:|---:|---:|
| 0 | 4658.42 | 2316.25 |
| 10 | 4138.82 | 2414.16 |
| 50 | 4138.82 | 2414.16 |
| 200 | 4138.82 | 2414.16 |
| 999 | 4138.82 | 2414.16 |

### Task 3: route and wind study

| Case | Fuel [kg] | Max cross-track distance [km] |
|---|---:|---:|
| zero_wind | 2464.62 | 0.0011 |
| east_wind | 2346.35 | 2.7591 |
| north_wind | 2297.80 | 0.1743 |
| combined_wind | 2218.96 | 2.8843 |
| heavier | 2587.52 | 0.0014 |
| bank_limit | 2218.96 | 2.8843 |

### Task 4: baseline Cost Index study

| CI | Time [s] | Fuel [kg] |
|---:|---:|---:|
| 0 | 4974.31 | 2381.27 |
| 10 | 4018.22 | 3742.35 |
| 50 | 4018.22 | 3742.35 |
| 200 | 4018.22 | 3742.35 |
| 999 | 4018.22 | 3742.35 |


## Interpretation limits

The Task 4 CI=0 solution retains oscillations in flight-path angle near the terminal descent on the default mesh. The stated model has no control-rate or passenger-comfort objective. No smoothing penalty was added to the lab objective to hide those variations. A successful discrete solve does not establish a smooth, mesh-independent continuous trajectory: refine the mesh and inspect the angle/lift histories before treating those small-scale features as physical conclusions. The saved results and plots retain them explicitly.
