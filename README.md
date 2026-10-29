# Lab3transport

Aircraft trajectory optimization laboratory for the Air Transport course of the MSc in Aeronautical Engineering at Universidad Carlos III de Madrid.

The project models an Airbus A320 using the supplied BADA 3.11 performance data and solves optimal-control problems with **Python, Pyomo and IPOPT**.

## Tasks

| Task | Script | Description |
| --- | --- | --- |
| 1a | `lab3_task1_trapezoidal.py` | Minimum-fuel 1D cruise using trapezoidal collocation. |
| 1b | `lab3_task1_hermite_simpson.py` | The same problem using Hermite-Simpson collocation. |
| 2 | `lab3_task2_cost_index.py` | Free arrival time, Cost Index study and boundary-condition sensitivities. |
| 3 | `lab3_task3_horizontal.py` | Horizontal trajectory on a spherical Earth, great-circle comparison and wind sensitivities. |
| 4 | `lab3_task4_vertical.py` | Continuous vertical climb, cruise and descent with Cost Index, mass and altitude-ceiling sensitivities. |

Task 1 covers **1,000 km at FL380**, starting at **420 kt TAS** and **51,000 kg**, with a fixed arrival time of **4,700 s**. Both methods run meshes of **50, 100, 150 and 300 endpoint nodes**.

Tasks 2-4 use Hermite-Simpson collocation with linear control interpolation. Their default mesh has 100 endpoint nodes and 99 additional midpoint stages. Task 1's Hermite-Simpson implementation instead optimizes midpoint thrust independently.

## Environment setup

From the repository folder, create and activate a local Conda environment:

```powershell
conda create --prefix .\.conda -c conda-forge python=3.11 pyomo ipopt matplotlib -y
conda activate .\.conda
```

If the environment already exists:

```powershell
conda activate .\.conda
conda install -c conda-forge pyomo ipopt matplotlib -y
```

Check that Pyomo can find IPOPT:

```powershell
python -c "from pyomo.environ import SolverFactory; print(SolverFactory('ipopt').available())"
```

The result should be `True`.

## Run the scripts

Run each task separately from the activated environment:

```powershell
python lab3_task1_trapezoidal.py
python lab3_task1_hermite_simpson.py
python lab3_task2_cost_index.py
python lab3_task3_horizontal.py
python lab3_task4_vertical.py
```

Settings are ordinary Python variables near the top of each script. Edit them directly to change node counts, flight conditions, sensitivity cases and plotting options. The scripts do not use command-line flags.

- `show_solver`: display IPOPT iteration logs.
- `make_plots`: generate figures; set to `False` to run without Matplotlib.
- `show_plots`: open figures interactively. Close the Task 1 plot windows to finish those scripts, or set this option to `False`.
- `output_folder`: choose the results directory.

The supplied A320 coefficients are hardcoded in each script. The TXT file is retained as a reference and is not required at runtime.

## Results and supporting material

Results are saved under:

```text
results_trapezoidal/
results_hermite_simpson/
results_task2/
results_task3/
results_task4/
```

These folders contain trajectory CSV files, JSON summaries and PNG figures. Rerunning a script replaces its corresponding output files. Summaries include solver or feasibility information; the Task 1 summaries also report dynamics defects in physical units.

The repository also includes the lab statement, lecture PDFs, BADA manual, A320 reference data and the original optimal-control exercise examples.

See [the implementation and results guide](lab3_tasks2_3_4_guide.md) for the assumptions, sensitivity cases, interpretation of the results and verified default result tables for Tasks 2-4.

## Modeling assumptions and interpretation

- State equations use SI units. BADA formulas receive the required conversions to knots and feet.
- Thrust and fuel flow follow the lab equations without the additional BADA cruise thrust or fuel correction factors.
- Task 2 uses `CI = [0, 10, 50, 200, 999]`. The default cost is the literal `CI * tf + fuel_kg / 100`, with time in seconds. This weighting strongly favors minimum time for nonzero CI. The statement's hour/pound CI units require an interpretation; the alternative hour-based time term is explicitly selectable and documented in the guide. These scores should not be presented as reconciled dollar costs.
- Task 3 uses approximate Madrid-Barajas and Paris-CDG horizontal endpoints at FL380, with a common fixed arrival time of approximately 5,001 s. Heading follows the statement's convention: zero is east, increasing toward north.
- Task 4 uses a 1,000 km clean-configuration segment starting and ending at 10,000 ft, 300 kt TAS and zero flight-path angle. These boundary conditions, lift-coefficient bounds and flight-path limits are explicit editable assumptions. It does not model runway operations or flap/gear transitions.
- Successful IPOPT termination establishes a numerical local solution. It does not prove global optimality, between-stage feasibility or mesh independence. Task 4's minimum-fuel descent retains flight-path-angle oscillations on the default mesh; refine the mesh before interpreting those features as physical conclusions.

Local Conda environments, package caches, Python caches, temporary files and editor settings are excluded through `.gitignore`.
