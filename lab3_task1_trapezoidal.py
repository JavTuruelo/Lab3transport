"""Lab 3 - Task 1: A320 minimum-fuel cruise, trapezoidal collocation.
Run: python lab3_task1_trapezoidal.py
Change the flight conditions and output settings below directly.
Thrust and fuel flow follow the lab equations without cruise corrections.
"""

import csv
import json
import math
from pathlib import Path
import pyomo.environ as pyo

# Flight conditions
N_values = [50, 100, 150, 300]                 # Use [50, 100, 150, 300] for convergence
final_time = 4700.0             # s
final_distance = 1_000_000.0    # m

knots_to_ms = 1852.0 / 3600.0
initial_speed = 420.0 * knots_to_ms
initial_mass = 51000.0          # kg

h_ft = 38000.0                  # ft
h = h_ft * 0.3048               # m

# Output settings
final_speed_kn = None           # None leaves final speed free
show_solver = False
show_plots = True
make_plots = True               # Set False if Matplotlib is not installed
output_folder = Path(__file__).resolve().parent / "results_trapezoidal"

# 2. A320 coefficients (from the supplied A320 data file)
# Masses: kg; speeds: knots; altitudes: ft; wing area: m2.
aircraft = {
    'm_ref': 64000.0,
    'CM16': 0.0,
    'V_MO_kn': 350.0,
    'C_f1': 0.75882,
    'C_f2': 2938.5,
    'C_f3': 8.9418,
    'C_f4': 93865.0,
    'G_t': -313.6,
    'G_w': 0.4325,
    'CD0_CR': 0.026659,
    'h_MO_ft': 41000.0,
    'Clbo_M0': 1.4041,
    'Vstall_CR_kn': 140.5,
    'K': 0.79242,
    'h_max_ft': 33295.0,
    'S': 122.6,
    'm_max': 77000.0,
    'M_MO': 0.82,
    'C_tc4': 10.138,
    'C_tc5': 0.008871,
    'C_tc2': 51680.0,
    'C_tc3': 5.6809e-11,
    'C_tc1': 142310.0,
    'm_min': 39000.0,
    'CD2_CR': 0.038726,
    'm_pyld': 21500.0,
}

S = aircraft['S']
CD0 = aircraft['CD0_CR']
CD2 = aircraft['CD2_CR']
Cf1 = aircraft['C_f1']
Cf2 = aircraft['C_f2']
Cf3 = aircraft['C_f3']
Cf4 = aircraft['C_f4']
mass_min = aircraft['m_min']
mass_max = aircraft['m_max']
mass_ref = aircraft['m_ref']

# 3. ISA atmosphere at the fixed pressure altitude (including above tropopause)
g = 9.80665
R = 287.05287
gamma = 1.4

if not 0 <= h <= 20000:
    raise ValueError('This ISA implementation supports altitudes from 0 to 20 km.')

air_temperature = 288.15 - 0.0065 * min(h, 11000)
pressure = 101325 * (air_temperature / 288.15)**(g / (R * 0.0065))

if h > 11000:
    pressure *= math.exp(-g * (h - 11000) / (R * air_temperature))

rho = pressure / (R * air_temperature)
sound_speed = math.sqrt(gamma * R * air_temperature)
a0 = math.sqrt(gamma * R * 288.15)

maximum_altitude = min(aircraft['h_MO_ft'], aircraft['h_max_ft']
                       + aircraft['G_w'] * (mass_max - initial_mass))

if not mass_min <= initial_mass <= mass_max or h_ft > maximum_altitude:
    raise ValueError('Initial mass/altitude is outside the BADA ISA envelope.')

Tmax = aircraft['C_tc1'] * (1 - h_ft / aircraft['C_tc2'] + aircraft['C_tc3'] * h_ft**2)
minimum_fuel_flow = Cf3 / 60 * (1 - h_ft / Cf4)



# Only three shared physical formulas need helper functions.
# These work with both ordinary numbers and Pyomo symbolic variables.
def drag(V, mass):
    qS = 0.5 * rho * V**2 * S
    return qS * CD0 + CD2 * (mass * g)**2 / qS


def eta(V):
    # BADA uses knots; convert kg/(kN min) to kg/(N s).
    return Cf1 / 60000 * (1 + (V / knots_to_ms) / Cf2)


def calibrated_speed(V):
    # Compressible TAS -> CAS conversion, in m/s.
    impact_pressure = pressure * ((1 + 0.2 * (V / sound_speed)**2)**3.5 - 1)
    return a0 * (5 * ((1 + impact_pressure / 101325)**(2 / 7) - 1))**0.5


# 4. Build and solve one model for each requested mesh
solver = pyo.SolverFactory('ipopt')
if not solver.available(exception_flag=False):
    raise RuntimeError('IPOPT unavailable. Activate the Conda lab environment.')
solver.options['max_iter'] = 3000
solver.options['tol'] = 1e-8
solver.options['nlp_scaling_method'] = 'user-scaling'
output_folder.mkdir(parents=True, exist_ok=True)
summaries = []
profiles = []

for N in N_values:

    n = N - 1
    step = final_time / n
    model = pyo.ConcreteModel()
    model.N = pyo.RangeSet(0, n)
    model.Ns = pyo.RangeSet(0, n - 1)

    # Variables use SI units directly; no pyo.Expression components.
    model.x      = pyo.Var(model.N, bounds=(0, final_distance), initialize=0)
    model.V      = pyo.Var(model.N, bounds=(10, aircraft['M_MO'] * sound_speed), initialize=final_distance / final_time)
    model.mass   = pyo.Var(model.N, bounds=(mass_min, initial_mass), initialize=initial_mass)
    model.thrust = pyo.Var(model.N, bounds=(0, Tmax), initialize=min(Tmax, drag(final_distance / final_time, initial_mass)))
    
    for i in model.N:
        model.x[i].value = final_distance * i / n

    # 5. Differential constraints: trapezoidal integration
    def distance_rule(model, i):
        return model.x[i+1] - model.x[i] == step / 2 * (model.V[i] + model.V[i+1])

    model.distance_dynamics = pyo.Constraint(model.Ns, rule=distance_rule)

    def speed_rule(model, i):
        acceleration_i = (model.thrust[i] - drag(model.V[i], model.mass[i])) / model.mass[i]
        acceleration_next = (model.thrust[i+1] - drag(model.V[i+1], model.mass[i+1])) / model.mass[i+1]
        return model.V[i+1] - model.V[i] == step / 2 * (acceleration_i + acceleration_next)

    model.speed_dynamics = pyo.Constraint(model.Ns, rule=speed_rule)

    def mass_rule(model, i):
        fuel_i = model.thrust[i] * eta(model.V[i])
        fuel_next = model.thrust[i+1] * eta(model.V[i+1])
        return model.mass[i+1] - model.mass[i] == -step / 2 * (fuel_i + fuel_next)

    model.mass_dynamics = pyo.Constraint(model.Ns, rule=mass_rule)

    # 6. Path constraints (Mach and maximum thrust are variable bounds)
    def minimum_thrust_rule(model, i):
        return model.thrust[i] * eta(model.V[i]) >= minimum_fuel_flow

    model.minimum_thrust = pyo.Constraint(model.N, rule=minimum_thrust_rule)

    def maximum_speed_rule(model, i):
        return calibrated_speed(model.V[i]) / knots_to_ms <= aircraft['V_MO_kn']

    model.maximum_speed = pyo.Constraint(model.N, rule=maximum_speed_rule)

    def minimum_speed_rule(model, i):
        stall_limit = 1.3 * aircraft['Vstall_CR_kn'] * (model.mass[i] / mass_ref)**0.5
        return calibrated_speed(model.V[i]) / knots_to_ms >= stall_limit

    model.minimum_speed = pyo.Constraint(model.N, rule=minimum_speed_rule)

    def buffet_rule(model, i):
        mach = model.V[i] / sound_speed
        return aircraft['Clbo_M0'] * mach**2 - aircraft['K'] * mach**3 >= model.mass[i] * g / (0.583 * pressure * S)

    if h_ft >= 15000 and aircraft['Clbo_M0'] > 0 and aircraft['K'] > 0:
        model.buffet = pyo.Constraint(model.N, rule=buffet_rule)

    # 7. Boundary constraints: final speed and final mass are free by default
    def initial_distance_rule(model):
        return model.x[0] == 0

    model.initial_distance = pyo.Constraint(rule=initial_distance_rule)

    def final_distance_rule(model):
        return model.x[n] == final_distance

    model.final_distance = pyo.Constraint(rule=final_distance_rule)

    def initial_speed_rule(model):
        return model.V[0] == initial_speed

    model.initial_speed = pyo.Constraint(rule=initial_speed_rule)

    def initial_mass_rule(model):
        return model.mass[0] == initial_mass

    model.initial_mass = pyo.Constraint(rule=initial_mass_rule)

    def final_speed_rule(model):
        return model.V[n] == final_speed_kn * knots_to_ms

    if final_speed_kn is not None:
        model.final_speed = pyo.Constraint(rule=final_speed_rule)

    # 8. Objective: minimize fuel burned [kg]
    def objective_rule(model):
        return initial_mass - model.mass[n]

    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Numerical scaling keeps the SI formulas above readable.
    model.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    for variable, factor in [(model.x, 1e-6), (model.V, 1e-2), (model.mass, 1e-4), (model.thrust, 1e-4)]:

        for i in model.N:

            model.scaling_factor[variable[i]] = factor

    for constraint in model.component_data_objects(pyo.Constraint):
        model.scaling_factor[constraint] = 1.0

    for i in model.Ns:

        model.scaling_factor[model.distance_dynamics[i]] = 1e-6
        model.scaling_factor[model.speed_dynamics[i]] = 1e-2
        model.scaling_factor[model.mass_dynamics[i]] = 1e-4

    model.scaling_factor[model.initial_distance] = 1e-6
    model.scaling_factor[model.final_distance] = 1e-6
    model.scaling_factor[model.initial_speed] = 1e-2
    model.scaling_factor[model.initial_mass] = 1e-4

    if final_speed_kn is not None:

        model.scaling_factor[model.final_speed] = 1e-2

    model.scaling_factor[model.obj] = 1e-4

    results = solver.solve(model, tee=show_solver)
    pyo.assert_optimal_termination(results)

    # 9. Extract results and save the trajectory
    time        = [i * step for i in model.N]
    distance    = [pyo.value(model.x[i]) for i in model.N]
    speed       = [pyo.value(model.V[i]) for i in model.N]
    mass        = [pyo.value(model.mass[i]) for i in model.N]
    thrust      = [pyo.value(model.thrust[i]) for i in model.N]
    fuel_flow   = [thrust[i] * eta(speed[i]) for i in model.N]
    fuel_burned = pyo.value(model.obj)

    summary = {'nodes': N, 'fuel_kg': fuel_burned,
               'final_speed_kn': speed[-1] / knots_to_ms,
               'termination': str(results.solver.termination_condition)}
    
    # Residuals in original physical units, not IPOPT's scaled units.
    for name, component in [('distance_m', model.distance_dynamics),
                             ('speed_m_s', model.speed_dynamics),
                             ('mass_kg', model.mass_dynamics)]:
        
        summary['max_defect_' + name] = max(abs(pyo.value(c.body - c.lower)) for c in component.values())

    summaries.append(summary)
    profiles.append((N, time, distance, speed, mass, thrust))
    print(json.dumps(summary, indent=2))

    with (output_folder / f'trajectory_N{N}.csv').open('w', newline='') as file:

        writer = csv.writer(file)
        writer.writerow(['time_s', 'distance_m', 'TAS_m_s', 'mass_kg', 'thrust_N', 'fuel_flow_kg_s'])
        writer.writerows(zip(time, distance, speed, mass, thrust, fuel_flow))

settings = {'nodes': N_values, 'tf_s': final_time, 'distance_m': final_distance,
            'altitude_ft': h_ft, 'initial_mass_kg': initial_mass,
            'initial_speed_m_s': initial_speed, 'final_speed_kn': final_speed_kn}

(output_folder / 'summary.json').write_text(json.dumps({'settings': settings, 'results': summaries}, indent=2), encoding='utf-8')

# 10. Plot profiles and mesh convergence; NumPy is not needed.
if make_plots:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for N, time, distance, speed, mass, thrust in profiles:
        axes[0, 0].plot(time, [x / 1000 for x in distance], label=f'{N} nodes')
        axes[0, 1].plot(time, [v / knots_to_ms for v in speed], label=f'{N} nodes')
        axes[1, 0].plot(time, mass, label=f'{N} nodes')
        axes[1, 1].plot(time, [t / 1000 for t in thrust], label=f'{N} nodes')
    for ax, label in zip(axes.flat, ['Distance [km]', 'TAS [knots]', 'Mass [kg]', 'Thrust [kN]']):
        ax.set_ylabel(label)
        ax.grid(True)
        ax.legend()
    for ax in axes[-1]:
        ax.set_xlabel('Time [s]')
    fig.tight_layout()
    fig.savefig(output_folder / 'profiles.png', dpi=180)
    if len(summaries) > 1:
        fig2, ax = plt.subplots()
        ax.plot([s['nodes'] for s in summaries], [s['fuel_kg'] for s in summaries], 'o-')
        ax.set(xlabel='Number of nodes', ylabel='Fuel burned [kg]')
        ax.grid(True)
        fig2.tight_layout()
        fig2.savefig(output_folder / 'fuel_convergence.png', dpi=180)
    if show_plots:
        plt.show()
    plt.close('all')
