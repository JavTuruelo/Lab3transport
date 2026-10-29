"""Task 2: free-time 1D cruise and Cost Index sensitivity.
Run: python lab3_task2_cost_index.py
Settings are below. DOC follows the printed formula: CI * time + fuel_kg/100.
Time is in seconds by default. Set cost_time_unit='hours' to explore the
alternative hour interpretation of the CI units; these are distinct models.
"""
import csv
import json
import math
from pathlib import Path
import pyomo.environ as pyo

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

knots_to_ms = 1852.0 / 3600.0
g = 9.80665
R = 287.05287
gamma_air = 1.4
S = aircraft['S']
a0 = math.sqrt(gamma_air * R * 288.15)

# BADA formulas, without the optional cruise fuel/thrust corrections.
def temperature(h):
    # Continuous ISA through the 11 km tropopause (smooth transition over ~2 m).
    below_tropopause = (h + 11000 - pyo.sqrt((h - 11000)**2 + 1.0)) / 2
    return 288.15 - 0.0065 * below_tropopause


def air_pressure(h):
    lower_h = (h + 11000 - pyo.sqrt((h - 11000)**2 + 1.0)) / 2
    temp = 288.15 - 0.0065 * lower_h
    return 101325 * (temp / 288.15)**(g / (R * 0.0065)) * pyo.exp(-g * (h - lower_h) / (R * 216.65))


def eta(V):
    return aircraft['C_f1'] / 60000 * (1 + V / knots_to_ms / aircraft['C_f2'])


def maximum_thrust(h):
    altitude_ft = h / 0.3048
    return aircraft['C_tc1'] * (1 - altitude_ft / aircraft['C_tc2'] + aircraft['C_tc3'] * altitude_ft**2)


def minimum_fuel(h):
    return aircraft['C_f3'] / 60 * (1 - h / 0.3048 / aircraft['C_f4'])


def calibrated_speed(V, h):
    sound_squared = gamma_air * R * temperature(h)
    impact = air_pressure(h) * ((1 + 0.2 * V**2 / sound_squared)**3.5 - 1)
    return a0 * pyo.sqrt(5 * ((1 + impact / 101325)**(2 / 7) - 1))


# 1. Editable settings
N = 100  # endpoint nodes; midpoint stages are additional
CI_values = [0, 10, 50, 200, 999]
cost_time_unit = 'seconds'  # literal lab formula with SI state equations
cases = [
    {'name': 'baseline', 'mass': 51000.0, 'final_speed_kn': None},
    {'name': 'heavier', 'mass': 55000.0, 'final_speed_kn': None},
    {'name': 'fixed_final_speed', 'mass': 51000.0, 'final_speed_kn': 420.0},
]
distance = 1_000_000.0
h = 38000 * 0.3048
V0 = 420 * knots_to_ms
time_bounds = (2500.0, 10000.0)  # numerical bounds; reported if active
show_solver = False
make_plots = True
show_plots = False
output_folder = Path(__file__).resolve().parent / 'results_task2'
if cost_time_unit not in ('seconds', 'hours'):
    raise ValueError('cost_time_unit must be seconds or hours')
cost_time_divisor = 3600.0 if cost_time_unit == 'hours' else 1.0
rho = pyo.value(air_pressure(h) / (R * temperature(h)))
sound = math.sqrt(gamma_air * R * pyo.value(temperature(h)))


def drag(V, mass):
    qS = 0.5 * rho * V**2 * S
    return qS * aircraft['CD0_CR'] + aircraft['CD2_CR'] * (mass * g)**2 / qS


def rates(model, i):
    return (model.V[i],
            (model.thrust[i]-drag(model.V[i],model.mass[i]))/model.mass[i],
            -model.thrust[i]*eta(model.V[i]))


solver = pyo.SolverFactory('ipopt')
if not solver.available(exception_flag=False):
    raise RuntimeError('Activate the Conda environment containing IPOPT.')
solver.options['max_iter'] = 5000
solver.options['tol'] = 1e-8
solver.options['nlp_scaling_method'] = 'user-scaling'
output_folder.mkdir(parents=True, exist_ok=True)
summaries = []
profiles = []

for case, CI in [(case, CI) for case in cases for CI in CI_values]:
    label = f"{case['name']}_CI{CI}"
    mass0 = case['mass']
    ceiling = min(aircraft['h_MO_ft'], aircraft['h_max_ft'] + aircraft['G_w'] * (aircraft['m_max'] - mass0))
    if h / 0.3048 > ceiling:
        raise ValueError(f'{label}: initial altitude exceeds the BADA ceiling.')
    n = 2 * (N - 1)  # endpoint and midpoint stages
    model = pyo.ConcreteModel()
    model.N = pyo.RangeSet(0, n)
    model.I = pyo.RangeSet(0, N-2)
    model.tf = pyo.Var(bounds=time_bounds, initialize=4700)
    model.x = pyo.Var(model.N, bounds=(0, distance), initialize=0)
    model.V = pyo.Var(model.N, bounds=(100, aircraft['M_MO'] * sound), initialize=215)
    model.mass = pyo.Var(model.N, bounds=(aircraft['m_min'], mass0), initialize=mass0)
    model.thrust = pyo.Var(model.N, bounds=(0, pyo.value(maximum_thrust(h))), initialize=pyo.value(drag(215, mass0)))
    for i in model.N:
        model.x[i].value = distance * i / n

    # Hermite-Simpson: even indices are endpoints; odd indices are midpoints.
    # N is the number of endpoint nodes, giving 2*N-1 total stages.
    def distance_dynamics_rule(model, i):
        left, middle, right = 2*i, 2*i+1, 2*i+2
        f_left = rates(model, left)[0]
        f_mid = rates(model, middle)[0]
        f_right = rates(model, right)[0]
        return model.x[right]-model.x[left] == model.tf/(3*n)*(f_left+4*f_mid+f_right)

    model.distance_dynamics = pyo.Constraint(model.I, rule=distance_dynamics_rule)

    def midpoint_x_rule(model, i):
        left, middle, right = 2*i, 2*i+1, 2*i+2
        f_left = rates(model, left)[0]
        f_right = rates(model, right)[0]
        return model.x[middle] == (model.x[left]+model.x[right])/2+model.tf/(4*n)*(f_left-f_right)

    model.midpoint_x = pyo.Constraint(model.I, rule=midpoint_x_rule)

    def speed_dynamics_rule(model, i):
        left, middle, right = 2*i, 2*i+1, 2*i+2
        f_left = rates(model, left)[1]
        f_mid = rates(model, middle)[1]
        f_right = rates(model, right)[1]
        return model.V[right]-model.V[left] == model.tf/(3*n)*(f_left+4*f_mid+f_right)

    model.speed_dynamics = pyo.Constraint(model.I, rule=speed_dynamics_rule)

    def midpoint_V_rule(model, i):
        left, middle, right = 2*i, 2*i+1, 2*i+2
        f_left = rates(model, left)[1]
        f_right = rates(model, right)[1]
        return model.V[middle] == (model.V[left]+model.V[right])/2+model.tf/(4*n)*(f_left-f_right)

    model.midpoint_V = pyo.Constraint(model.I, rule=midpoint_V_rule)

    def mass_dynamics_rule(model, i):
        left, middle, right = 2*i, 2*i+1, 2*i+2
        f_left = rates(model, left)[2]
        f_mid = rates(model, middle)[2]
        f_right = rates(model, right)[2]
        return model.mass[right]-model.mass[left] == model.tf/(3*n)*(f_left+4*f_mid+f_right)

    model.mass_dynamics = pyo.Constraint(model.I, rule=mass_dynamics_rule)

    def midpoint_mass_rule(model, i):
        left, middle, right = 2*i, 2*i+1, 2*i+2
        f_left = rates(model, left)[2]
        f_right = rates(model, right)[2]
        return model.mass[middle] == (model.mass[left]+model.mass[right])/2+model.tf/(4*n)*(f_left-f_right)

    model.midpoint_mass = pyo.Constraint(model.I, rule=midpoint_mass_rule)

    def midpoint_thrust_rule(model, i):
        return model.thrust[2*i+1] == (model.thrust[2*i]+model.thrust[2*i+2])/2

    model.midpoint_thrust = pyo.Constraint(model.I, rule=midpoint_thrust_rule)

    def minimum_thrust_rule(model, i):
        return model.thrust[i] * eta(model.V[i]) >= minimum_fuel(h)

    model.minimum_thrust = pyo.Constraint(model.N, rule=minimum_thrust_rule)

    def maximum_speed_rule(model, i):
        return calibrated_speed(model.V[i], h) / knots_to_ms <= aircraft['V_MO_kn']

    model.maximum_speed = pyo.Constraint(model.N, rule=maximum_speed_rule)

    def minimum_speed_rule(model, i):
        return calibrated_speed(model.V[i], h) / knots_to_ms >= 1.3 * aircraft['Vstall_CR_kn'] * pyo.sqrt(model.mass[i] / aircraft['m_ref'])

    model.minimum_speed = pyo.Constraint(model.N, rule=minimum_speed_rule)

    def buffet_rule(model, i):
        mach = model.V[i] / sound
        return aircraft['Clbo_M0'] * mach**2 - aircraft['K'] * mach**3 >= model.mass[i] * g / (0.583 * air_pressure(h) * S)

    model.buffet = pyo.Constraint(model.N, rule=buffet_rule)

    def initial_distance_rule(model):
        return model.x[0] == 0

    model.initial_distance = pyo.Constraint(rule=initial_distance_rule)

    def final_distance_rule(model):
        return model.x[n] == distance

    model.final_distance = pyo.Constraint(rule=final_distance_rule)

    def initial_speed_rule(model):
        return model.V[0] == V0

    model.initial_speed = pyo.Constraint(rule=initial_speed_rule)

    def initial_mass_rule(model):
        return model.mass[0] == mass0

    model.initial_mass = pyo.Constraint(rule=initial_mass_rule)

    def final_speed_rule(model):
        return model.V[n] == case['final_speed_kn'] * knots_to_ms

    if case['final_speed_kn'] is not None:
        model.final_speed = pyo.Constraint(rule=final_speed_rule)

    def objective_rule(model):
        return CI * model.tf / cost_time_divisor + (mass0 - model.mass[n]) / 100

    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
    model.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    model.scaling_factor[model.tf] = 1e-3
    for var, scale in [(model.x, 1e-6), (model.V, 1e-2), (model.mass, 1e-4), (model.thrust, 1e-4)]:
        for i in model.N:
            model.scaling_factor[var[i]] = scale
    for c in model.component_data_objects(pyo.Constraint):
        model.scaling_factor[c] = 1.0
    for c, scale in [(model.distance_dynamics, 1e-6), (model.speed_dynamics, 1e-2), (model.mass_dynamics, 1e-4)]:
        for i in model.I:
            model.scaling_factor[c[i]] = scale
    for c, scale in [(model.initial_distance, 1e-6), (model.final_distance, 1e-6), (model.initial_speed, 1e-2), (model.initial_mass, 1e-4)]:
        model.scaling_factor[c] = scale
    model.scaling_factor[model.obj] = 1 / max(1, CI * 4700 / cost_time_divisor)
    for i in model.I:
        model.scaling_factor[model.midpoint_x[i]] = 1e-06
    for i in model.I:
        model.scaling_factor[model.midpoint_V[i]] = 0.01
    for i in model.I:
        model.scaling_factor[model.midpoint_mass[i]] = 0.0001
    for i in model.I:
        model.scaling_factor[model.midpoint_thrust[i]] = 0.0001

    # Confirm solver termination and physical constraint residuals.
    result = solver.solve(model, tee=show_solver)
    pyo.assert_optimal_termination(result)
    worst_scaled_violation = 0.0
    for constraint in model.component_data_objects(pyo.Constraint, active=True):
        body = pyo.value(constraint.body)
        violation = 0.0
        if constraint.has_lb():
            violation = max(violation, pyo.value(constraint.lower) - body)
        if constraint.has_ub():
            violation = max(violation, body - pyo.value(constraint.upper))
        factor = model.scaling_factor.get(constraint, 1.0)
        worst_scaled_violation = max(worst_scaled_violation, factor * violation)
    if worst_scaled_violation > 1e-5:
        raise RuntimeError(f'Constraint violation too large: {worst_scaled_violation}')
    tf = pyo.value(model.tf)
    time = [tf * i / n for i in model.N]
    V = [pyo.value(model.V[i]) for i in model.N]
    mass = [pyo.value(model.mass[i]) for i in model.N]
    thrust = [pyo.value(model.thrust[i]) for i in model.N]
    summary = dict(case=case['name'], CI=CI, cost_time_unit=cost_time_unit,
                   time_s=tf, fuel_kg=mass0-mass[-1], DOC=pyo.value(model.obj),
                   time_bound_active=min(abs(tf-time_bounds[0]), abs(tf-time_bounds[1])) < 0.1,
                   max_scaled_violation=worst_scaled_violation)
    summary['nodes'] = N
    summary['stages'] = n + 1
    summary['termination'] = str(result.solver.termination_condition)
    summaries.append(summary)
    profiles.append((label, time, V, mass, thrust))
    with (output_folder / f'{label}.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['time_s', 'distance_m', 'TAS_m_s', 'mass_kg', 'thrust_N'])
        writer.writerows(zip(time, [pyo.value(model.x[i]) for i in model.N], V, mass, thrust))
    print(json.dumps(summary), flush=True)

(output_folder / 'summary.json').write_text(json.dumps(summaries, indent=2))
if make_plots:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for label, time, V, mass, thrust in profiles:
        if label.startswith('baseline_'):
            axes[0].plot(time, [v/knots_to_ms for v in V], label=label)
            axes[1].plot(time, mass, label=label)
            axes[2].plot(time, [t/1000 for t in thrust], label=label)
    for ax, ylabel in zip(axes, ['TAS [knots]', 'Mass [kg]', 'Thrust [kN]']):
        ax.set(xlabel='Time [s]', ylabel=ylabel)
        ax.grid(True)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_folder / 'profiles.png', dpi=180)
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
    for case in cases:
        rows = [s for s in summaries if s['case'] == case['name']]
        axes2[0].plot([s['CI'] for s in rows], [s['time_s'] for s in rows], 'o-', label=case['name'])
        axes2[1].plot([s['time_s'] for s in rows], [s['fuel_kg'] for s in rows], 'o-', label=case['name'])
    axes2[0].set(xlabel='Cost Index', ylabel='Flight time [s]')
    axes2[1].set(xlabel='Flight time [s]', ylabel='Fuel burned [kg]')
    for ax in axes2:
        ax.grid(True)
        ax.legend()
    fig2.tight_layout()
    fig2.savefig(output_folder / 'tradeoff.png', dpi=180)
    if show_plots:
        plt.show()
    plt.close('all')
