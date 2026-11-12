"""Task 2: free-time 1D cruise and Cost Index sensitivity.
Run: python lab3_task2_cost_index.py
Settings are below. DOC follows the printed formula: CI * time + fuel_kg/100.
Time is in seconds by default. Set cost_time_unit='hours' to explore the
alternative hour interpretation of the CI units; these are distinct models.
"""
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
def eta(V):
    return aircraft['C_f1'] / 60000 * (1 + V / knots_to_ms / aircraft['C_f2'])



# 1. Editable settings
N = 100  # endpoint nodes; midpoint stages are additional
CI_values = [0, 10, 50, 200, 999]
cost_time_unit = 'seconds'  # literal lab formula with SI state equations
cases = [
    {'name': 'baseline', 'mass': 51000.0, 'final_speed_kn': None},
    {'name': 'heavier', 'mass': 55000.0, 'final_speed_kn': None},
    {'name': 'fixed_final_speed', 'mass': 51000.0, 'final_speed_kn': 420.0},
    {'name': 'max_thrust_off', 'mass': 51000.0, 'final_speed_kn': None, 'enforce_max_thrust': False},
    {'name': 'min_thrust_off', 'mass': 51000.0, 'final_speed_kn': None, 'enforce_min_thrust': False},
    {'name': 'acceleration_limit_off', 'mass': 51000.0, 'final_speed_kn': None, 'enforce_acceleration_limit': False},
]
distance = 1_000_000.0
h_ft = 38000.0
h = h_ft * 0.3048
V0 = 420 * knots_to_ms
time_bounds = (2500.0, 10000.0)  # numerical bounds; reported if active
use_scaling = True
show_solver = False
make_plots = True
show_plots = False
output_folder = Path(__file__).resolve().parent / 'results_task2'
if cost_time_unit not in ('seconds', 'hours'):
    raise ValueError('cost_time_unit must be seconds or hours')
cost_time_divisor = 3600.0 if cost_time_unit == 'hours' else 1.0
if not 0 <= h <= 20000:
    raise ValueError('This ISA implementation supports altitudes from 0 to 20 km.')

T0 = 288.15
p0 = 101325
beta_T = -0.0065
h_trop = 11000
T_isa_trop = T0 + beta_T * h_trop  # BADA (3.1-14)

# Temperature for ISA (delta T = 0), BADA (3.1-13) and (3.1-16).
if h <= h_trop:
    air_temperature = T0 + beta_T * h
else:
    air_temperature = T_isa_trop

# Pressure, BADA (3.1-18) to (3.1-20).
if h <= h_trop:
    pressure = p0 * (air_temperature / T0)**(-g / (beta_T * R))
else:
    p_trop = p0 * (T_isa_trop / T0)**(-g / (beta_T * R))
    pressure = p_trop * math.exp(-g * (h - h_trop) / (R * T_isa_trop))

rho = pressure / (R * air_temperature)
sound_speed = math.sqrt(gamma_air * R * air_temperature)
rho0 = p0 / (R * T0)

Tmax = aircraft['C_tc1'] * (1 - h_ft / aircraft['C_tc2'] + aircraft['C_tc3'] * h_ft**2)
minimum_fuel_flow = aircraft['C_f3'] / 60 * (1 - h_ft / aircraft['C_f4'])
a_l_max = 2.0 * 0.3048  # BADA (5.2-1), converted from ft/s² to m/s².


def calibrated_speed(V_TAS):
    # BADA (3.1-24) and (3.1-25). Both speeds are in m/s.
    p = pressure
    mu = (gamma_air - 1) / gamma_air
    return (
        (2 * p0 / (mu * rho0))
        * ((1 + (p / p0) * ((1 + (mu * rho / (2 * p)) * V_TAS**2)**(1 / mu) - 1))**mu - 1)
    )**(1 / 2)


def drag(V, mass):
    qS = 0.5 * rho * V**2 * S
    return qS * aircraft['CD0_CR'] + aircraft['CD2_CR'] * (mass * g)**2 / qS


solver = pyo.SolverFactory('ipopt')
if not solver.available(exception_flag=False):
    raise RuntimeError('Activate the Conda environment containing IPOPT.')
solver.options['max_iter'] = 5000
solver.options['tol'] = 1e-8
solver.options['nlp_scaling_method'] = 'user-scaling' if use_scaling else 'none'
summaries = []
profiles = []

for case, CI in [(case, CI) for case in cases for CI in CI_values]:
    label = f"{case['name']}_CI{CI}"
    mass0 = case['mass']
    enforce_max_thrust = case.get('enforce_max_thrust', True)
    enforce_min_thrust = case.get('enforce_min_thrust', True)
    enforce_acceleration_limit = case.get('enforce_acceleration_limit', True)
    thrust_upper_bound = Tmax if enforce_max_thrust else None
    # BADA (3.5-1), with delta T = 0 for ISA.
    delta_T = 0.0
    ceiling = aircraft['h_MO_ft']
    if aircraft['h_max_ft'] != 0:
        temperature_correction = aircraft['G_t'] * max(0.0, delta_T - aircraft['C_tc4'])
        ceiling = min(
            ceiling,
            aircraft['h_max_ft'] + temperature_correction
            + aircraft['G_w'] * (aircraft['m_max'] - mass0),
        )
    if not aircraft['m_min'] <= mass0 <= aircraft['m_max'] or h_ft > ceiling:
        raise ValueError(f'{label}: initial mass or altitude is outside the BADA ISA envelope.')
    n = N - 1
    model = pyo.ConcreteModel()
    model.N = pyo.RangeSet(0, n)
    model.Ns = pyo.RangeSet(0, n - 1)
    model.tf = pyo.Var(bounds=time_bounds, initialize=4700)
    step = model.tf / n

    # Variables use SI units directly; no pyo.Expression components.
    model.x      = pyo.Var(model.N, bounds=(0, distance), initialize=0)
    model.V      = pyo.Var(model.N, bounds=(100, aircraft['M_MO'] * sound_speed), initialize=215)
    model.mass   = pyo.Var(model.N, bounds=(aircraft['m_min'], mass0), initialize=mass0)
    model.thrust = pyo.Var(model.N, bounds=(0, thrust_upper_bound), initialize=min(Tmax, drag(215, mass0)))

    for i in model.N:
        model.x[i].value = distance * i / n

    # 5. Hermite-Simpson: interpolate states and optimize midpoint thrust.
    model.thrust_mid = pyo.Var(model.Ns, bounds=(0, thrust_upper_bound), initialize=min(Tmax, drag(215, mass0)))

    def midpoint_distance(model, i):
        return (model.x[i] + model.x[i+1]) / 2 + step / 8 * (model.V[i] - model.V[i+1])

    def midpoint_speed(model, i):
        acceleration_i = (model.thrust[i] - drag(model.V[i], model.mass[i])) / model.mass[i]
        acceleration_next = (model.thrust[i+1] - drag(model.V[i+1], model.mass[i+1])) / model.mass[i+1]
        return (model.V[i] + model.V[i+1]) / 2 + step / 8 * (acceleration_i - acceleration_next)

    def midpoint_mass(model, i):
        fuel_i = model.thrust[i] * eta(model.V[i])
        fuel_next = model.thrust[i+1] * eta(model.V[i+1])
        return (model.mass[i] + model.mass[i+1]) / 2 + step / 8 * (-fuel_i + fuel_next)

    def distance_rule(model, i):
        return model.x[i+1] - model.x[i] == step / 6 * (model.V[i] + 4 * midpoint_speed(model, i) + model.V[i+1])

    model.distance_dynamics = pyo.Constraint(model.Ns, rule=distance_rule)

    def speed_rule(model, i):
        acceleration_i = (model.thrust[i] - drag(model.V[i], model.mass[i])) / model.mass[i]
        speed_mid = midpoint_speed(model, i)
        mass_mid = midpoint_mass(model, i)
        acceleration_mid = (model.thrust_mid[i] - drag(speed_mid, mass_mid)) / mass_mid
        acceleration_next = (model.thrust[i+1] - drag(model.V[i+1], model.mass[i+1])) / model.mass[i+1]
        return model.V[i+1] - model.V[i] == step / 6 * (acceleration_i + 4 * acceleration_mid + acceleration_next)

    model.speed_dynamics = pyo.Constraint(model.Ns, rule=speed_rule)

    def mass_rule(model, i):
        fuel_i = model.thrust[i] * eta(model.V[i])
        fuel_mid = model.thrust_mid[i] * eta(midpoint_speed(model, i))
        fuel_next = model.thrust[i+1] * eta(model.V[i+1])
        return model.mass[i+1] - model.mass[i] == -step / 6 * (fuel_i + 4 * fuel_mid + fuel_next)

    model.mass_dynamics = pyo.Constraint(model.Ns, rule=mass_rule)

    # 6. Path constraints (Mach and maximum thrust are variable bounds)
    # BADA (5.2-1): consecutive stages are half a time step apart.
    def longitudinal_acceleration_to_mid_rule(model, i):
        return midpoint_speed(model, i) - model.V[i] <= a_l_max * (step / 2)

    def longitudinal_acceleration_from_mid_rule(model, i):
        return model.V[i + 1] - midpoint_speed(model, i) <= a_l_max * (step / 2)

    if enforce_acceleration_limit:
        model.longitudinal_acceleration_to_mid = pyo.Constraint(model.Ns, rule=longitudinal_acceleration_to_mid_rule)
        model.longitudinal_acceleration_from_mid = pyo.Constraint(model.Ns, rule=longitudinal_acceleration_from_mid_rule)

    def minimum_thrust_rule(model, i):
        return model.thrust[i] * eta(model.V[i]) >= minimum_fuel_flow

    if enforce_min_thrust:
        model.minimum_thrust = pyo.Constraint(model.N, rule=minimum_thrust_rule)

    def maximum_speed_rule(model, i):
        return calibrated_speed(model.V[i]) / knots_to_ms <= aircraft['V_MO_kn']

    model.maximum_speed = pyo.Constraint(model.N, rule=maximum_speed_rule)

    def minimum_speed_rule(model, i):
        stall_limit = 1.3 * aircraft['Vstall_CR_kn'] * (model.mass[i] / aircraft['m_ref'])**0.5
        return calibrated_speed(model.V[i]) / knots_to_ms >= stall_limit

    model.minimum_speed = pyo.Constraint(model.N, rule=minimum_speed_rule)

    def buffet_rule(model, i):
        mach = model.V[i] / sound_speed
        return aircraft['Clbo_M0'] * mach**2 - aircraft['K'] * mach**3 >= model.mass[i] * g / (0.583 * pressure * S)

    if h_ft >= 15000 and aircraft['Clbo_M0'] > 0 and aircraft['K'] > 0:
        model.buffet = pyo.Constraint(model.N, rule=buffet_rule)

    # Enforce the same flight envelope at midpoint stages.
    def midpoint_distance_bounds_rule(model, i):
        return (0, midpoint_distance(model, i), distance)

    def midpoint_speed_bounds_rule(model, i):
        return (100, midpoint_speed(model, i), aircraft['M_MO'] * sound_speed)

    def midpoint_mass_bounds_rule(model, i):
        return (aircraft['m_min'], midpoint_mass(model, i), mass0)

    model.midpoint_distance_bounds = pyo.Constraint(model.Ns, rule=midpoint_distance_bounds_rule)
    model.midpoint_speed_bounds = pyo.Constraint(model.Ns, rule=midpoint_speed_bounds_rule)
    model.midpoint_mass_bounds = pyo.Constraint(model.Ns, rule=midpoint_mass_bounds_rule)

    def minimum_thrust_mid_rule(model, i):
        return model.thrust_mid[i] * eta(midpoint_speed(model, i)) >= minimum_fuel_flow

    if enforce_min_thrust:
        model.minimum_thrust_mid = pyo.Constraint(model.Ns, rule=minimum_thrust_mid_rule)

    def maximum_speed_mid_rule(model, i):
        return calibrated_speed(midpoint_speed(model, i)) / knots_to_ms <= aircraft['V_MO_kn']

    model.maximum_speed_mid = pyo.Constraint(model.Ns, rule=maximum_speed_mid_rule)

    def minimum_speed_mid_rule(model, i):
        stall_limit = 1.3 * aircraft['Vstall_CR_kn'] * (midpoint_mass(model, i) / aircraft['m_ref'])**0.5
        return calibrated_speed(midpoint_speed(model, i)) / knots_to_ms >= stall_limit

    model.minimum_speed_mid = pyo.Constraint(model.Ns, rule=minimum_speed_mid_rule)

    def buffet_mid_rule(model, i):
        mach = midpoint_speed(model, i) / sound_speed
        return aircraft['Clbo_M0'] * mach**2 - aircraft['K'] * mach**3 >= midpoint_mass(model, i) * g / (0.583 * pressure * S)

    if h_ft >= 15000 and aircraft['Clbo_M0'] > 0 and aircraft['K'] > 0:
        model.buffet_mid = pyo.Constraint(model.Ns, rule=buffet_mid_rule)

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
    if use_scaling:
        model.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        model.scaling_factor[model.tf] = 1e-3
        for i in model.Ns:
            model.scaling_factor[model.thrust_mid[i]] = 1e-4
        for var, scale in [(model.x, 1e-6), (model.V, 1e-2), (model.mass, 1e-4), (model.thrust, 1e-4)]:
            for i in var:
                model.scaling_factor[var[i]] = scale
        for c in model.component_data_objects(pyo.Constraint):
            model.scaling_factor[c] = 1.0
        for c, scale in [(model.distance_dynamics, 1e-6), (model.speed_dynamics, 1e-2), (model.mass_dynamics, 1e-4)]:
            for i in model.Ns:
                model.scaling_factor[c[i]] = scale
        for c, scale in [(model.initial_distance, 1e-6), (model.final_distance, 1e-6), (model.initial_speed, 1e-2), (model.initial_mass, 1e-4)]:
            model.scaling_factor[c] = scale
        model.scaling_factor[model.obj] = 1 / max(1, CI * 4700 / cost_time_divisor)
        for i in model.Ns:
            model.scaling_factor[model.midpoint_distance_bounds[i]] = 1e-06
        for i in model.Ns:
            model.scaling_factor[model.midpoint_speed_bounds[i]] = 0.01
        for i in model.Ns:
            model.scaling_factor[model.midpoint_mass_bounds[i]] = 0.0001

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
        factor = model.scaling_factor.get(constraint, 1.0) if use_scaling else 1.0
        worst_scaled_violation = max(worst_scaled_violation, factor * violation)
    if worst_scaled_violation > 1e-5:
        raise RuntimeError(f'Constraint violation too large: {worst_scaled_violation}')
    tf = pyo.value(model.tf)
    dt = tf / n
    samples = []
    for i in model.Ns:
        samples.append((i * dt, pyo.value(model.V[i]),
                        pyo.value(model.mass[i]), pyo.value(model.thrust[i])))
        samples.append(((i + 0.5) * dt, pyo.value(midpoint_speed(model, i)), pyo.value(midpoint_mass(model, i)),
                        pyo.value(model.thrust_mid[i])))
    samples.append((tf, pyo.value(model.V[n]),
                    pyo.value(model.mass[n]), pyo.value(model.thrust[n])))
    time, V, mass, thrust = zip(*samples)
    summary = dict(case=case['name'], CI=CI, cost_time_unit=cost_time_unit,
                   use_scaling=use_scaling, time_s=tf, fuel_kg=mass0-mass[-1], DOC=pyo.value(model.obj),
                   time_bound_active=min(abs(tf-time_bounds[0]), abs(tf-time_bounds[1])) < 0.1,
                   max_scaled_violation=worst_scaled_violation)
    summary.update(
        enforce_max_thrust=enforce_max_thrust,
        enforce_min_thrust=enforce_min_thrust,
        enforce_acceleration_limit=enforce_acceleration_limit,
        min_thrust_kN=min(thrust) / 1000,
        max_thrust_kN=max(thrust) / 1000,
        max_tas_increase_m_s2=max((right - left) / (dt / 2) for left, right in zip(V, V[1:])),
    )
    summary['nodes'] = N
    summary['stages'] = 2 * n + 1
    summary['termination'] = str(result.solver.termination_condition)
    summaries.append(summary)
    profiles.append((label, time, V, mass, thrust))
    print(json.dumps(summary), flush=True)

if make_plots:
    import matplotlib.pyplot as plt

    output_folder.mkdir(parents=True, exist_ok=True)
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
    baseline = {row['CI']: row for row in summaries if row['case'] == 'baseline'}
    metrics = [
        ('time_s', 'Flight time difference [s]'),
        ('fuel_kg', 'Fuel burn difference [kg]'),
        ('min_thrust_kN', 'Minimum thrust difference [kN]'),
        ('max_thrust_kN', 'Peak thrust difference [kN]'),
    ]
    fig3, sensitivity_axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    for ax, (key, ylabel) in zip(sensitivity_axes.flat, metrics):
        for case in cases:
            if case['name'] == 'baseline':
                continue
            rows = [row for row in summaries if row['case'] == case['name']]
            ax.plot(range(len(rows)), [row[key] - baseline[row['CI']][key] for row in rows],
                    'o-', label=case['name'].replace('_', ' '))
        ax.axhline(0, color='gray', linewidth=0.8)
        ax.set(xlabel='Cost Index', ylabel=ylabel, xticks=range(len(CI_values)), xticklabels=CI_values)
        ax.grid(alpha=0.3)
    handles, labels = sensitivity_axes[0, 0].get_legend_handles_labels()
    fig3.legend(handles, labels, loc='outside lower center', ncols=3)
    fig3.savefig(output_folder / 'sensitivity.png', dpi=180)
    if show_plots:
        plt.show()
    plt.close('all')
