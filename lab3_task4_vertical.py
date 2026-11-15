"""Task 4: continuous vertical A320 profile with free arrival time.
"""
import json
import math
from pathlib import Path
from types import SimpleNamespace
import pyomo.environ as pyo

import lab3_task4_rules as rules
from lab3_functions import aircraft, g, R, S, knots_to_ms


# 1. Editable flight conditions and sensitivity cases
N = 100  # endpoint nodes; midpoint stages are additional
CI_values = [0, 10, 50, 200, 999]
cost_time_unit = 'hours'  # Hours and pounds for DOC; seconds and kg for dynamics.
distance = 1_000_000.0
initial_altitude = 10000 * 0.3048
final_altitude = 10000 * 0.3048
initial_speed = 300 * knots_to_ms
final_speed = 300 * knots_to_ms
initial_gamma = 0.0
final_gamma = 0.0
time_bounds = (2500.0, 12000.0)
flight_path_limit = math.radians(10)
CL_bounds = (0.05, 1.5)  # explicit assumed clean control bounds; sensitivity parameter
baseline_case = {'name': 'baseline', 'mass': 51000.0, 'ceiling_ft': 41000.0}
cases = [
    baseline_case,
    {'name': 'lower_ceiling', 'mass': 51000.0, 'ceiling_ft': 35000.0},
    {'name': 'heavier', 'mass': 55000.0, 'ceiling_ft': 41000.0},
    {**baseline_case, 'name': 'max_thrust_off', 'enforce_max_thrust': False},
    {**baseline_case, 'name': 'min_thrust_off', 'enforce_min_thrust': False},
]
use_scaling = True  # Use manual IPOPT scaling when enabled.
show_solver = False
make_plots = True
show_plots = False
output_folder = Path(__file__).resolve().parent / 'results_task4'
if cost_time_unit not in ('seconds', 'hours'):
    raise ValueError('cost_time_unit must be seconds or hours')
cost_time_divisor = 3600.0 if cost_time_unit == 'hours' else 1.0
cost_fuel_multiplier = 1 / 0.45359237 if cost_time_unit == 'hours' else 1.0

solver = pyo.SolverFactory('ipopt')
if not solver.available(exception_flag=False):
    raise RuntimeError('Activate the Conda environment containing IPOPT.')
solver.options['max_iter'] = 5000
solver.options['tol'] = 1e-8
solver.options['nlp_scaling_method'] = 'user-scaling' if use_scaling else 'none'
summaries = []
profiles = []
previous_case = None
previous_CI = None
previous_values = {}

for case, CI in [(case, CI) for case in cases for CI in CI_values]:
    label = f"{case['name']}_CI{CI}"
    mass0 = case['mass']
    enforce_max_thrust = case.get('enforce_max_thrust', True)
    enforce_min_thrust = case.get('enforce_min_thrust', True)
    ceiling = min(case['ceiling_ft'], aircraft['h_MO_ft']) * 0.3048
    floor = min(initial_altitude, final_altitude)
    n = 2 * (N - 1)  # endpoint and midpoint stages
    model = pyo.ConcreteModel()
    model.params = SimpleNamespace(
        n=n, distance=distance, mass0=mass0, CI=CI, cost_time_divisor=cost_time_divisor,
        cost_fuel_multiplier=cost_fuel_multiplier,
        initial_altitude=initial_altitude, final_altitude=final_altitude,
        initial_speed=initial_speed, final_speed=final_speed,
        initial_gamma=initial_gamma, final_gamma=final_gamma,
    )
    model.N = pyo.RangeSet(0, n)
    model.I = pyo.RangeSet(0, N-2)
    model.tf = pyo.Var(bounds=time_bounds, initialize=5500)
    model.x = pyo.Var(model.N, bounds=(0, distance), initialize=0)
    model.h = pyo.Var(model.N, bounds=(floor, ceiling), initialize=initial_altitude)
    model.V = pyo.Var(model.N, bounds=(100, 270), initialize=210)
    model.gamma = pyo.Var(model.N, bounds=(-flight_path_limit, flight_path_limit), initialize=0)
    model.mass = pyo.Var(model.N, bounds=(aircraft['m_min'], mass0), initialize=mass0)
    model.thrust = pyo.Var(model.N, bounds=(0, aircraft['C_tc1'] if enforce_max_thrust else None), initialize=30000)
    model.CL = pyo.Var(model.N, bounds=CL_bounds, initialize=0.5)
    for i in model.N:
        fraction = i/n
        model.x[i].value = distance*fraction
        model.h[i].value = initial_altitude + (final_altitude-initial_altitude)*fraction + (min(33000*0.3048, ceiling)-floor)*math.sin(math.pi*fraction)**2
        model.V[i].value = initial_speed + (final_speed-initial_speed)*fraction + 55*math.sin(math.pi*fraction)
        model.mass[i].value = mass0 - 2500*fraction
        rho_guess = pyo.value(rules.air_pressure(model.h[i].value)/(R*rules.temperature(model.h[i].value)))
        model.CL[i].value = min(CL_bounds[1], max(CL_bounds[0], 2*model.mass[i].value*g/(rho_guess*model.V[i].value**2*S)))

    # Hermite-Simpson: even indices are endpoints; odd indices are midpoints.
    # N is the number of endpoint nodes, giving 2*N-1 total stages.
    model.distance_dynamics = pyo.Constraint(model.I, rule=rules.distance_dynamics_rule)

    model.midpoint_x = pyo.Constraint(model.I, rule=rules.midpoint_x_rule)

    model.altitude_dynamics = pyo.Constraint(model.I, rule=rules.altitude_dynamics_rule)

    model.midpoint_h = pyo.Constraint(model.I, rule=rules.midpoint_h_rule)

    model.speed_dynamics = pyo.Constraint(model.I, rule=rules.speed_dynamics_rule)

    model.midpoint_V = pyo.Constraint(model.I, rule=rules.midpoint_V_rule)

    model.flight_path_dynamics = pyo.Constraint(model.I, rule=rules.flight_path_dynamics_rule)

    model.midpoint_gamma = pyo.Constraint(model.I, rule=rules.midpoint_gamma_rule)

    model.mass_dynamics = pyo.Constraint(model.I, rule=rules.mass_dynamics_rule)

    model.midpoint_mass = pyo.Constraint(model.I, rule=rules.midpoint_mass_rule)

    model.midpoint_thrust = pyo.Constraint(model.I, rule=rules.midpoint_thrust_rule)

    model.midpoint_CL = pyo.Constraint(model.I, rule=rules.midpoint_CL_rule)

    # 3. Performance and envelope constraints at each node
    if enforce_max_thrust:
        model.maximum_thrust = pyo.Constraint(model.N, rule=rules.maximum_thrust_rule)

    if enforce_min_thrust:
        model.minimum_thrust = pyo.Constraint(model.N, rule=rules.minimum_thrust_rule)

    model.maximum_mach = pyo.Constraint(model.N, rule=rules.maximum_mach_rule)

    model.maximum_speed = pyo.Constraint(model.N, rule=rules.maximum_speed_rule)

    model.minimum_speed = pyo.Constraint(model.N, rule=rules.minimum_speed_rule)

    model.altitude_limit = pyo.Constraint(model.N, rule=rules.altitude_limit_rule)

    model.buffet = pyo.Constraint(model.N, rule=rules.buffet_rule)

    # 4. Boundary conditions; no fixed cruise altitude or cruise phase duration
    model.initial_distance = pyo.Constraint(rule=rules.initial_distance_rule)

    model.final_distance = pyo.Constraint(rule=rules.final_distance_rule)

    model.initial_altitude = pyo.Constraint(rule=rules.initial_altitude_rule)

    model.final_altitude = pyo.Constraint(rule=rules.final_altitude_rule)

    model.initial_speed = pyo.Constraint(rule=rules.initial_speed_rule)

    model.final_speed = pyo.Constraint(rule=rules.final_speed_rule)

    model.initial_gamma = pyo.Constraint(rule=rules.initial_gamma_rule)

    model.final_gamma = pyo.Constraint(rule=rules.final_gamma_rule)

    model.initial_mass = pyo.Constraint(rule=rules.initial_mass_rule)

    model.obj = pyo.Objective(rule=rules.objective_rule,sense=pyo.minimize)
    # Keep the residual check comparable in both modes.
    scaling_direction = pyo.Suffix.EXPORT if use_scaling else pyo.Suffix.LOCAL
    model.scaling_factor = pyo.Suffix(direction=scaling_direction)
    model.scaling_factor[model.tf] = 1e-3
    for var, scale in [(model.x,1e-6),(model.h,1e-4),(model.V,1e-2),(model.gamma,1),(model.mass,1e-4),(model.thrust,1e-4),(model.CL,1)]:
        for i in model.N:
            model.scaling_factor[var[i]] = scale
    for c in model.component_data_objects(pyo.Constraint):
        model.scaling_factor[c] = 1.0
    for component,scale in [(model.distance_dynamics,1e-6),(model.altitude_dynamics,1e-4),(model.speed_dynamics,1e-2),(model.mass_dynamics,1e-4)]:
        for i in model.I:
            model.scaling_factor[component[i]] = scale
    if enforce_max_thrust:
        for i in model.N:
            model.scaling_factor[model.maximum_thrust[i]] = 1e-4
    for component,scale in [(model.maximum_mach,1e-4),(model.altitude_limit,1e-4)]:
        for i in model.N:
            model.scaling_factor[component[i]] = scale
    for component,scale in [(model.initial_distance,1e-6),(model.final_distance,1e-6),(model.initial_altitude,1e-4),(model.final_altitude,1e-4),(model.initial_speed,1e-2),(model.final_speed,1e-2),(model.initial_mass,1e-4)]:
        model.scaling_factor[component] = scale
    model.scaling_factor[model.obj] = 1/max(1, CI*5500/cost_time_divisor)
    for i in model.I:
        model.scaling_factor[model.midpoint_x[i]] = 1e-06
    for i in model.I:
        model.scaling_factor[model.midpoint_h[i]] = 0.0001
    for i in model.I:
        model.scaling_factor[model.midpoint_V[i]] = 0.01
    for i in model.I:
        model.scaling_factor[model.midpoint_gamma[i]] = 1
    for i in model.I:
        model.scaling_factor[model.midpoint_mass[i]] = 0.0001
    for i in model.I:
        model.scaling_factor[model.midpoint_thrust[i]] = 0.0001
    for i in model.I:
        model.scaling_factor[model.midpoint_CL[i]] = 1

    # Continue from the preceding CI solution for this boundary-condition case.
    if previous_case == case['name'] and previous_CI != 0:
        for variable in model.component_data_objects(pyo.Var):
            guess = previous_values[variable.name]
            if variable.has_lb():
                guess = max(guess, pyo.value(variable.lb))
            if variable.has_ub():
                guess = min(guess, pyo.value(variable.ub))
            variable.set_value(guess)

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
    previous_case = case['name']
    previous_CI = CI
    previous_values = {var.name: pyo.value(var) for var in model.component_data_objects(pyo.Var)}
    tf = pyo.value(model.tf)
    time = [tf*i/n for i in model.N]
    x = [pyo.value(model.x[i]) for i in model.N]
    altitude = [pyo.value(model.h[i]) for i in model.N]
    V = [pyo.value(model.V[i]) for i in model.N]
    gamma = [pyo.value(model.gamma[i]) for i in model.N]
    mass = [pyo.value(model.mass[i]) for i in model.N]
    thrust = [pyo.value(model.thrust[i]) for i in model.N]
    CL = [pyo.value(model.CL[i]) for i in model.N]
    near_level = [i for i in range(n+1) if abs(gamma[i]) < math.radians(0.3) and 0.2*distance < x[i] < 0.8*distance]
    summary = dict(case=case['name'],CI=CI,cost_time_unit=cost_time_unit, cost_fuel_unit='lb' if cost_time_unit == 'hours' else 'kg',use_scaling=use_scaling,
                   time_s=tf,fuel_kg=mass0-mass[-1],DOC=pyo.value(model.obj),
                   peak_altitude_ft=max(altitude)/0.3048,
                   mean_cruise_altitude_ft=sum(altitude[i] for i in near_level)/len(near_level)/0.3048 if near_level else None,
                   mean_cruise_speed_kn=sum(V[i] for i in near_level)/len(near_level)/knots_to_ms if near_level else None,
                   max_scaled_violation=worst_scaled_violation,
                   time_bound_active=min(abs(tf-time_bounds[0]),abs(tf-time_bounds[1])) < 0.1,
                   CL_bound_active=min(min(CL)-CL_bounds[0], CL_bounds[1]-max(CL)) < 1e-4)
    summary.update(
        enforce_max_thrust=enforce_max_thrust,
        enforce_min_thrust=enforce_min_thrust,
        min_thrust_kN=min(thrust) / 1000,
        max_thrust_kN=max(thrust) / 1000,
    )
    summary['nodes'] = N
    summary['stages'] = n + 1
    summary['termination'] = str(result.solver.termination_condition)
    summaries.append(summary)
    profiles.append((label,x,altitude,V,time,thrust,gamma))
    print(json.dumps(summary),flush=True)

if make_plots:
    import matplotlib.pyplot as plt

    output_folder.mkdir(parents=True, exist_ok=True)
    for case in cases:
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for label,x,altitude,V,time,thrust,gamma in profiles:
            if not label.startswith(case['name']+'_CI'):
                continue
            axes[0,0].plot([v/1000 for v in x],[v/0.3048 for v in altitude],label=label)
            axes[0,1].plot(time,[v/knots_to_ms for v in V],label=label)
            axes[1,0].plot(time,[v/1000 for v in thrust],label=label)
            axes[1,1].plot(time,list(map(math.degrees,gamma)),label=label)
        axes[0,0].set(xlabel='Distance [km]',ylabel='Altitude [ft]')
        axes[0,1].set(xlabel='Time [s]',ylabel='TAS [knots]')
        axes[1,0].set(xlabel='Time [s]',ylabel='Thrust [kN]')
        axes[1,1].set(xlabel='Time [s]',ylabel='Flight-path angle [deg]')
        for ax in axes.flat:
            ax.grid(True)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output_folder/f"profiles_{case['name']}.png",dpi=180)
    fig2,ax=plt.subplots()
    for case in cases:
        rows=[s for s in summaries if s['case']==case['name']]
        ax.plot([s['time_s'] for s in rows],[s['fuel_kg'] for s in rows],'o-',label=case['name'])
    ax.set(xlabel='Flight time [s]',ylabel='Fuel burned [kg]')
    ax.grid(True)
    ax.legend()
    fig2.tight_layout()
    fig2.savefig(output_folder/'tradeoff.png',dpi=180)
    fig_sensitivity, axes_sensitivity = plt.subplots(2, 2, figsize=(11, 7))
    baseline = {row['CI']: row for row in summaries if row['case'] == 'baseline'}
    metrics = [
        ('fuel_kg', 'Fuel difference [kg]'),
        ('time_s', 'Flight time difference [s]'),
        ('peak_altitude_ft', 'Peak altitude difference [ft]'),
        ('mean_cruise_speed_kn', 'Mean cruise TAS difference [kt]'),
    ]
    for ax, (key, ylabel) in zip(axes_sensitivity.flat, metrics):
        for case_name in ('max_thrust_off', 'min_thrust_off'):
            rows = [row for row in summaries if row['case'] == case_name]
            differences = [
                row[key] - baseline[row['CI']][key]
                if row[key] is not None and baseline[row['CI']][key] is not None else math.nan
                for row in rows
            ]
            ax.plot(range(len(CI_values)), differences, 'o-', label=case_name)
        ax.set(xlabel='Cost index', ylabel=ylabel, xticks=range(len(CI_values)), xticklabels=CI_values)
    for ax in axes_sensitivity.flat:
        ax.grid(True)
        ax.legend()
    fig_sensitivity.tight_layout()
    fig_sensitivity.savefig(output_folder / 'thrust_sensitivity.png', dpi=180)
    if show_plots:
        plt.show()
    plt.close('all')
