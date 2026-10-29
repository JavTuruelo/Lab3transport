"""Task 4: continuous vertical A320 profile with free arrival time.
Run: python lab3_task4_vertical.py
Hermite-Simpson collocation with linear control interpolation of all five lab equations; controls T and CL.
Boundary assumptions: 1000 km, start/end at 10000 ft, 300 kt TAS, gamma=0.
The clean configuration is retained throughout: this is not a runway-to-runway
model. Change the explicit case settings for sensitivity studies.
DOC follows the lab's CI*tf + fuel_kg/100, with tf in seconds by default.
ISA is smoothed over approximately 2 m at the tropopause for IPOPT.
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


# 1. Editable flight conditions and sensitivity cases
N = 100  # endpoint nodes; midpoint stages are additional
CI_values = [0, 10, 50, 200, 999]
cost_time_unit = 'seconds'
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
cases = [
    {'name': 'baseline', 'mass': 51000.0, 'ceiling_ft': 41000.0},
    {'name': 'lower_ceiling', 'mass': 51000.0, 'ceiling_ft': 35000.0},
    {'name': 'heavier', 'mass': 55000.0, 'ceiling_ft': 41000.0},
]
show_solver = False
make_plots = True
show_plots = False
output_folder = Path(__file__).resolve().parent / 'results_task4'
if cost_time_unit not in ('seconds', 'hours'):
    raise ValueError('cost_time_unit must be seconds or hours')
cost_time_divisor = 3600.0 if cost_time_unit == 'hours' else 1.0


def lift(h, V, CL):
    rho = air_pressure(h) / (R * temperature(h))
    return 0.5 * rho * V**2 * S * CL


def drag(h, V, CL):
    rho = air_pressure(h) / (R * temperature(h))
    return 0.5 * rho * V**2 * S * (aircraft['CD0_CR'] + aircraft['CD2_CR'] * CL**2)


def rates(model, i):
    return (model.V[i]*pyo.cos(model.gamma[i]),
            model.V[i]*pyo.sin(model.gamma[i]),
            (model.thrust[i]-drag(model.h[i],model.V[i],model.CL[i]))/model.mass[i]-g*pyo.sin(model.gamma[i]),
            (lift(model.h[i],model.V[i],model.CL[i])-model.mass[i]*g*pyo.cos(model.gamma[i]))/(model.mass[i]*model.V[i]),
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
previous_case = None
previous_CI = None
previous_values = {}

for case, CI in [(case, CI) for case in cases for CI in CI_values]:
    label = f"{case['name']}_CI{CI}"
    mass0 = case['mass']
    ceiling = min(case['ceiling_ft'], aircraft['h_MO_ft']) * 0.3048
    floor = min(initial_altitude, final_altitude)
    n = 2 * (N - 1)  # endpoint and midpoint stages
    model = pyo.ConcreteModel()
    model.N = pyo.RangeSet(0, n)
    model.I = pyo.RangeSet(0, N-2)
    model.tf = pyo.Var(bounds=time_bounds, initialize=5500)
    model.x = pyo.Var(model.N, bounds=(0, distance), initialize=0)
    model.h = pyo.Var(model.N, bounds=(floor, ceiling), initialize=initial_altitude)
    model.V = pyo.Var(model.N, bounds=(100, 270), initialize=210)
    model.gamma = pyo.Var(model.N, bounds=(-flight_path_limit, flight_path_limit), initialize=0)
    model.mass = pyo.Var(model.N, bounds=(aircraft['m_min'], mass0), initialize=mass0)
    model.thrust = pyo.Var(model.N, bounds=(0, aircraft['C_tc1']), initialize=30000)
    model.CL = pyo.Var(model.N, bounds=CL_bounds, initialize=0.5)
    for i in model.N:
        fraction = i/n
        model.x[i].value = distance*fraction
        model.h[i].value = initial_altitude + (final_altitude-initial_altitude)*fraction + (min(33000*0.3048, ceiling)-floor)*math.sin(math.pi*fraction)**2
        model.V[i].value = initial_speed + (final_speed-initial_speed)*fraction + 55*math.sin(math.pi*fraction)
        model.mass[i].value = mass0 - 2500*fraction
        rho_guess = pyo.value(air_pressure(model.h[i].value)/(R*temperature(model.h[i].value)))
        model.CL[i].value = min(CL_bounds[1], max(CL_bounds[0], 2*model.mass[i].value*g/(rho_guess*model.V[i].value**2*S)))

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

    def altitude_dynamics_rule(model, i):
        left, middle, right = 2*i, 2*i+1, 2*i+2
        f_left = rates(model, left)[1]
        f_mid = rates(model, middle)[1]
        f_right = rates(model, right)[1]
        return model.h[right]-model.h[left] == model.tf/(3*n)*(f_left+4*f_mid+f_right)

    model.altitude_dynamics = pyo.Constraint(model.I, rule=altitude_dynamics_rule)

    def midpoint_h_rule(model, i):
        left, middle, right = 2*i, 2*i+1, 2*i+2
        f_left = rates(model, left)[1]
        f_right = rates(model, right)[1]
        return model.h[middle] == (model.h[left]+model.h[right])/2+model.tf/(4*n)*(f_left-f_right)

    model.midpoint_h = pyo.Constraint(model.I, rule=midpoint_h_rule)

    def speed_dynamics_rule(model, i):
        left, middle, right = 2*i, 2*i+1, 2*i+2
        f_left = rates(model, left)[2]
        f_mid = rates(model, middle)[2]
        f_right = rates(model, right)[2]
        return model.V[right]-model.V[left] == model.tf/(3*n)*(f_left+4*f_mid+f_right)

    model.speed_dynamics = pyo.Constraint(model.I, rule=speed_dynamics_rule)

    def midpoint_V_rule(model, i):
        left, middle, right = 2*i, 2*i+1, 2*i+2
        f_left = rates(model, left)[2]
        f_right = rates(model, right)[2]
        return model.V[middle] == (model.V[left]+model.V[right])/2+model.tf/(4*n)*(f_left-f_right)

    model.midpoint_V = pyo.Constraint(model.I, rule=midpoint_V_rule)

    def flight_path_dynamics_rule(model, i):
        left, middle, right = 2*i, 2*i+1, 2*i+2
        f_left = rates(model, left)[3]
        f_mid = rates(model, middle)[3]
        f_right = rates(model, right)[3]
        return model.gamma[right]-model.gamma[left] == model.tf/(3*n)*(f_left+4*f_mid+f_right)

    model.flight_path_dynamics = pyo.Constraint(model.I, rule=flight_path_dynamics_rule)

    def midpoint_gamma_rule(model, i):
        left, middle, right = 2*i, 2*i+1, 2*i+2
        f_left = rates(model, left)[3]
        f_right = rates(model, right)[3]
        return model.gamma[middle] == (model.gamma[left]+model.gamma[right])/2+model.tf/(4*n)*(f_left-f_right)

    model.midpoint_gamma = pyo.Constraint(model.I, rule=midpoint_gamma_rule)

    def mass_dynamics_rule(model, i):
        left, middle, right = 2*i, 2*i+1, 2*i+2
        f_left = rates(model, left)[4]
        f_mid = rates(model, middle)[4]
        f_right = rates(model, right)[4]
        return model.mass[right]-model.mass[left] == model.tf/(3*n)*(f_left+4*f_mid+f_right)

    model.mass_dynamics = pyo.Constraint(model.I, rule=mass_dynamics_rule)

    def midpoint_mass_rule(model, i):
        left, middle, right = 2*i, 2*i+1, 2*i+2
        f_left = rates(model, left)[4]
        f_right = rates(model, right)[4]
        return model.mass[middle] == (model.mass[left]+model.mass[right])/2+model.tf/(4*n)*(f_left-f_right)

    model.midpoint_mass = pyo.Constraint(model.I, rule=midpoint_mass_rule)

    def midpoint_thrust_rule(model, i):
        return model.thrust[2*i+1] == (model.thrust[2*i]+model.thrust[2*i+2])/2

    model.midpoint_thrust = pyo.Constraint(model.I, rule=midpoint_thrust_rule)

    def midpoint_CL_rule(model, i):
        return model.CL[2*i+1] == (model.CL[2*i]+model.CL[2*i+2])/2

    model.midpoint_CL = pyo.Constraint(model.I, rule=midpoint_CL_rule)

    # 3. Performance and envelope constraints at each node
    def maximum_thrust_rule(model, i):
        return model.thrust[i] <= maximum_thrust(model.h[i])

    model.maximum_thrust = pyo.Constraint(model.N, rule=maximum_thrust_rule)

    def minimum_thrust_rule(model, i):
        return model.thrust[i]*eta(model.V[i]) >= minimum_fuel(model.h[i])

    model.minimum_thrust = pyo.Constraint(model.N, rule=minimum_thrust_rule)

    def maximum_mach_rule(model, i):
        return model.V[i]**2 <= aircraft['M_MO']**2*gamma_air*R*temperature(model.h[i])

    model.maximum_mach = pyo.Constraint(model.N, rule=maximum_mach_rule)

    def maximum_speed_rule(model, i):
        return calibrated_speed(model.V[i],model.h[i])/knots_to_ms <= aircraft['V_MO_kn']

    model.maximum_speed = pyo.Constraint(model.N, rule=maximum_speed_rule)

    def minimum_speed_rule(model, i):
        return calibrated_speed(model.V[i],model.h[i])/knots_to_ms >= 1.3*aircraft['Vstall_CR_kn']*pyo.sqrt(model.mass[i]/aircraft['m_ref'])

    model.minimum_speed = pyo.Constraint(model.N, rule=minimum_speed_rule)

    def altitude_limit_rule(model, i):
        return model.h[i]/0.3048 <= aircraft['h_max_ft']+aircraft['G_w']*(aircraft['m_max']-model.mass[i])

    model.altitude_limit = pyo.Constraint(model.N, rule=altitude_limit_rule)

    def buffet_rule(model, i):
        mach = model.V[i]/pyo.sqrt(gamma_air*R*temperature(model.h[i]))
        # Smooth activation above 15000 ft avoids a discrete altitude switch.
        # Below the threshold the positive allowance relaxes this constraint;
        # the independent clean stall-speed constraint remains in force.
        allowance = 0.5/(1+pyo.exp((model.h[i]-15000*0.3048)/30))
        return aircraft['Clbo_M0']*mach**2-aircraft['K']*mach**3+allowance >= model.mass[i]*g/(0.583*air_pressure(model.h[i])*S)

    model.buffet = pyo.Constraint(model.N, rule=buffet_rule)

    # 4. Boundary conditions; no fixed cruise altitude or cruise phase duration
    def initial_distance_rule(model):
        return model.x[0] == 0

    model.initial_distance = pyo.Constraint(rule=initial_distance_rule)

    def final_distance_rule(model):
        return model.x[n] == distance

    model.final_distance = pyo.Constraint(rule=final_distance_rule)

    def initial_altitude_rule(model):
        return model.h[0] == initial_altitude

    model.initial_altitude = pyo.Constraint(rule=initial_altitude_rule)

    def final_altitude_rule(model):
        return model.h[n] == final_altitude

    model.final_altitude = pyo.Constraint(rule=final_altitude_rule)

    def initial_speed_rule(model):
        return model.V[0] == initial_speed

    model.initial_speed = pyo.Constraint(rule=initial_speed_rule)

    def final_speed_rule(model):
        return model.V[n] == final_speed

    model.final_speed = pyo.Constraint(rule=final_speed_rule)

    def initial_gamma_rule(model):
        return model.gamma[0] == initial_gamma

    model.initial_gamma = pyo.Constraint(rule=initial_gamma_rule)

    def final_gamma_rule(model):
        return model.gamma[n] == final_gamma

    model.final_gamma = pyo.Constraint(rule=final_gamma_rule)

    def initial_mass_rule(model):
        return model.mass[0] == mass0

    model.initial_mass = pyo.Constraint(rule=initial_mass_rule)

    def objective_rule(model):
        return CI*model.tf/cost_time_divisor+(mass0-model.mass[n])/100

    model.obj = pyo.Objective(rule=objective_rule,sense=pyo.minimize)
    model.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    model.scaling_factor[model.tf] = 1e-3
    for var, scale in [(model.x,1e-6),(model.h,1e-4),(model.V,1e-2),(model.gamma,1),(model.mass,1e-4),(model.thrust,1e-4),(model.CL,1)]:
        for i in model.N:
            model.scaling_factor[var[i]] = scale
    for c in model.component_data_objects(pyo.Constraint):
        model.scaling_factor[c] = 1.0
    for component,scale in [(model.distance_dynamics,1e-6),(model.altitude_dynamics,1e-4),(model.speed_dynamics,1e-2),(model.mass_dynamics,1e-4)]:
        for i in model.I:
            model.scaling_factor[component[i]] = scale
    for component,scale in [(model.maximum_thrust,1e-4),(model.maximum_mach,1e-4),(model.altitude_limit,1e-4)]:
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
    summary = dict(case=case['name'],CI=CI,cost_time_unit=cost_time_unit,
                   time_s=tf,fuel_kg=mass0-mass[-1],DOC=pyo.value(model.obj),
                   peak_altitude_ft=max(altitude)/0.3048,
                   mean_cruise_altitude_ft=sum(altitude[i] for i in near_level)/len(near_level)/0.3048 if near_level else None,
                   mean_cruise_speed_kn=sum(V[i] for i in near_level)/len(near_level)/knots_to_ms if near_level else None,
                   max_scaled_violation=worst_scaled_violation,
                   time_bound_active=min(abs(tf-time_bounds[0]),abs(tf-time_bounds[1])) < 0.1,
                   CL_bound_active=min(min(CL)-CL_bounds[0], CL_bounds[1]-max(CL)) < 1e-4)
    summary['nodes'] = N
    summary['stages'] = n + 1
    summary['termination'] = str(result.solver.termination_condition)
    summaries.append(summary)
    profiles.append((label,x,altitude,V,time,thrust,gamma))
    with (output_folder/f'{label}.csv').open('w',newline='') as file:
        writer=csv.writer(file)
        writer.writerow(['time_s','distance_m','altitude_m','TAS_m_s','flight_path_deg','mass_kg','thrust_N','CL'])
        writer.writerows(zip(time,x,altitude,V,map(math.degrees,gamma),mass,thrust,CL))
    print(json.dumps(summary),flush=True)

(output_folder/'summary.json').write_text(json.dumps(summaries,indent=2))
if make_plots:
    import matplotlib.pyplot as plt
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
    if show_plots:
        plt.show()
    plt.close('all')
