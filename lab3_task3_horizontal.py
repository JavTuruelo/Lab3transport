"""Task 3: horizontal spherical-Earth A320 trajectory at FL380.
Run: python lab3_task3_horizontal.py
Hermite-Simpson collocation with linear control interpolation; minimum fuel at a fixed common arrival time.
Approximate Madrid-Barajas / Paris-CDG coordinates are editable modelling
inputs. They locate the horizontal endpoints only: no airport climb/descent.
Heading chi is measured from EAST towards NORTH, as in the lab equations.
The heading equation follows the statement exactly (no added transport term).
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


# 1. Settings and explicitly chosen boundary conditions
N = 100  # endpoint nodes; midpoint stages are additional
Re = 6_371_000.0
start_lat_deg, start_lon_deg = 40.472, -3.561
end_lat_deg, end_lon_deg = 49.010, 2.548
h = 38000 * 0.3048
V0 = 420 * knots_to_ms
show_solver = False
make_plots = True
show_plots = False
output_folder = Path(__file__).resolve().parent / 'results_task3'
cases = [
    {'name': 'zero_wind', 'wx': 0.0, 'wy': 0.0, 'mass': 51000.0, 'bank_deg': 35.0},
    {'name': 'east_wind', 'wx': 30.0, 'wy': 0.0, 'mass': 51000.0, 'bank_deg': 35.0},
    {'name': 'north_wind', 'wx': 0.0, 'wy': 20.0, 'mass': 51000.0, 'bank_deg': 35.0},
    {'name': 'combined_wind', 'wx': 30.0, 'wy': 20.0, 'mass': 51000.0, 'bank_deg': 35.0},
    {'name': 'heavier', 'wx': 0.0, 'wy': 0.0, 'mass': 55000.0, 'bank_deg': 35.0},
    {'name': 'bank_limit', 'wx': 30.0, 'wy': 20.0, 'mass': 51000.0, 'bank_deg': 15.0},
]
lon0, lat0 = math.radians(start_lon_deg), math.radians(start_lat_deg)
lonf, latf = math.radians(end_lon_deg), math.radians(end_lat_deg)
a = [math.cos(lat0)*math.cos(lon0), math.cos(lat0)*math.sin(lon0), math.sin(lat0)]
b = [math.cos(latf)*math.cos(lonf), math.cos(latf)*math.sin(lonf), math.sin(latf)]
angle = math.acos(max(-1.0, min(1.0, sum(x*y for x, y in zip(a, b)))))
great_circle_distance = Re * angle
final_time = great_circle_distance / 212.766  # same fixed time in every wind case
n = 2 * (N - 1)  # endpoint and midpoint stages
step = final_time / n
# Spherical linear interpolation gives the exact orthodromic reference.
reference_lon = []
reference_lat = []
for i in range(n+1):
    f = i / n
    vector = [(math.sin((1-f)*angle)*a[j] + math.sin(f*angle)*b[j]) / math.sin(angle) for j in range(3)]
    reference_lon.append(math.atan2(vector[1], vector[0]))
    reference_lat.append(math.atan2(vector[2], math.hypot(vector[0], vector[1])))
normal = [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
normal_length = math.sqrt(sum(v*v for v in normal))
normal = [v/normal_length for v in normal]
rho = pyo.value(air_pressure(h) / (R * temperature(h)))
sound = math.sqrt(gamma_air * R * pyo.value(temperature(h)))


def drag(V, mass, bank):
    qS = 0.5 * rho * V**2 * S
    lift = mass * g / pyo.cos(bank)
    return qS * aircraft['CD0_CR'] + aircraft['CD2_CR'] * lift**2 / qS


def rates(model, i):
    return ((model.V[i]*pyo.cos(model.chi[i])+wx)/(Re*pyo.cos(model.lat[i])),
            (model.V[i]*pyo.sin(model.chi[i])+wy)/Re,
            (model.thrust[i]-drag(model.V[i],model.mass[i],model.bank[i]))/model.mass[i],
            g*pyo.tan(model.bank[i])/model.V[i],
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

for case in cases:
    label = case['name']
    wx, wy = case['wx'], case['wy']  # eastward/northward wind [m/s]
    mass0 = case['mass']
    bank_limit = math.radians(case['bank_deg'])
    ceiling = min(aircraft['h_MO_ft'], aircraft['h_max_ft'] + aircraft['G_w'] * (aircraft['m_max']-mass0))
    if h / 0.3048 > ceiling:
        raise ValueError(f'{label}: initial altitude exceeds the BADA ceiling.')
    model = pyo.ConcreteModel()
    model.N = pyo.RangeSet(0, n)
    model.I = pyo.RangeSet(0, N-2)
    model.lon = pyo.Var(model.N, bounds=(min(lon0, lonf)-0.2, max(lon0, lonf)+0.2))
    model.lat = pyo.Var(model.N, bounds=(min(lat0, latf)-0.2, max(lat0, latf)+0.2))
    model.V = pyo.Var(model.N, bounds=(100, aircraft['M_MO']*sound), initialize=213)
    model.chi = pyo.Var(model.N, bounds=(-math.pi, math.pi), initialize=0.8)
    model.mass = pyo.Var(model.N, bounds=(aircraft['m_min'], mass0), initialize=mass0)
    model.thrust = pyo.Var(model.N, bounds=(0, pyo.value(maximum_thrust(h))), initialize=25000)
    model.bank = pyo.Var(model.N, bounds=(-bank_limit, bank_limit), initialize=0)
    for i in model.N:
        model.lon[i].value = reference_lon[i]
        model.lat[i].value = reference_lat[i]
        j = min(i, n-1)
        east = Re * math.cos(reference_lat[j]) * (reference_lon[j+1]-reference_lon[j]) / step
        north = Re * (reference_lat[j+1]-reference_lat[j]) / step
        model.chi[i].value = math.atan2(north-wy, east-wx)
        model.V[i].value = min(aircraft['M_MO']*sound, math.hypot(east-wx, north-wy))

    # Hermite-Simpson: even indices are endpoints; odd indices are midpoints.
    # N is the number of endpoint nodes, giving 2*N-1 total stages.
    def longitude_dynamics_rule(model, i):
        left, middle, right = 2*i, 2*i+1, 2*i+2
        f_left = rates(model, left)[0]
        f_mid = rates(model, middle)[0]
        f_right = rates(model, right)[0]
        return model.lon[right]-model.lon[left] == final_time/(3*n)*(f_left+4*f_mid+f_right)

    model.longitude_dynamics = pyo.Constraint(model.I, rule=longitude_dynamics_rule)

    def midpoint_lon_rule(model, i):
        left, middle, right = 2*i, 2*i+1, 2*i+2
        f_left = rates(model, left)[0]
        f_right = rates(model, right)[0]
        return model.lon[middle] == (model.lon[left]+model.lon[right])/2+final_time/(4*n)*(f_left-f_right)

    model.midpoint_lon = pyo.Constraint(model.I, rule=midpoint_lon_rule)

    def latitude_dynamics_rule(model, i):
        left, middle, right = 2*i, 2*i+1, 2*i+2
        f_left = rates(model, left)[1]
        f_mid = rates(model, middle)[1]
        f_right = rates(model, right)[1]
        return model.lat[right]-model.lat[left] == final_time/(3*n)*(f_left+4*f_mid+f_right)

    model.latitude_dynamics = pyo.Constraint(model.I, rule=latitude_dynamics_rule)

    def midpoint_lat_rule(model, i):
        left, middle, right = 2*i, 2*i+1, 2*i+2
        f_left = rates(model, left)[1]
        f_right = rates(model, right)[1]
        return model.lat[middle] == (model.lat[left]+model.lat[right])/2+final_time/(4*n)*(f_left-f_right)

    model.midpoint_lat = pyo.Constraint(model.I, rule=midpoint_lat_rule)

    def speed_dynamics_rule(model, i):
        left, middle, right = 2*i, 2*i+1, 2*i+2
        f_left = rates(model, left)[2]
        f_mid = rates(model, middle)[2]
        f_right = rates(model, right)[2]
        return model.V[right]-model.V[left] == final_time/(3*n)*(f_left+4*f_mid+f_right)

    model.speed_dynamics = pyo.Constraint(model.I, rule=speed_dynamics_rule)

    def midpoint_V_rule(model, i):
        left, middle, right = 2*i, 2*i+1, 2*i+2
        f_left = rates(model, left)[2]
        f_right = rates(model, right)[2]
        return model.V[middle] == (model.V[left]+model.V[right])/2+final_time/(4*n)*(f_left-f_right)

    model.midpoint_V = pyo.Constraint(model.I, rule=midpoint_V_rule)

    def heading_dynamics_rule(model, i):
        left, middle, right = 2*i, 2*i+1, 2*i+2
        f_left = rates(model, left)[3]
        f_mid = rates(model, middle)[3]
        f_right = rates(model, right)[3]
        return model.chi[right]-model.chi[left] == final_time/(3*n)*(f_left+4*f_mid+f_right)

    model.heading_dynamics = pyo.Constraint(model.I, rule=heading_dynamics_rule)

    def midpoint_chi_rule(model, i):
        left, middle, right = 2*i, 2*i+1, 2*i+2
        f_left = rates(model, left)[3]
        f_right = rates(model, right)[3]
        return model.chi[middle] == (model.chi[left]+model.chi[right])/2+final_time/(4*n)*(f_left-f_right)

    model.midpoint_chi = pyo.Constraint(model.I, rule=midpoint_chi_rule)

    def mass_dynamics_rule(model, i):
        left, middle, right = 2*i, 2*i+1, 2*i+2
        f_left = rates(model, left)[4]
        f_mid = rates(model, middle)[4]
        f_right = rates(model, right)[4]
        return model.mass[right]-model.mass[left] == final_time/(3*n)*(f_left+4*f_mid+f_right)

    model.mass_dynamics = pyo.Constraint(model.I, rule=mass_dynamics_rule)

    def midpoint_mass_rule(model, i):
        left, middle, right = 2*i, 2*i+1, 2*i+2
        f_left = rates(model, left)[4]
        f_right = rates(model, right)[4]
        return model.mass[middle] == (model.mass[left]+model.mass[right])/2+final_time/(4*n)*(f_left-f_right)

    model.midpoint_mass = pyo.Constraint(model.I, rule=midpoint_mass_rule)

    def midpoint_thrust_rule(model, i):
        return model.thrust[2*i+1] == (model.thrust[2*i]+model.thrust[2*i+2])/2

    model.midpoint_thrust = pyo.Constraint(model.I, rule=midpoint_thrust_rule)

    def midpoint_bank_rule(model, i):
        return model.bank[2*i+1] == (model.bank[2*i]+model.bank[2*i+2])/2

    model.midpoint_bank = pyo.Constraint(model.I, rule=midpoint_bank_rule)

    def minimum_thrust_rule(model, i):
        return model.thrust[i]*eta(model.V[i]) >= minimum_fuel(h)

    model.minimum_thrust = pyo.Constraint(model.N, rule=minimum_thrust_rule)

    def maximum_speed_rule(model, i):
        return calibrated_speed(model.V[i], h)/knots_to_ms <= aircraft['V_MO_kn']

    model.maximum_speed = pyo.Constraint(model.N, rule=maximum_speed_rule)

    def minimum_speed_rule(model, i):
        return calibrated_speed(model.V[i], h)/knots_to_ms >= 1.3*aircraft['Vstall_CR_kn']*pyo.sqrt(model.mass[i]/aircraft['m_ref']/pyo.cos(model.bank[i]))

    model.minimum_speed = pyo.Constraint(model.N, rule=minimum_speed_rule)

    def buffet_rule(model, i):
        mach = model.V[i]/sound
        return aircraft['Clbo_M0']*mach**2-aircraft['K']*mach**3 >= model.mass[i]*g/(pyo.cos(model.bank[i])*0.583*air_pressure(h)*S)

    model.buffet = pyo.Constraint(model.N, rule=buffet_rule)

    def initial_longitude_rule(model):
        return model.lon[0] == lon0

    model.initial_longitude = pyo.Constraint(rule=initial_longitude_rule)

    def final_longitude_rule(model):
        return model.lon[n] == lonf

    model.final_longitude = pyo.Constraint(rule=final_longitude_rule)

    def initial_latitude_rule(model):
        return model.lat[0] == lat0

    model.initial_latitude = pyo.Constraint(rule=initial_latitude_rule)

    def final_latitude_rule(model):
        return model.lat[n] == latf

    model.final_latitude = pyo.Constraint(rule=final_latitude_rule)

    def initial_speed_rule(model):
        return model.V[0] == V0

    model.initial_speed = pyo.Constraint(rule=initial_speed_rule)

    def initial_mass_rule(model):
        return model.mass[0] == mass0

    model.initial_mass = pyo.Constraint(rule=initial_mass_rule)

    def objective_rule(model):
        return mass0 - model.mass[n]

    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
    model.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    for var, scale in [(model.lon, 10), (model.lat, 10), (model.chi, 1), (model.bank, 1), (model.V, 1e-2), (model.mass, 1e-4), (model.thrust, 1e-4)]:
        for i in model.N:
            model.scaling_factor[var[i]] = scale
    for c in model.component_data_objects(pyo.Constraint):
        model.scaling_factor[c] = 1.0
    for component, scale in [(model.longitude_dynamics, 10), (model.latitude_dynamics, 10), (model.speed_dynamics, 1e-2), (model.mass_dynamics, 1e-4)]:
        for i in model.I:
            model.scaling_factor[component[i]] = scale
    model.scaling_factor[model.initial_speed] = 1e-2
    model.scaling_factor[model.initial_mass] = 1e-4
    model.scaling_factor[model.obj] = 1e-4
    for i in model.I:
        model.scaling_factor[model.midpoint_lon[i]] = 10
    for i in model.I:
        model.scaling_factor[model.midpoint_lat[i]] = 10
    for i in model.I:
        model.scaling_factor[model.midpoint_V[i]] = 0.01
    for i in model.I:
        model.scaling_factor[model.midpoint_chi[i]] = 1
    for i in model.I:
        model.scaling_factor[model.midpoint_mass[i]] = 0.0001
    for i in model.I:
        model.scaling_factor[model.midpoint_thrust[i]] = 0.0001
    for i in model.I:
        model.scaling_factor[model.midpoint_bank[i]] = 1

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
    time = [i*step for i in model.N]
    lon = [pyo.value(model.lon[i]) for i in model.N]
    lat = [pyo.value(model.lat[i]) for i in model.N]
    V = [pyo.value(model.V[i]) for i in model.N]
    mass = [pyo.value(model.mass[i]) for i in model.N]
    thrust = [pyo.value(model.thrust[i]) for i in model.N]
    chi = [pyo.value(model.chi[i]) for i in model.N]
    bank = [pyo.value(model.bank[i]) for i in model.N]
    cross_track = []
    for lo, la in zip(lon, lat):
        vector = [math.cos(la)*math.cos(lo), math.cos(la)*math.sin(lo), math.sin(la)]
        cross_track.append(Re*math.asin(max(-1, min(1, sum(x*y for x, y in zip(normal, vector))))))
    route_length = 0.0
    for i in range(n):
        dlat, dlon = lat[i+1]-lat[i], lon[i+1]-lon[i]
        hav = math.sin(dlat/2)**2+math.cos(lat[i])*math.cos(lat[i+1])*math.sin(dlon/2)**2
        route_length += 2*Re*math.asin(math.sqrt(min(1, max(0, hav))))
    summary = dict(case=label, time_s=final_time, fuel_kg=mass0-mass[-1],
                   great_circle_km=great_circle_distance/1000, route_km=route_length/1000,
                   max_cross_track_km=max(abs(v) for v in cross_track)/1000,
                   wx_m_s=wx, wy_m_s=wy, max_scaled_violation=worst_scaled_violation)
    summary['nodes'] = N
    summary['stages'] = n + 1
    summary['termination'] = str(result.solver.termination_condition)
    summaries.append(summary)
    profiles.append((label, time, lon, lat, V, thrust, cross_track))
    with (output_folder / f'{label}.csv').open('w', newline='') as file:
        writer=csv.writer(file)
        writer.writerow(['time_s','longitude_deg','latitude_deg','TAS_m_s','heading_from_east_deg','mass_kg','thrust_N','bank_deg','cross_track_m'])
        writer.writerows(zip(time, map(math.degrees,lon), map(math.degrees,lat), V, map(math.degrees,chi), mass, thrust, map(math.degrees,bank),cross_track))
    print(json.dumps(summary), flush=True)

with (output_folder / 'great_circle.csv').open('w', newline='') as file:
    writer=csv.writer(file)
    writer.writerow(['longitude_deg','latitude_deg'])
    writer.writerows(zip(map(math.degrees,reference_lon), map(math.degrees,reference_lat)))
(output_folder / 'summary.json').write_text(json.dumps(summaries,indent=2))
if make_plots:
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(2,2,figsize=(12,8))
    axes[0,0].plot(list(map(math.degrees,reference_lon)),list(map(math.degrees,reference_lat)),'k--',label='Great circle')
    for label,time,lon,lat,V,thrust,cross_track in profiles:
        axes[0,0].plot(list(map(math.degrees,lon)),list(map(math.degrees,lat)),label=label)
        axes[0,1].plot(time,[c/1000 for c in cross_track],label=label)
        axes[1,0].plot(time,[v/knots_to_ms for v in V],label=label)
        axes[1,1].plot(time,[t/1000 for t in thrust],label=label)
    axes[0,0].set(xlabel='Longitude [deg]',ylabel='Latitude [deg]')
    axes[0,0].set_aspect(1/math.cos((lat0+latf)/2))
    axes[0,1].set(xlabel='Time [s]',ylabel='Cross-track distance [km]')
    axes[1,0].set(xlabel='Time [s]',ylabel='TAS [knots]')
    axes[1,1].set(xlabel='Time [s]',ylabel='Thrust [kN]')
    for ax in axes.flat:
        ax.grid(True)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_folder/'horizontal_profiles.png',dpi=180)
    if show_plots:
        plt.show()
    plt.close('all')
