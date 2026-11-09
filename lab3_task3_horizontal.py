"""Task 3: horizontal spherical-Earth A320 trajectory at FL380 (main script).
Run: python lab3_task3_horizontal.py
Hermite-Simpson collocation with linear control interpolation; minimum fuel at a fixed common arrival time.
Approximate Madrid-Barajas / Paris-CDG coordinates are editable modelling
inputs. They locate the horizontal endpoints only: no airport climb/descent.
Heading chi is measured from EAST towards NORTH, as in the lab equations.
The heading equation follows the statement exactly (no added transport term).

Structure:
  lab3_task3_functions.py  aircraft data, constants and auxiliary functions
  lab3_task3_rules.py      Pyomo constraint/objective rules
  lab3_task3_horizontal.py settings, geometry, model building, solve, outputs
"""
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
import pyomo.environ as pyo

import lab3_task3_rules as rules
from lab3_task3_functions import (
    R, Re, aircraft, air_pressure, gamma_air, knots_to_ms, maximum_thrust,
    temperature,
)


# Datos de la geometria (paris, madrid)
N = 100
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
final_time = great_circle_distance / 212.766  
n = 2 * (N - 1) 
step = final_time / n

reference_lon = []
reference_lat = []

# Ortodromica Madrid Paris

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


    # Datos del caso que leen las rules (lab3_task3_rules.py)
    model.params = SimpleNamespace(
        final_time=final_time, n=n, h=h, rho=rho, sound=sound,
        wx=wx, wy=wy, lon0=lon0, lonf=lonf, lat0=lat0, latf=latf,
        V0=V0, mass0=mass0,
    )

    # Dinamica (Hermite-Simpson)
    model.longitude_dynamics = pyo.Constraint(model.I, rule=rules.longitude_dynamics_rule)
    model.midpoint_lon = pyo.Constraint(model.I, rule=rules.midpoint_lon_rule)
    model.latitude_dynamics = pyo.Constraint(model.I, rule=rules.latitude_dynamics_rule)
    model.midpoint_lat = pyo.Constraint(model.I, rule=rules.midpoint_lat_rule)
    model.speed_dynamics = pyo.Constraint(model.I, rule=rules.speed_dynamics_rule)
    model.midpoint_V = pyo.Constraint(model.I, rule=rules.midpoint_V_rule)
    model.heading_dynamics = pyo.Constraint(model.I, rule=rules.heading_dynamics_rule)
    model.midpoint_chi = pyo.Constraint(model.I, rule=rules.midpoint_chi_rule)
    model.mass_dynamics = pyo.Constraint(model.I, rule=rules.mass_dynamics_rule)
    model.midpoint_mass = pyo.Constraint(model.I, rule=rules.midpoint_mass_rule)

    # Controles
    model.midpoint_thrust = pyo.Constraint(model.I, rule=rules.midpoint_thrust_rule)
    model.midpoint_bank = pyo.Constraint(model.I, rule=rules.midpoint_bank_rule)

    # Envolvente de vuelo
    model.minimum_thrust = pyo.Constraint(model.N, rule=rules.minimum_thrust_rule)
    model.maximum_speed = pyo.Constraint(model.N, rule=rules.maximum_speed_rule)
    model.minimum_speed = pyo.Constraint(model.N, rule=rules.minimum_speed_rule)
    model.buffet = pyo.Constraint(model.N, rule=rules.buffet_rule)

    # Condiciones de contorno
    model.initial_longitude = pyo.Constraint(rule=rules.initial_longitude_rule)
    model.final_longitude = pyo.Constraint(rule=rules.final_longitude_rule)
    model.initial_latitude = pyo.Constraint(rule=rules.initial_latitude_rule)
    model.final_latitude = pyo.Constraint(rule=rules.final_latitude_rule)
    model.initial_speed = pyo.Constraint(rule=rules.initial_speed_rule)
    model.initial_mass = pyo.Constraint(rule=rules.initial_mass_rule)

    # Objetivo: minimo combustible
    model.obj = pyo.Objective(rule=rules.objective_rule, sense=pyo.minimize)
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
