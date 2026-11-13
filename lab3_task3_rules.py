"""Task 3: Pyomo constraint and objective rules.

Every rule reads case data from model.params (final_time, n, h, wind, boundary
conditions, ...), which the main script lab3_task3_horizontal.py attaches.

Hermite-Simpson: even indices are endpoints; odd indices are midpoints.
N is the number of endpoint nodes, giving 2*N-1 total stages.
"""
import pyomo.environ as pyo

from lab3_functions import (
    S, aircraft, air_pressure, calibrated_speed, eta, g, knots_to_ms,
    minimum_fuel, rates,
)

# Dinamica: Simpson y Hermite


def longitude_dynamics_rule(model, i):
    p = model.params
    left, middle, right = 2*i, 2*i+1, 2*i+2
    f_left = rates(model, left)[0]
    f_mid = rates(model, middle)[0]
    f_right = rates(model, right)[0]
    return model.lon[right]-model.lon[left] == p.final_time/(3*p.n)*(f_left+4*f_mid+f_right)


def midpoint_lon_rule(model, i):
    p = model.params
    left, middle, right = 2*i, 2*i+1, 2*i+2
    f_left = rates(model, left)[0]
    f_right = rates(model, right)[0]
    return model.lon[middle] == (model.lon[left]+model.lon[right])/2+p.final_time/(4*p.n)*(f_left-f_right)


def latitude_dynamics_rule(model, i):
    p = model.params
    left, middle, right = 2*i, 2*i+1, 2*i+2
    f_left = rates(model, left)[1]
    f_mid = rates(model, middle)[1]
    f_right = rates(model, right)[1]
    return model.lat[right]-model.lat[left] == p.final_time/(3*p.n)*(f_left+4*f_mid+f_right)

def midpoint_lat_rule(model, i):
    p = model.params
    left, middle, right = 2*i, 2*i+1, 2*i+2
    f_left = rates(model, left)[1]
    f_right = rates(model, right)[1]
    return model.lat[middle] == (model.lat[left]+model.lat[right])/2+p.final_time/(4*p.n)*(f_left-f_right)


def speed_dynamics_rule(model, i):
    p = model.params
    left, middle, right = 2*i, 2*i+1, 2*i+2
    f_left = rates(model, left)[2]
    f_mid = rates(model, middle)[2]
    f_right = rates(model, right)[2]
    return model.V[right]-model.V[left] == p.final_time/(3*p.n)*(f_left+4*f_mid+f_right)


def midpoint_V_rule(model, i):
    p = model.params
    left, middle, right = 2*i, 2*i+1, 2*i+2
    f_left = rates(model, left)[2]
    f_right = rates(model, right)[2]
    return model.V[middle] == (model.V[left]+model.V[right])/2+p.final_time/(4*p.n)*(f_left-f_right)


def heading_dynamics_rule(model, i):
    p = model.params
    left, middle, right = 2*i, 2*i+1, 2*i+2
    f_left = rates(model, left)[3]
    f_mid = rates(model, middle)[3]
    f_right = rates(model, right)[3]
    return model.chi[right]-model.chi[left] == p.final_time/(3*p.n)*(f_left+4*f_mid+f_right)


def midpoint_chi_rule(model, i):
    p = model.params
    left, middle, right = 2*i, 2*i+1, 2*i+2
    f_left = rates(model, left)[3]
    f_right = rates(model, right)[3]
    return model.chi[middle] == (model.chi[left]+model.chi[right])/2+p.final_time/(4*p.n)*(f_left-f_right)


def mass_dynamics_rule(model, i):
    p = model.params
    left, middle, right = 2*i, 2*i+1, 2*i+2
    f_left = rates(model, left)[4]
    f_mid = rates(model, middle)[4]
    f_right = rates(model, right)[4]
    return model.mass[right]-model.mass[left] == p.final_time/(3*p.n)*(f_left+4*f_mid+f_right)


def midpoint_mass_rule(model, i):
    p = model.params
    left, middle, right = 2*i, 2*i+1, 2*i+2
    f_left = rates(model, left)[4]
    f_right = rates(model, right)[4]
    return model.mass[middle] == (model.mass[left]+model.mass[right])/2+p.final_time/(4*p.n)*(f_left-f_right)

# Controles


def midpoint_thrust_rule(model, i):
    return model.thrust[2*i+1] == (model.thrust[2*i]+model.thrust[2*i+2])/2


def midpoint_bank_rule(model, i):
    return model.bank[2*i+1] == (model.bank[2*i]+model.bank[2*i+2])/2

# Constraints de vuelo


def minimum_thrust_rule(model, i):
    return model.thrust[i]*eta(model.V[i]) >= minimum_fuel(model.params.h)


def maximum_speed_rule(model, i):
    return calibrated_speed(model.V[i], model.params.h)/knots_to_ms <= aircraft['V_MO_kn']


def minimum_speed_rule(model, i):
    return calibrated_speed(model.V[i], model.params.h)/knots_to_ms >= 1.3*aircraft['Vstall_CR_kn']*pyo.sqrt(model.mass[i]/aircraft['m_ref']/pyo.cos(model.bank[i]))


def buffet_rule(model, i):
    mach = model.V[i]/model.params.sound
    return aircraft['Clbo_M0']*mach**2-aircraft['K']*mach**3 >= model.mass[i]*g/(pyo.cos(model.bank[i])*0.583*air_pressure(model.params.h)*S)

# Contorno


def initial_longitude_rule(model):
    return model.lon[0] == model.params.lon0


def final_longitude_rule(model):
    return model.lon[model.params.n] == model.params.lonf


def initial_latitude_rule(model):
    return model.lat[0] == model.params.lat0


def final_latitude_rule(model):
    return model.lat[model.params.n] == model.params.latf


def initial_speed_rule(model):
    return model.V[0] == model.params.V0


def initial_mass_rule(model):
    return model.mass[0] == model.params.mass0

# Objetivo


def objective_rule(model):
    return model.params.mass0 - model.mass[model.params.n]
