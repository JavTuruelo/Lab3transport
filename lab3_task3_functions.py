"""Task 3: aircraft data, constants and auxiliary (non-rule) functions.

These functions are used by the Pyomo rules in lab3_task3_rules.py and by the
main script lab3_task3_horizontal.py. Case-dependent data (wind, density, ...)
is read from model.params, which the main script attaches to each model.
"""
import math
import pyomo.environ as pyo

# Datos referencia del avion

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
Re = 6_371_000.0

# Funciones para las restricciones


def temperature(h):
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


def drag(V, mass, bank, rho):
    qS = 0.5 * rho * V**2 * S
    lift = mass * g / pyo.cos(bank)
    return qS * aircraft['CD0_CR'] + aircraft['CD2_CR'] * lift**2 / qS


def rates(model, i):
    """State derivatives (lon, lat, V, chi, mass) at stage i."""
    p = model.params
    return ((model.V[i]*pyo.cos(model.chi[i])+p.wx)/(Re*pyo.cos(model.lat[i])),
            (model.V[i]*pyo.sin(model.chi[i])+p.wy)/Re,
            (model.thrust[i]-drag(model.V[i], model.mass[i], model.bank[i], p.rho))/model.mass[i],
            g*pyo.tan(model.bank[i])/model.V[i],
            -model.thrust[i]*eta(model.V[i]))
