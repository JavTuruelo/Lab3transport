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
Re = 6_371_000.0

# ISA constants (same as tasks 1 and 2)
T0 = 288.15
p0 = 101325
beta_T = -0.0065
h_trop = 11000
T_isa_trop = T0 + beta_T * h_trop  # BADA (3.1-14)
rho0 = p0 / (R * T0)

# Funciones para las restricciones
# Altitude h is a fixed number (m) in task 3, so the exact ISA with if/else is valid.


def check_altitude(h):
    if not 0 <= h <= 20000:
        raise ValueError('This ISA implementation supports altitudes from 0 to 20 km.')


def temperature(h):
    # Temperature for ISA (delta T = 0), BADA (3.1-13) and (3.1-16).
    check_altitude(h)
    if h <= h_trop:
        return T0 + beta_T * h
    return T_isa_trop


def air_pressure(h):
    # Pressure, BADA (3.1-18) to (3.1-20).
    check_altitude(h)
    if h <= h_trop:
        return p0 * (temperature(h) / T0)**(-g / (beta_T * R))
    p_trop = p0 * (T_isa_trop / T0)**(-g / (beta_T * R))
    return p_trop * math.exp(-g * (h - h_trop) / (R * T_isa_trop))


def eta(V):
    return aircraft['C_f1'] / 60000 * (1 + V / knots_to_ms / aircraft['C_f2'])


def maximum_thrust(h):
    h_ft = h / 0.3048
    return aircraft['C_tc1'] * (1 - h_ft / aircraft['C_tc2'] + aircraft['C_tc3'] * h_ft**2)


def minimum_fuel(h):
    h_ft = h / 0.3048
    return aircraft['C_f3'] / 60 * (1 - h_ft / aircraft['C_f4'])


def calibrated_speed(V_TAS, h):
    # BADA (3.1-24) and (3.1-25). Both speeds are in m/s.
    p = air_pressure(h)
    rho = p / (R * temperature(h))
    mu = (gamma_air - 1) / gamma_air
    return (
        (2 * p0 / (mu * rho0))
        * ((1 + (p / p0) * ((1 + (mu * rho / (2 * p)) * V_TAS**2)**(1 / mu) - 1))**mu - 1)
    )**(1 / 2)


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
