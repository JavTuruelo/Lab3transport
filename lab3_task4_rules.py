"""Vertical-flight equations and Pyomo rules for Task 4."""
import pyomo.environ as pyo

from lab3_functions import (
    R, S, aircraft, eta, g, gamma_air, knots_to_ms, maximum_thrust, minimum_fuel,
    air_pressure_smooth as air_pressure,
    calibrated_speed_smooth as calibrated_speed,
    rates_vertical as rates,
    temperature_smooth as temperature,
)


def distance_dynamics_rule(model, i):
    left, middle, right = 2*i, 2*i+1, 2*i+2
    f_left = rates(model, left)[0]
    f_mid = rates(model, middle)[0]
    f_right = rates(model, right)[0]
    return model.x[right]-model.x[left] == model.tf/(3*model.params.n)*(f_left+4*f_mid+f_right)


def midpoint_x_rule(model, i):
    left, middle, right = 2*i, 2*i+1, 2*i+2
    f_left = rates(model, left)[0]
    f_right = rates(model, right)[0]
    return model.x[middle] == (model.x[left]+model.x[right])/2+model.tf/(4*model.params.n)*(f_left-f_right)


def altitude_dynamics_rule(model, i):
    left, middle, right = 2*i, 2*i+1, 2*i+2
    f_left = rates(model, left)[1]
    f_mid = rates(model, middle)[1]
    f_right = rates(model, right)[1]
    return model.h[right]-model.h[left] == model.tf/(3*model.params.n)*(f_left+4*f_mid+f_right)


def midpoint_h_rule(model, i):
    left, middle, right = 2*i, 2*i+1, 2*i+2
    f_left = rates(model, left)[1]
    f_right = rates(model, right)[1]
    return model.h[middle] == (model.h[left]+model.h[right])/2+model.tf/(4*model.params.n)*(f_left-f_right)


def speed_dynamics_rule(model, i):
    left, middle, right = 2*i, 2*i+1, 2*i+2
    f_left = rates(model, left)[2]
    f_mid = rates(model, middle)[2]
    f_right = rates(model, right)[2]
    return model.V[right]-model.V[left] == model.tf/(3*model.params.n)*(f_left+4*f_mid+f_right)


def midpoint_V_rule(model, i):
    left, middle, right = 2*i, 2*i+1, 2*i+2
    f_left = rates(model, left)[2]
    f_right = rates(model, right)[2]
    return model.V[middle] == (model.V[left]+model.V[right])/2+model.tf/(4*model.params.n)*(f_left-f_right)


def flight_path_dynamics_rule(model, i):
    left, middle, right = 2*i, 2*i+1, 2*i+2
    f_left = rates(model, left)[3]
    f_mid = rates(model, middle)[3]
    f_right = rates(model, right)[3]
    return model.gamma[right]-model.gamma[left] == model.tf/(3*model.params.n)*(f_left+4*f_mid+f_right)


def midpoint_gamma_rule(model, i):
    left, middle, right = 2*i, 2*i+1, 2*i+2
    f_left = rates(model, left)[3]
    f_right = rates(model, right)[3]
    return model.gamma[middle] == (model.gamma[left]+model.gamma[right])/2+model.tf/(4*model.params.n)*(f_left-f_right)


def mass_dynamics_rule(model, i):
    left, middle, right = 2*i, 2*i+1, 2*i+2
    f_left = rates(model, left)[4]
    f_mid = rates(model, middle)[4]
    f_right = rates(model, right)[4]
    return model.mass[right]-model.mass[left] == model.tf/(3*model.params.n)*(f_left+4*f_mid+f_right)


def midpoint_mass_rule(model, i):
    left, middle, right = 2*i, 2*i+1, 2*i+2
    f_left = rates(model, left)[4]
    f_right = rates(model, right)[4]
    return model.mass[middle] == (model.mass[left]+model.mass[right])/2+model.tf/(4*model.params.n)*(f_left-f_right)


def midpoint_thrust_rule(model, i):
    return model.thrust[2*i+1] == (model.thrust[2*i]+model.thrust[2*i+2])/2


def midpoint_CL_rule(model, i):
    return model.CL[2*i+1] == (model.CL[2*i]+model.CL[2*i+2])/2


def maximum_thrust_rule(model, i):
    return model.thrust[i] <= maximum_thrust(model.h[i])


def minimum_thrust_rule(model, i):
    return model.thrust[i]*eta(model.V[i]) >= minimum_fuel(model.h[i])


def maximum_mach_rule(model, i):
    return model.V[i]**2 <= aircraft['M_MO']**2*gamma_air*R*temperature(model.h[i])


def maximum_speed_rule(model, i):
    return calibrated_speed(model.V[i],model.h[i])/knots_to_ms <= aircraft['V_MO_kn']


def minimum_speed_rule(model, i):
    return calibrated_speed(model.V[i],model.h[i])/knots_to_ms >= 1.3*aircraft['Vstall_CR_kn']*pyo.sqrt(model.mass[i]/aircraft['m_ref'])


def altitude_limit_rule(model, i):
    return model.h[i]/0.3048 <= aircraft['h_max_ft']+aircraft['G_w']*(aircraft['m_max']-model.mass[i])


def buffet_rule(model, i):
    mach = model.V[i]/pyo.sqrt(gamma_air*R*temperature(model.h[i]))
    # Smooth activation above 15000 ft avoids a discrete altitude switch.
    # Below the threshold the positive allowance relaxes this constraint;
    # the independent clean stall-speed constraint remains in force.
    allowance = 0.5/(1+pyo.exp((model.h[i]-15000*0.3048)/30))
    return aircraft['Clbo_M0']*mach**2-aircraft['K']*mach**3+allowance >= model.mass[i]*g/(0.583*air_pressure(model.h[i])*S)


def initial_distance_rule(model):
    return model.x[0] == 0


def final_distance_rule(model):
    return model.x[model.params.n] == model.params.distance


def initial_altitude_rule(model):
    return model.h[0] == model.params.initial_altitude


def final_altitude_rule(model):
    return model.h[model.params.n] == model.params.final_altitude


def initial_speed_rule(model):
    return model.V[0] == model.params.initial_speed


def final_speed_rule(model):
    return model.V[model.params.n] == model.params.final_speed


def initial_gamma_rule(model):
    return model.gamma[0] == model.params.initial_gamma


def final_gamma_rule(model):
    return model.gamma[model.params.n] == model.params.final_gamma


def initial_mass_rule(model):
    return model.mass[0] == model.params.mass0


def objective_rule(model):
    return model.params.CI*model.tf/model.params.cost_time_divisor+(model.params.mass0 - model.mass[model.params.n]) * model.params.cost_fuel_multiplier / 100
