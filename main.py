import pyomo
from pyomo.environ import *
import numpy as np
import math

N = 50

m = ConcreteModel()

m.n = Param(initialize = N-1, within = Integers)     # n = number of subintervals      
m.N = Set(initialize = range(0,value(m.n)+1))       # n+1 elements (nodes)
m.Ns = Set(initialize = range(0,value(m.n)))        # n elements (subintervals)



#### Definicion de parametros:

x1_0 = 0
x1_f = 1

x2_0 = 0
x2_f = 0

#####################################################################################
#### Defining Variables and parameters ---------------------------------
#####################################################################################
m.x1 = Var(m.N, within = Reals)
m.x2 = Var(m.N, within = Reals)
m.u = Var(m.N, within = Reals)

m.step = 1/m.n

#####################################################################################
###  Differential Constraints   --------------------------------
#####################################################################################
def x1dot_definition(model, i):
    return m.x1[i+1] - m.x1[i] == m.step*0.5*(m.x2[i+1] + m.x2[i])
m.x1dot_const = Constraint(m.Ns, rule = x1dot_definition)

def x2dot_definition(model, i):
    return m.x2[i+1] - m.x2[i] == m.step*0.5*(m.u[i+1] + m.u[i])
m.x2dot_const = Constraint(m.Ns, rule = x2dot_definition)

#####################################################################################
####  Objectivefunction  --------------------------------------------
#####################################################################################
def objecive_rule(model):
    return sum(m.step*0.5*(m.u[i+1]**2 + m.u[i]**2) for i in m.Ns)
m.obj = Objective(expr = objecive_rule, sense = minimize)


#####################################################################################
####  Boundary Constraints  --------------------------------------------
######################################################################################
m.x1_init_const = Constraint(expr = m.x1[0]          == 0 )
m.x1_end_const  = Constraint(expr = m.x1[value(m.n)] == 1 )
m.x2_init_const = Constraint(expr = m.x2[0]          == 0 )
m.x2_end_const  = Constraint(expr = m.x2[value(m.n)] == 0 )


m.display()


# -----------------------------
# SOLVE
# -----------------------------
solver = SolverFactory('ipopt')
solver.solve(m, tee=True)


# ######################################################################################
# #####-------------- Datos del Solver ----------------------
# #######################################################################################
# solver = 'ipopt'
# opt = SolverFactory(solver)
# opt.options['halt_on_ampl_error'] = 'yes'
# opt.options['max_iter'] =3000
# solver_manager = SolverManagerFactory('serial')
# results = solver_manager.solve(m, opt=opt, tee=True, timelimit=None)
   

########################################################################################


x1 = list(range(0,value(m.n)+1))
x2 = list(range(0,value(m.n)+1))
u  = list(range(0,value(m.n)+1))
time = list(range(0,value(m.n)+1))

for i in m.N:
    x1[i] = value(m.x1[i])
    x2[i] = value(m.x2[i])
    u[i]  = value(m.u[i])
    time[i] = value(m.step)*i

#####################################################################################
####  Imprimir en consola --------------------------------------------  
#####################################################################################

# print ('')
# print ('n           time             x1              x2   \n')

# for i in m.N:
#         print ('  {:02.0f}  {:12.2f} {:12.2f}  {:12.2f}  \n'.format(i,time[i], x1[i], x2[i]))

# #####################################################################################
# #Crear y sobreescribir output txt  ---------------------------------------
# #####################################################################################
# def creartxt():
#     archi = open('min_distancia.txt','w') #w crear archivo txt y lo abre para escribir write
#     archi.close()
    
# def grabartxt():
#     archi = open('min_distancia.txt','a') # a abre el archivo para escribir, y lo crea si no lo está
#     archi.write('n        x           y   \n')
#     for i in m.N:
#         archi.write('  {:02.0f}  {:12.2f}  {:12.2f}  \n'.format(i, x[i], y[i]))
# archivo = open("datos_moon_landing.txt", "r")

# creartxt()
# grabartxt()

# def leertxtenlista():
#     archi=open('min_distancia.txt','r')
#     lineas=archi.readlines()
#     for li in lineas:
#         print (li)
#     archi.close()
    
#####################################################################################
#### Gráficas  ---------------------------------------
#####################################################################################
import matplotlib.pyplot as plt
#pylab inline

plt.figure(1)
plt.plot(time, x1)
plt.grid(True)
plt.xlabel('Time')
plt.ylabel('x1')
plt.show()

plt.figure(2)
plt.plot(time, x2)
plt.grid(True)
plt.xlabel('Time')
plt.ylabel('x2')
plt.show()

plt.figure(3)
plt.plot(time, x1, label='x1')
plt.plot(time, x2, label='x2')
plt.grid(True)
plt.xlabel('Time')
plt.ylabel('x1, x2')
plt.legend()
plt.show()

plt.figure(4)
plt.plot(time, u)
plt.grid(True)
plt.xlabel('Time')
plt.ylabel('u')
plt.show()
