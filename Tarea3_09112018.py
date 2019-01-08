# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 16:37:21 2018

@author: AZENZANO
"""


#
import numpy as np
#librerias coin ar
from pulp import *
#libreria google
from ortools.linear_solver import pywraplp


#matriz de distancia


#matriz de distancias hardcodeada
D = (  
1,2,7,1,3,9,1,4,8,1,5,4,1,6,3,1,7,7,1,8,7,1,9,8,1,10,5,
2,1,7,2,3,4,2,4,4,2,5,9,2,6,8,2,7,4,2,8,2,2,9,1,2,10,8,
3,1,9,3,2,4,3,4,8,3,5,9,3,6,8,3,7,8,3,8,2,3,9,3,3,10,11,
4,1,8,4,2,4,4,3,8,4,5,11,4,6,10,4,7,1,4,8,6,4,9,5,4,10,6,
5,1,4,5,2,9,5,3,9,5,4,11,5,6,1,5,7,11,5,8,8,5,9,9,5,10,9,
6,1,3,6,2,8,6,3,8,6,4,10,6,5,1,6,7,9,6,8,7,6,9,8,6,10,8,
7,1,7,7,2,4,7,3,8,7,4,1,7,5,11,7,6,9,7,8,6,7,9,5,7,10,5,
8,1,7,8,2,2,8,3,2,8,4,6,8,5,8,8,6,7,8,7,6,8,9,1,8,10,9,
9,1,8,9,2,1,9,3,3,9,4,5,9,5,9,9,6,8,9,7,5,9,8,1,9,10,9,
10,1,5,10,2,8,10,3,11,10,4,6,10,5,9,10,6,8,10,7,5,10,8,9,10,9,9);
D = np.asarray(D).reshape(90, 3)




import random
random.seed(4)


#x = [random.randint(1,ncities**2+1) for i in opciones] 
#y = [random.randint(1,ncities**2+1) for i in opciones] 
#
#dist = []
#for i in opciones:
#    for j in opciones:
#        if i<j:
#            dist[i,j] = int(np.sqrt((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j])))
#
#
#for i in opciones:
#    for j in opciones:
#        if i<j:
#            dist[j,i] = int(np.sqrt((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j])))
#
#
#
#
#for i in opciones:
#    for j in opciones:
#        if i<j:
#            dist = int(np.sqrt((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j])))



ncities = 4
opciones  = range(ncities)

cost  =  [[90, 76, 75, 70],
          [35, 85, 55, 65],
          [125, 95, 90, 105],
          [45, 110, 95, 115]]

solver = pywraplp.Solver('cities', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)


# variable lógica: es binario
x = {}
for i in opciones:
    for j in opciones:
         x[i,j] = solver.BoolVar('x[%i,%i]' % (i,j))
         
level = {}
for i in opciones:
        level[i] = solver.IntVar(0.0, solver.infinity(), 'level[%i]' % (i))



# Objective
solver.Minimize(solver.Sum([cost[i][j] * x[i,j] for i in opciones
                                                  for j in opciones]))

# Constraints
for i in opciones:
    solver.Add(solver.Sum([x[i, j] for j in opciones if i!=j]) == 1)

for j in opciones:
    solver.Add(solver.Sum([x[i, j] for i in opciones if i!=j]) == 1)  

#Eliminar Subturs; 
for i in opciones:
    for j in opciones:
        if i>0 and j>0:
            print (i,j)
            solver.Add(level[j] >= level[i] + x[i,j] - (ncities-2) * (1-x[i,j]))
#            solver.Add(level[j] >= level[i] + x[i,j] - (ncities) * (1-x[i,j]))
       
 
    
# resolvemos:  
sol = solver.Solve()

print('Costo total = ', solver.Objective().Value())
print()
for i in opciones:
    for j in opciones:
        if x[i, j].solution_value() > 0:
            print('Desde %d hasta %d.  Costo = %d' % (i,j,cost[i][j]))
print()
print("Time = ", solver.WallTime(), " milliseconds") 
    
    

#############################################################################

level = [solver.IntVar(0.0, solver.infinity(), 'level[%i]' % (i)) for i in opciones]
level
#############################################################################































    
    
    
    
cost  =  [[90, 76, 75, 70],
      [35, 85, 55, 65],
      [125, 95, 90, 105],
      [45, 110, 95, 115],
      [60, 105, 80, 75],
      [45, 65, 110, 95]]
num_workers = len(cost)
num_tasks   = len(cost[1])

solver = pywraplp.Solver('ProblemaAsignacion2dimensional',
                           pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

x = {}
for i in range(num_workers):
    for j in range(num_tasks):
        x[i, j] = solver.BoolVar('x[%i,%i]' % (i, j))

  # Objective
solver.Minimize(solver.Sum([cost[i][j] * x[i,j] for i in range(num_workers)
                                              for j in range(num_tasks)]))

  # Constraints

  # Each worker is assigned to at most 1 task.

for i in range(num_workers):
    solver.Add(solver.Sum([x[i, j] for j in range(num_tasks)]) <= 1)

  # Each task is assigned to exactly one worker.

for j in range(num_tasks):
    solver.Add(solver.Sum([x[i, j] for i in range(num_workers)]) == 1)

  # resolvemos:  
sol = solver.Solve()

print('Costo total = ', solver.Objective().Value())
print()
for i in range(num_workers):
    for j in range(num_tasks):
        if x[i, j].solution_value() > 0:
            print('Trabajador %d asignado a la tarea %d.  Costo = %d' % (i,j,cost[i][j]))
print()
print("Time = ", solver.WallTime(), " milliseconds")
    
    
##################################################################################
    
solver = pywraplp.Solver('RunIntegerExampleNaturalLanguageAPI',
                           pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

infinity = solver.infinity()
  # x1 and x2 are integer non-negative variables.
x1 = solver.IntVar(0.0, infinity, 'x1')
x2 = solver.IntVar(0.0, infinity, 'x2')

solver.Minimize(x1 + 2 * x2)
solver.Add(3 * x1 + 2 * x2 >= 17)

SolveAndPrint(solver, [x1, x2])

def SolveAndPrint(solver, variable_list):
  """Solve the problem and print the solution."""
  print(('Number of variables = %d' % solver.NumVariables()))
  print(('Number of constraints = %d' % solver.NumConstraints()))

  result_status = solver.Solve()

  # The problem has an optimal solution.
  assert result_status == pywraplp.Solver.OPTIMAL

  # The solution looks legit (when using solvers others than
  # GLOP_LINEAR_PROGRAMMING, verifying the solution is highly recommended!).
  assert solver.VerifySolution(1e-7, True)

  print(('Problem solved in %f milliseconds' % solver.wall_time()))

  # The objective value of the solution.
  print(('Optimal objective value = %f' % solver.Objective().Value()))

  # The value of each variable in the solution.
  for variable in variable_list:
    print(('%s = %f' % (variable.name(), variable.solution_value())))

  print('Advanced usage:')
print(('Problem solved in %d branch-and-bound nodes' % solver.nodes()))



