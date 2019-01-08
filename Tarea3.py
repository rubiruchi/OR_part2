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
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

#matriz de distancia



def create_data_model():
  """Stores the data for the problem"""

# Cities
    city_names = ["New York", "Los Angeles", "Chicago", "Minneapolis", "Denver", "Dallas", "Seattle",
                "Boston", "San Francisco", "St. Louis", "Houston", "Phoenix", "Salt Lake City"]
    # Distance matrix
    dist_matrix = [
        [   0, 2451,  713, 1018, 1631, 1374, 2408,  213, 2571,  875, 1420, 2145, 1972], # New York
        [2451,    0, 1745, 1524,  831, 1240,  959, 2596,  403, 1589, 1374,  357,  579], # Los Angeles
        [ 713, 1745,    0,  355,  920,  803, 1737,  851, 1858,  262,  940, 1453, 1260], # Chicago
        [1018, 1524,  355,    0,  700,  862, 1395, 1123, 1584,  466, 1056, 1280,  987], # Minneapolis
        [1631,  831,  920,  700,    0,  663, 1021, 1769,  949,  796,  879,  586,  371], # Denver
        [1374, 1240,  803,  862,  663,    0, 1681, 1551, 1765,  547,  225,  887,  999], # Dallas
        [2408,  959, 1737, 1395, 1021, 1681,    0, 2493,  678, 1724, 1891, 1114,  701], # Seattle
        [ 213, 2596,  851, 1123, 1769, 1551, 2493,    0, 2699, 1038, 1605, 2300, 2099], # Boston
        [2571,  403, 1858, 1584,  949, 1765,  678, 2699,    0, 1744, 1645,  653,  600], # San Francisco
        [ 875, 1589,  262,  466,  796,  547, 1724, 1038, 1744,    0,  679, 1272, 1162], # St. Louis
        [1420, 1374,  940, 1056,  879,  225, 1891, 1605, 1645,  679,    0, 1017, 1200], # Houston
        [2145,  357, 1453, 1280,  586,  887, 1114, 2300,  653, 1272, 1017,    0,  504], # Phoenix
        [1972,  579, 1260,  987,  371,  999,  701, 2099,  600, 1162,  1200,  504,   0]] # Salt Lake City
    
    len(dist_matrix)
    tsp_size = len(city_names)
    num_routes = 1
    depot = 0

#https://github.com/google/or-tools/blob/master/examples/python/tsp.py









import random
random.seed(4)

ncities = tsp_size
opciones  = range(ncities)

from_node = [random.randint(1,ncities**2+1) for i in opciones] 
to_node = [random.randint(1,ncities**2+1) for i in opciones] 
#
dist = []
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




#ncities = 4
opciones  = range(ncities)

#dist_matrix  =  [[90, 76, 75, 70],
#          [35, 85, 55, 65],
#          [125, 95, 90, 105],
#          [45, 110, 95, 115]]

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
solver.Minimize(solver.Sum([dist_matrix[i][j] * x[i,j] for i in opciones
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
            print('Desde %d hasta %d.  Costo = %d' % (i,j,dist_matrix[i][j]))
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



