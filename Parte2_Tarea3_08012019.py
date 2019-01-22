# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 16:37:21 2018

@author: AZS
"""


import numpy as np
import pandas as pd
import math 
from ortools.linear_solver import pywraplp
import random
import itertools
import datetime


def euclid_distance(coor_x,coor_y,opts):
    """    
        Distancia euclidea entre puntos.
    """
    
    dist ={}
    dist =[[int(np.sqrt((coor_x[i]-coor_x[j])*(coor_x[i]-coor_x[j])+
                  (coor_y[i]-coor_y[j])*(coor_y[i]-coor_y[j])))] 
                            for i in opts
                                for j in opts]
    dist_matrix = np.asarray(dist).reshape(len(coor_x), len(coor_y))
    return dist_matrix


# IDEA 0_______________________________________________________________________
def tsp_brute_force(ncities,city_names,dist_matrix):
    """
        Algoritmo de fuerza bruta
    """
    
    t1_now = datetime.datetime.now()
    n =ncities
    minLength = math.inf
    minTour = []
    for tour in itertools.permutations(list(range(1,n))):
        fr = 0
        length = 0
        count = 0
        while count < n-1:
            to = tour[count]
            length += dist_matrix[fr][to]
            fr = to
            count += 1
        length += dist_matrix[fr][0]
        if length < minLength:
            minLength = length
            minTour = tour
    minTour = (0,) + minTour + (0,)
    print('La ruta más corta es:', minTour)
    print('Tiene un tamaño de:', minLength , 'km')
    t2_now = datetime.datetime.now()
    dt = t2_now-t1_now
    print('delta time: ',dt.microseconds / 1000)
    return None

# IDEA 1_v0____________________________________________________________________
def tsp_mtzI(ncities,city_names,dist_matrix):
    """
        ATSP basado en Miller-Tucker-Zemlin (1960) (MTZ)
    """
    
    solver = pywraplp.Solver('cities', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    
    opciones =range(ncities)
    
    #Set de variables binarias
    x = {}
    for i in opciones:
        for j in opciones:
             x[i,j] = solver.BoolVar('x[%i,%i]' % (i,j))
             
             
    level = {}
    for i in opciones:
            level[i] = solver.IntVar(0.0, solver.infinity(), 'level[%i]' % (i))
    
    #F. Objetivo
    solver.Minimize(solver.Sum([dist_matrix[i][j] * x[i,j] for i in opciones
                                                      for j in opciones if i!=j]))
    
    #Restricciones
    for i in opciones:
        solver.Add(solver.Sum([x[i, j] for j in opciones if i!=j]) == 1)
    
    for j in opciones:
        solver.Add(solver.Sum([x[i, j] for i in opciones if i!=j]) == 1)  
    
    #Eliminar Subturs; 
    for i in opciones:
        for j in opciones:
            if (i>0 and j>0 and i!=j): #and i!=j
                solver.Add(level[j] >= level[i] + x[i,j] - (ncities-2) * (1-x[i,j]))
             
    conver = solver.Solve()                
    sol = solver
                    
    return [sol,x,conver]
  
   
# IDEA 1_______________________________________________________________________
def tsp_mtzII(ncities,city_names,dist_matrix):
    """
        ATSP basado (mejorado respecto a tsp_mtz) en Miller-Tucker-Zemlin 
        (1960) (MTZ)
    """    
    solver = pywraplp.Solver('cities', 
                             pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    
    opciones =range(ncities)
    #Set de variables binarias
    x = {}
    for i in opciones:
        for j in opciones:
             x[i,j] = solver.BoolVar('x[%i,%i]' % (i,j))
    #Set de variables enteras 
    level = {}
    for i in opciones:
            level[i] = solver.IntVar(0.0, solver.infinity(), 'level[%i]' % (i))
    
    #F. Objetivo
    solver.Minimize(
            solver.Sum([dist_matrix[i][j] * x[i,j] 
                    for i in opciones for j in opciones if i!=j]))
    
    #Restricciones
    for i in opciones:
        solver.Add(solver.Sum([x[i, j] for j in opciones if i!=j]) == 1)
    
    for j in opciones:
        solver.Add(solver.Sum([x[i, j] for i in opciones if i!=j]) == 1)  
    
    #Eliminar Subturs; 
    for i in opciones:
        for j in opciones:
            if (i>0 and j>0 and i!=j):
                solver.Add(level[j] >= level[i] + x[i,j] - (ncities-2) * 
                                       (1-x[i,j]) + (ncities-3) * x[j,i])
                
    conver = solver.Solve()                
    sol = solver
                    
    return [sol,x,conver]

 

# IDEA 2_______________________________________________________________________
def tsp_precedence(ncities,city_names,dist_matrix):
    """
        ATSP con variables de Precedencia
    """
    solver = pywraplp.Solver('cities', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)   

    opciones =range(ncities)    
    #Set de variables binarias 
    x = {}
    for i in opciones:
        for j in opciones:
             x[i,j] = solver.BoolVar('x[%i,%i]' % (i,j))
    #Set de variables binarias              
    preced = {}
    for i in opciones:
        for j in opciones:
             preced[i,j] = solver.BoolVar('preced[%i,%i]' % (i,j))
    
    #F. Objetivo
    solver.Minimize(solver.Sum([dist_matrix[i][j] * x[i,j] for i in opciones
                                                      for j in opciones if i!=j]))
    
    #Restricciones
    for i in opciones:
        solver.Add(solver.Sum([x[i, j] for j in opciones if i!=j]) == 1)
    
    for j in opciones:
        solver.Add(solver.Sum([x[i, j] for i in opciones if i!=j]) == 1)  
    
    #Eliminar Subturs; 
    for i in opciones:
        for j in opciones:
            if (i>0 and j>0 and i!=j):
                #antisimetría
                solver.Add(preced[i,j] + preced[j,i] == 1)
                #compatibilidad/coherencia
                solver.Add(x[i,j] <= preced[i,j])
                #transitividad
                for k in opciones: 
                    if (k>0 and k!=i and k!=j):        
                        solver.Add(preced[i,j] + preced[j,k] <= preced[i,k] + 1)
                
                
    conver = solver.Solve()                
    sol = solver
                    
    return [sol,x,conver]


  
# IDEA 4_______________________________________________________________________
def tsp_mcf(ncities,city_names,dist_matrix):
    """
        ATSP con multiflujo basado en Wong (1980) and Claus (1984)    
    """
    solver = pywraplp.Solver('cities', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    opciones =range(ncities)    
    #Set de variables binarias 
    x = {}
    for i in opciones:
        for j in opciones:
             x[i,j] = solver.BoolVar('x[%i,%i]' % (i,j))
    #Set de variables enteras              
    flow = {}
    for i in opciones:        
        for j in opciones:
            for k in opciones:
                 flow[i,j,k] = solver.BoolVar('flow[%i,%i,%i]' % (i,j,k))
    
    #F. Objetivo
    solver.Minimize(solver.Sum([dist_matrix[i][j] * x[i,j] for i in opciones
                                                              for j in opciones if i!=j]))
    
    #Restricciones
    for i in opciones:
        solver.Add(solver.Sum([x[i, j] for j in opciones if i!=j]) == 1)
    
    for j in opciones:
        solver.Add(solver.Sum([x[i, j] for i in opciones if i!=j]) == 1)  
    
    #Eliminar Subturs; 
    for k in opciones:
        if (k>0):
            solver.Add(solver.Sum([flow[0,j,k] for j in opciones if j!=0]) == 1)  
            for j in opciones:
                if (j>0):
                    solver.Add(flow[j,0,k] == 0)                    
            solver.Add(solver.Sum([flow[j,k,k] for j in opciones if j!=k]) == 1)  
            for j in opciones:
                if (j!=k):
                    solver.Add(flow[k,j,k] == 0)   #(16)
            for i in opciones:
                if (i!=k and i!=0): 
                    solver.Add(solver.Sum([flow[j,i,k] - flow[i,j,k] for j in opciones if (j!=i)]) == 0) #(15)
            for i in opciones:        
                for j in opciones:
                    if (i!=j):
                        solver.Add(flow[i,j,k] <= x[i,j])  #(16)

    conver = solver.Solve()                
    sol = solver
                    
    return [sol,x,conver]
#______________________________________________________________________________
 
# IDEA SSB_____________________________________________________________________ 
def tsp_ssb(ncities,city_names,dist_matrix):
    """
        ATSP con relajación LP basado en la formalación (L2ATSPxy) de Srin (2005)
    """
    solver = pywraplp.Solver('cities', 
                             pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    opciones =range(ncities)    
    #Set de variables binarias 
    x = {}
    for i in opciones:
        for j in opciones:
             x[i,j] = solver.BoolVar('x[%i,%i]' % (i,j))
    #Set de variables binarias 
    d = {}
    for i in opciones:
        for j in opciones:
            d[i,j] = solver.BoolVar('d[%i,%i]' % (i,j))             
    
    #F. Objetivo
    solver.Minimize(
            solver.Sum([dist_matrix[i][j] * x[i,j] 
                for i in opciones
                    for j in opciones if i!=j]))
    
    #Restricciones
    for i in opciones:
        solver.Add(solver.Sum([x[i, j] for j in opciones if i!=j]) == 1)
    
    for j in opciones:
        solver.Add(solver.Sum([x[i, j] for i in opciones if i!=j]) == 1) 

    #Eliminar Subturs; 
    for i in opciones:
        if (i>0):
            for j in opciones:
                if (j>0):
                    solver.Add(d[i,j] - x[i,j] >= 0)
    for i in opciones:
        if (i>0):
            for j in opciones:
                if (j>0 and i!=j):
                    solver.Add(d[i,j] + d[j,i] == 1)
    for j in opciones:
        if (j>0):
            solver.Add(x[1,j] + x[j,1] <= 1)

    for i in opciones:
        if (i>0):
            for j in opciones:
                if (j>0):
                    for k in opciones:
                        if (k>0):
                            solver.Add(x[i,j] + d[j,k] + 
                                       x[k,j] + d[k,i] + x[i,k] <= 2) 

    conver = solver.Solve()                
    sol = solver                    
    return [sol,x,conver]



# IDEA 5_______________________________________________________________________ 
def tsp_subtours(ncities,city_names,dist_matrix):
    """
        ATSP con eliminación de subtour
    """
    
    solver = pywraplp.Solver('cities', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    
    opciones =range(ncities)
    
    #Set de variables binarias 
    x = {}
    for i in opciones:
        for j in opciones:
             x[i,j] = solver.BoolVar('x[%i,%i]' % (i,j))
    #Set de variables entera             
    level = {}
    for i in opciones:
            level[i] = solver.IntVar(0.0, solver.infinity(), 'level[%i]' % (i))
    
    #Set de variables enteras
    NEXTC = {}
    for i in opciones:
            NEXTC[i] = solver.IntVar(0.0, solver.infinity(), 'NEXTC[%i]' % (i))
    
    #F. Objetivo
    solver.Minimize(solver.Sum([dist_matrix[i][j] * x[i,j] for i in opciones
                                                      for j in opciones if i!=j]))
    
    #Restricciones
    for i in opciones:
        solver.Add(solver.Sum([x[i, j] for j in opciones if i!=j]) == 1)
    
    for j in opciones:
        solver.Add(solver.Sum([x[i, j] for i in opciones if i!=j]) == 1)  
    
    #Eliminar Subturs; 
    mips = 1       
    print('Convergence = %d' % solver.Solve())
    print('Valor óptimo para la F. Objetivo = %d' % solver.Objective().Value())
    break_subtour(solver,x,mips,ncities)
    print('Convergencia = %d' % solver.Solve())
    print('Valor óptimo para la F. Objetivo = %d' % solver.Objective().Value())
    
    conver = solver.Solve()                
    sol = solver
                    
    return [sol,x,conver]


def break_subtour(solver,x,mips,ncities):
    """
        Rotura de subtour
    """
    
    opciones = range(ncities)
    ALLCITIES =[]
    NEXTC = {}
    for i in opciones:
            NEXTC[i] = int(round(sum([j*x[i,j].solution_value() for j in opciones])))       
      
    #Obtener el subtour que contiene al primer nodo
    TOUR=[]
    first=0
   
    while True:
        TOUR.append(first)
        first = NEXTC[first]
        if first == 0:
            break
    size = len(TOUR)

    #Encontrar el subtour más pequeño
    if (size < ncities):
        SMALLEST=TOUR        
        if (size > 2):
            ALLCITIES = TOUR
            for i in opciones:                
                if (i not in ALLCITIES):
                    TOUR=[]
                    first = i                  
                    while True:
                        TOUR.append(first)
                        first = NEXTC[first]
                        if first == i: 
                            break 
                        
                        ALLCITIES.append(TOUR)                       
                        #ya tenemos todos los subtours como grafos sueltos
                        if (len(TOUR)<size):
                            SMALLEST = TOUR
                            size = len(SMALLEST)
                        if (size==2):
                            break
        #Añado una restriccion de rotura de subtour
        print ('añadimos restriccion ', SMALLEST)
        solver.Add(solver.Sum([x[i,j] for i in SMALLEST for j in SMALLEST if i!=j]) <= len(SMALLEST)-1)
        conv = solver.Solve()
        print('Convergence = %d' % conv)
        print(mips, '  ', ' Cost: ', solver.Objective().Value(), 'size = ', size)
        #Llamada recursiva a break_subtour()
        break_subtour(solver,x,mips,ncities)
    
    return None


def get_lst_graph(sol_arcs):
    """
        Convertir el resultado de la routa en un lista
    """
    
    grafo_lst=[] 
    grafo_lst.append(sol_arcs[0][0])
    df_sol_arcs = pd.DataFrame(sol_arcs)
    for i in range(len(df_sol_arcs)):
        grafo_lst.append(df_sol_arcs[1].iloc[grafo_lst[-1]]) 

    return grafo_lst


def benchmark_tsp(n_min_nodes,n_max_nodes):
    """
        Comparativa de los diferentes metodos de calculo para TSP
    """
    
    #Generación de datos aleatorios para las coordenadas de los nodos
    tot_res = []
    for n in range(n_min_nodes,n_max_nodes+1):
        print(n)
        random.seed(4)
        ncities = n
        opciones =range(ncities)
        coor_x = [random.randint(1,ncities**2+1) for i in opciones] 
        coor_y = [random.randint(1,ncities**2+1) for i in opciones] 
        
        city_names = range(ncities)
        
        dist ={}
        dist =[[int(np.sqrt(
                  (coor_x[i]-coor_x[j])*(coor_x[i]-coor_x[j])+
                  (coor_y[i]-coor_y[j])*(coor_y[i]-coor_y[j])))] 
                            for i in opciones
                                for j in opciones]
        dist_matrix = np.asarray(dist).reshape(len(coor_x), len(coor_y))
                
        sol = {}
#        sol[0] = tsp_brute_force(ncities,city_names,dist_matrix) #
        sol[0] = tsp_mtzI(ncities,city_names,dist_matrix)
        sol[1] = tsp_mtzII(ncities,city_names,dist_matrix)
        sol[2] = tsp_precedence(ncities,city_names,dist_matrix)
        sol[3] = tsp_mcf(ncities,city_names,dist_matrix)
        sol[4] = tsp_ssb(ncities,city_names,dist_matrix)
        sol[5] = tsp_subtours(ncities,city_names,dist_matrix)
        
        
        res = []
        for s in range(len(sol)):
            solver = sol[s][0]
            var = sol[s][1]
            conver = sol[s][2]
            res.append([conver,solver.NumVariables(),solver.NumConstraints(),
                        solver.Objective().Value(),solver.WallTime(),
                        solver.Iterations(),solver.nodes()])
 
        tot_res.append([n,res])

    return tot_res


###############################################################################
###############################################################################
###############################################################################


def main():  
    """
        Desde el main, primero se generan aleatoriamente con semilla las dos 
        coordenadas x e y. Después con la función "euclid_distance" se calcula
        las distancias euclídeas entre nodos o ciudades.
        
        Depués se tratará de hallar el recorrido minimo entre ciudades o nodos, 
        visitando cada nodo una única vez. 
        
        Para hallar la solución del problema se utilizan diferentes 
        formulaciónes del modelo y se optienen las soluciones. Todas las 
        halladas por las diferentes formulaciones implementadas convergen y 
        llegan al mismo óptimo en cuanto a la ruta. Existen diferencias en el 
        tiempo computacional, nº de variables y restricciones utilizadas.
        
        La llamada a la función "tsp_brute_force" esta comentada ya que para 
        grafos con cierto tamaño el tiempo de cálculo es elevado. No obstante, 
        a modo ilustrativo se mantiene expuesto el código.
        
        Una vez cálculado el ejemplo introducido, especificando en nº de nodos,
        (por ejemplo: con 20 nodos) en "ncities", los resultados se almacenan
        en la variable "df_res".
        
        Además, en el código existe la posibilidad de realizar un cálculo 
        comparativo o bechmarking especificando un rango de nº de nodos 
        mediante la función "benchmark_tsp". Los resulados se obtendrán para
        la serie de nodos especificados y para cada tipo de formulación. 
        
        Después, tras finalizar, los resultados estarán guardados en la 
        variable "tot_res" y los resultados resumen en la variable 
        "df_each_res". Por defecto esta función calcula entre 5 y 10 nodos para 
        no incrementar mucho el tiempo de cálculo.
        
    """
    
    #_____________Datos aleatorios_____________________________________________
    random.seed(4)
    ncities = 20
    opciones =range(ncities)
    coor_x = [random.randint(1,ncities**2+1) for i in opciones] 
    coor_y = [random.randint(1,ncities**2+1) for i in opciones] 
    city_names = range(ncities)
    dist_matrix = euclid_distance(coor_x,coor_y,opciones)
    
    #__________________________________________________________________________
    sol = {}
    #sol[0] = tsp_brute_force(ncities,city_names,dist_matrix)
    sol[0] = tsp_mtzI(ncities,city_names,dist_matrix)
    sol[1] = tsp_mtzII(ncities,city_names,dist_matrix)
    sol[2] = tsp_precedence(ncities,city_names,dist_matrix)
    sol[3] = tsp_mcf(ncities,city_names,dist_matrix)
    sol[4] = tsp_ssb(ncities,city_names,dist_matrix)
    sol[5] = tsp_subtours(ncities,city_names,dist_matrix)
    
    
    res = []
    for s in range(len(sol)):
        solver = sol[s][0]
        var = sol[s][1]
        conver = sol[s][2]
        res.append([conver,solver.NumVariables(),solver.NumConstraints(),
                    solver.Objective().Value(),solver.WallTime(),
                    solver.Iterations(),solver.nodes()])
    
    df_res = pd.DataFrame(res, columns= ['conver','n_var','n_constr',
                                         'opt','time','iter','nodes'])
    print ('Los resultados de tsp son: \n\n',df_res)
    
    #____ Presentamos la ruta del problema ATSP________________________________
    print('\nLa distancia entre nodos es: \n')
    sol_arcs =[]
    for i in range(ncities):
        for j in range(ncities):
            if var[i, j].solution_value() > 0:
                print('Desde %s hasta %s.  km = %d' % (city_names[i],
                                                       city_names[j],
                                                       dist_matrix[i][j]))
                sol_arcs.append([city_names[i],city_names[j]])
    print()
    print('La ruta es: ',get_lst_graph(sol_arcs))
    
    
    #_______ Benchmark ATSP____________________________________________________
    #calculamos TSP para grafos de entre 5 y 20 nodos
    tot_res = benchmark_tsp(n_min_nodes=5,n_max_nodes=10)
    
    #obtenemos el resumen de los resultados
    df_each_res = pd.DataFrame([])
    for res in tot_res:
        df_each_res_aux = pd.DataFrame(res[1:][0])
        df_each_res_aux['n_nodes'] = res[0]
        df_each_res=df_each_res.append(df_each_res_aux)
    
    
    cab_arr = ['Conver','NumVariables','NumConstraints','ObjectiveValue',
               'Time','Iterations','nodes','NumNodes']
    df_each_res.columns = [cab_arr]
    


if __name__ == '__main__':
    print('')    
    main()
    print('FIN')














    
    
   