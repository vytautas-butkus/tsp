# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 17:11:51 2016

@author: butkus
"""

from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functools import reduce
from scipy.optimize import differential_evolution, basinhopping
from pyswarm import pso

def get_distance (id1, id2):
    if id1==id2:
        return 0
        
    return np.sqrt((cities.ix[id1].x-cities.ix[id2].x)**2 + (cities.ix[id1].y-cities.ix[id2].y)**2)
    
def get_cost (data):
    return np.sum([dist[id1][id2] for id1, id2 in zip(data, np.roll(data,1))])

def plot_route (route, name):
    # create new figure, axes instances.
    fig=plt.figure(name)
    fig.clf()
    fig.add_axes([0.1,0.1,0.8,0.8])
    
    # setup mercator map projection.
    map = Basemap(llcrnrlon=20.84,llcrnrlat=53.83,urcrnrlon=26.94,urcrnrlat=56.53,\
                rsphere=(6378137.00,6356752.3142),\
                resolution='l',projection='merc',\
                lat_0=40.,lon_0=-20.,lat_ts=20.)
    
    map.drawcoastlines()
    
    map.drawcountries()
    map.fillcontinents()
    
    points = np.array([map(x[0], x[1]) for x in cities.ix[route][['latitude', 'longitude']].values])
    
    first = cities.ix[route[0]][['latitude', 'longitude']].values
    start = np.array([map(first[0], first[1])])
    
    points = np.concatenate((points,start), axis=0)
    
    
    map.plot(points[:,0], points[:,1], 'r-')
    map.plot(points[:,0], points[:,1], 'bo')
    
    plt.show()
    


def take_step (x):
    x1 = np.random.uniform(low = 1, high = n-1)
    x_temp = x[x1]
    x[x1] = x[x1+1]
    x[x1+1] = x_temp

def greedy_route ():
    cities_left = list(np.arange(1,n))
    route = [0]

    while len(cities_left) > 0:
        i = route[0]
        min_value = 10000;
        min_arg = i
        
        for j in cities_left:
            if (dist[i][j] < min_value):
                min_arg = j
                min_value = dist[i][j]

        route.insert(0, min_arg)
        cities_left.remove(min_arg)
            
    return route
    

cities = pd.read_table('LT_miestai_koordinates.txt')

n = len(cities)

dist = ([[(get_distance(id1,id2)) for id2 in range(n)] for id1 in range(n)])

#init_cost = get_cost(init_route)   
#print("Initial cost", init_cost, "kilometers")
#init_route = np.arange(0, n)    
#lb = [0]*n         
#ub = [100.0]*n    
#res = differential_evolution (get_cost, list(zip(lb,ub)), popsize = 50, maxiter = 100, disp=True)
#xopt, fopt = pso (get_cost, lb, ub, phip=0.4, phig=0.4, swarmsize=50, maxiter = 5000, debug=True)
#res = basinhopping (get_cost, init_route, niter=10000, T=500.0, disp=True)
    
groute = greedy_route()

#route from (C) Concorde:
oroute = [0,14,29,9,12,22,23,4,18,3,25,32,30,7,11,15,16,19,2,20,17,13,28,26,27,6,21,5,31,1,10,8,24]

plot_route(groute, "Greedy route")    
print("Greedy route cost:", get_cost(groute), "kilometers")

plot_route(oroute, "Optimal route")
print("Optimal route cost:", get_cost(oroute), "kilometers")




    
    
    
    
    
    
    
    
    
    
    
    





