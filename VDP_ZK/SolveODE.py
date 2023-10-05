#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 11:31:09 2023

@author: ym
"""

import numpy as np
import torch
#from torch import nn
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import DataCollection as dc
#from mpl_toolkits.mplot3d import Axes3D
import time
import multiprocessing
import os

model_name = 'Van der Pol'

#Specify ROI
low=-3
high=3
ROI = [[low, high], [low, high]]

#Generate sample points within ROI using random samples or uniform grids
M_for_1d = 100
M = M_for_1d**2
    
tic0 = time.time()
#sample = np.random.uniform(low=low, high=high, size=(M, 2))
xx = np.linspace(low, high, M_for_1d)
yy = np.linspace(low, high, M_for_1d)
x_mesh, y_mesh = np.meshgrid(xx, yy)

#Define the eta function
eta = lambda x: 0.2* (x[0]**2 + x[1]**2)
    
#Define the augmented vector field [f, eta] for a 2d system.
def Van_der_Pol(t, var):
    x, y, z = var
    return [-y, x - (1-x**2)*y,  eta([x,y])]

def poly(t, var):
    x, y, z = var
    return [y, -2*x+(x**3)/3-y,  eta([x,y])]

dic = {'Van der Pol': Van_der_Pol, 'poly': poly}


#Define a function to complete the ODE solution with lenth N
def fullarr(arr, N):
    if arr.shape[0]<N:
        d = N - arr.shape[0]
        # Create an array of length d filled with the last element of n_dim_array
        append_array = np.full((d,), arr[-1])
        # Concatenate n_dim_array with append_array to create an m-dimensional row array
        long_arr = np.concatenate((arr, append_array))
        return long_arr
    else:
        return arr

#Definie the trajectory solving and modification function for each initial condition.
def solve_ode(initial_setup, t_span, t_eval, NN, ROI, ODE):
    ode_function = dic[ODE]
    solution = solve_ivp(ode_function, t_span, initial_setup, method='RK45', t_eval=t_eval) #, rtol=1e-6, atol=1e-9)
    data0 = fullarr(solution.y[0], NN)
    data1 = fullarr(solution.y[1], NN)
    data2 = fullarr(solution.y[2], NN)
    data = np.column_stack((data0, data1))
    integral = data2
    index = dc.id_out(data, ROI)
    if index:
        mod_data, intersect = dc.data_mod(data, index, ROI)
        mod_integral_data = dc.integral_mod(integral, mod_data, index, eta, t_span[1])
    else:
        mod_data = data
        mod_integral_data = integral
    return [mod_data[-1], mod_integral_data[-1]]

def ode_data_generator(initial_setups, t_span, t_eval, NN, ROI, model_name):
    print('Start solving ODE')
    tic1 = time.time()
    #results = pool.map(solve_ode, initial_setups)
    results = pool.starmap(solve_ode, [(setup, t_span, t_eval, NN, ROI, model_name) for setup in initial_setups])
    # Close the pool and wait for all processes to complete
    pool.close()
    pool.join()

    #Get the modified trajectory and integral data at the termination time.
    phi = np.stack([results[i][0] for i in range(M)], axis=0)
    I = np.stack([results[i][1] for i in range(M)], axis=0)
    print('ODE solving & modification time = {} sec'.format(time.time()-tic1))
    #Obtain the exp{-integral}
    exp = np.exp(-I)
    
    data = np.column_stack((sample, phi, exp))
    print("Saving data...",flush=True)
    np.save(filename, data)
    total_time = time.time() - tic0
    print(f"Data saved. Total time for data generation: {total_time:.2f} seconds")



if __name__ == "__main__":
    span = 1.5
    t_span = [0, span]
    NN = 1000
    t_eval=np.linspace(0, span, NN)
    
    filename = f'{model_name}_train_data_{M}_samples_ylim_[{low},{high}]_span_{span}.npy'
    sample = np.column_stack((x_mesh.ravel(), y_mesh.ravel()))
    #initial_setups = [[*sample[i], 0] for i in range(M)]
    initial_setups = [[*sample[i], 0] for i in range(M)]

    #Set up parameters for RK45
    
    
    #Use all available CPU cores
    num_processes = multiprocessing.cpu_count()  
    print('cpu count =', num_processes)
    pool = multiprocessing.Pool(processes=num_processes)
    
    ##############################################################################

    if os.path.exists(filename):
        # Path exists, ask the user if they want to overwrite
        overwrite = input("The file " + "'" + filename + "'" + " already exists. Do you want to overwrite it? (yes/no): ")
        if overwrite.lower() == "no":
        # User does not want to overwrite, do nothing
            print("Pass.")
        else:
            ode_data_generator(initial_setups, t_span, t_eval, NN, ROI, model_name)
    else:
        ode_data_generator(initial_setups, t_span, t_eval, NN, ROI, model_name)
        
    ##############################################################################
    
    
    #Plotting for exp
    ##scatters
    data = np.load(filename)
    sample, phi, exp = data[:, 0:2], data[:, 2:4], data[:, -1]
    """
    fig = plt.figure()
    ax = fig.add_subplot(111) #, projection='3d')
    ind = np.where(exp>=0.02)
    ax.scatter(sample[ind, 0], sample[ind, 1],cmap='GnBu',  marker='o', s=1)
    #elev_angle = 10  # Elevation angle (in degrees)
    #azim_angle = 30  # Azimuthal angle (in degrees)
    #ax.view_init(elev=elev_angle, azim=azim_angle)
    plt.show()
    """
    
    ##surface
    print('Plotting for exp(-I)')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    mesh = ax.plot_surface(x_mesh, y_mesh, exp.reshape(x_mesh.shape), cmap='GnBu', rstride=5, cstride=5, linewidth=0.1)
    # Add labels and a color bar
    #ax.set_xlabel('X-axis')
    #ax.set_ylabel('Y-axis')
    #ax.set_zlabel('Z-axis')
    fig.colorbar(mesh, label='$U(\Delta t,x)$', shrink=0.7)
    ax.grid(True, linestyle='--', linewidth=0.2, color='gray')
    plt.show()
    
    ##############################################################################
