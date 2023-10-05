#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 19:32:57 2023

@author: ym
"""

import numpy as np
import torch
#from torch import nn
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import DataCollection as dc
import SolveODE as ss
import ZK_Learning as zk
#from mpl_toolkits.mplot3d import Axes3D
import time
import multiprocessing
import torch.multiprocessing as mp
import matplotlib.ticker as ticker
import os

model_name = ss.model_name
N = 50
basis_num = (2*N-1)**2
#Specify ROI
low = ss.low
high = ss.high
ROI = ss.ROI


M_for_1d = ss.M_for_1d
M = ss.M

#Generate test points within ROI using random samples or uniform grids
M_test_for_1d = 500
M_test = M_test_for_1d**2

tolerance = 1e-2
max_iterations = 5

def exp_g(n, m, x):
    return np.cos(np.pi/high*(n*x[:, 0])+np.pi/high*(m*x[:, 1])) * np.exp(-x[:, 0]**2/40-x[:, 1]**2/40)

def U_predict(vec, x):
    basis = np.stack([exp_g(i, j, x) for i in range(-(N-1), N) for j in range(-(N-1), N)]).T
    return basis @ vec

# Define the quadratic Lyapunov function
def quad_func(x1, x2):
    return 1.5*x1**2 - x1*x2 + x2**2

#Define the eta function
eta = ss.eta

Van_der_Pol = ss.Van_der_Pol
poly = ss.poly

def reverse(t, xy, mu=6.0):
    x, y = xy[0], xy[1]
    return [y, mu * (1.0 - x ** 2) * y - x]


# Solve the van der Pol system to get the limit cycle as ROA boundary
x_init = [0.1, 0.1]
T_max = 100.0
t_eval = np.linspace(T_max-14.14, T_max, 1000)  # Evaluate only over the last ~14.14 second (one period)
vsol = solve_ivp(reverse, [0, T_max], x_init, rtol=1e-6, atol=1e-9, t_eval=t_eval)

# Function to perform an iterative update on the tensor
def iterative_update(tensor, tolerance, max_iterations):
    norm = torch.norm(tensor, 'fro')
    iteration = 0

    while norm > tolerance and iteration < max_iterations:
        
        # Perform your update step here, e.g., matrix multiplication
        # Replace this with your specific update logic
        tensor_new = update_function(tensor)
        norm = torch.norm(tensor_new-tensor, 'fro')
        tensor = tensor_new
        
        print('Iteration', iteration+1)
        print('Norm difference=', norm)
        iteration += 1
    return tensor

# Function to update the tensor (replace with your specific update logic)
def update_function(tensor):
    # Replace this with your update logic, e.g., matrix multiplication
    return tensor @ tensor


def plots(tensor, x_mesh, y_mesh, mesh_stack_test):
    
    fig = plt.figure(figsize=(20,5))

    ax = fig.add_subplot(1, 3, 1)
    ax.scatter(mesh_stack_test[:,0], mesh_stack_test[:,1], c=1.0-tensor, cmap='coolwarm', s=2)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Training data')
    ax.set_xlim(-8.5, 8.5) 
    ax.set_ylim(-9, 9) 
    # Plot the predicted function
    
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.plot_surface(x_mesh, y_mesh, tensor.reshape(x_mesh.shape), cmap='GnBu')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('U')
    ax.set_title('Learned U')
    
    fig2 = plt.figure(figsize=(20,5))
    ax = fig2.add_subplot(1, 3, 1)
    levels = [0.29]
    ax.contour(x_mesh, y_mesh, quad_func(x_mesh, y_mesh), levels=levels, colors='r', linewidths=2, linestyles='--')

    # Plot the target set described by the quadratic function
    #levels = [4*1e-1**(i+1) for i in range(5)][::-1]
    levels = [0.2]
    #manual_labels = {4e-5: '0.00004', 0.0004: '0.0004', 0.004: '0.004', 0.04: '0.04', 0.4: '0.4'}
    #formatter = ticker.ScalarFormatter(useMathText=True, useOffset=False)
    cs = ax.contour(x_mesh, y_mesh, tensor.reshape(x_mesh.shape), levels=levels, linewidths=2)
    # Plot the level sets
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.clabel(cs, inline=1, fontsize=10)#  fmt=manual_labels)
    #ax.plot(vsol.y[0], vsol.y[1], color='red', linewidth=2, label='Limit cycle (ROA boundary)')
    ax.legend()
    ax.set_title('Level sets')
    
    


def U_pred_generator(x_mesh, y_mesh, mesh_stack_test):
    torch_filename = f'{model_name}_learned_ZK_{M}_samples_{basis_num}_basis_ylim_[{low},{high}].pt'
    T = torch.load(torch_filename)

    tic = time.time()
    T_iter = iterative_update(T, tolerance, max_iterations)
    print(time.time()-tic)

    T_iter = T_iter.numpy()
    E = np.eye(basis_num) 
    a = E[0]
    vec = T_iter @ a
    
    
    print('Evaluate U_pred at test points')
    tic0 = time.time()
    U_pred = U_predict(vec, mesh_stack_test)
    #U_pred = X_mesh @ vec
    #X_mesh = np.stack([exp_g(i, j, mesh_stack_test)  for i in range(-(N-1), N) for j in range(-(N-1), N)]).T
    
    print('Calculation time: {} sec'.format(time.time()-tic0))
    
    
    largest_value = np.max(np.abs(U_pred))
    ind_l = np.argmax(U_pred)
    U_pred = U_pred/U_pred[ind_l]
    #U_reshape = U_pred.reshape(x_mesh.shape)
    
    plots(U_pred, x_mesh, y_mesh, mesh_stack_test)
    data_for_smoothing = np.column_stack((mesh_stack_test, U_pred))
    print(data_for_smoothing.shape)

    np.save(U_filename, data_for_smoothing)
    np.save(weight_filename, vec)
    print('ZK operator Data saved.')

if __name__ == "__main__":
    U_filename = f'{model_name}_predicted_U_at_{M_test}_samples_ylim_[{low},{high}]_{M}.npy'
    weight_filename = f'{model_name}_U_weights_{M}_samples_{basis_num}_basis_ylim_[{low},{high}].npy'
    xx = np.linspace(-4, 4, M_test_for_1d)
    yy = np.linspace(-8.5, 8.5, M_test_for_1d)
    x_mesh, y_mesh = np.meshgrid(xx, yy)
    mesh_stack_test = np.column_stack((x_mesh.ravel(), y_mesh.ravel()))

    if os.path.exists(U_filename):
        overwrite = input("The file " + "'" + U_filename + "'" + " already exists. Do you want to overwrite it? (yes/no): ")
        #plot = input("Do you want to plot U_pred? (yes/no): ")
        if overwrite.lower() == "no":
            print('Plotting data')
            data = np.load(U_filename)
            x_train, U_pred = data[:, 0:2], data[:, -1]
            mesh_size = len(x_train)
            side_length = int(np.sqrt(mesh_size))
            x_mesh = x_train[:, 0].reshape(side_length, side_length)
            y_mesh = x_train[:, 1].reshape(side_length, side_length)
            plots(U_pred, x_mesh, y_mesh, mesh_stack_test)
            plt.plot(vsol.y[0], vsol.y[1], color='red', linewidth=2, label='Limit cycle (ROA boundary)')
            

        else:
            U_pred_generator(x_mesh, y_mesh, mesh_stack_test)
    else:
        U_pred_generator(x_mesh, y_mesh, mesh_stack_test)
        




