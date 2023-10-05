#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 16:39:55 2023

@author: ym
"""


import numpy as np
import torch
#from torch import nn
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import DataCollection as dc
import SolveODE as ss
#from mpl_toolkits.mplot3d import Axes3D
import time
import multiprocessing
import torch.multiprocessing as mp
import os

model_name = ss.model_name
N = 50
basis_num = (2*N-1)**2
#Specify ROI
low = ss.low
high = ss.high
ROI = ss.ROI

#Generate sample points within ROI using random samples or uniform grids
M_for_1d = ss.M_for_1d
M = ss.M



#Prepare dictionary functions (observables)


def exp_gt(n, m, x):
    return torch.cos(torch.pi/high *n*x[:, 0] + torch.pi/high * m*x[:, 1]) * torch.exp(-x[:, 0]**2/25 - x[:, 1]**2/25)
    #return torch.cos(torch.pi/3 * (n*x[:, 0] + m*x[:, 1])) * torch.exp(-(x[:, 0]**2 + x[:, 1]**2)/4)


def compute_pseudo_inverse(tensor):
    pseudo_inverse = torch.linalg.pinv(tensor)
    return pseudo_inverse

def compute_eigen(tensor, result_queue):
    # Move the tensor to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = tensor.to(device)
    eigenvalues, eigenvectors = torch.linalg.eig(tensor)
    result_queue.put((eigenvalues, eigenvectors))

def learned_op_generator(sam_torch, phi_torch, exp_torch):
    print("Converting ODE data to EDMD trainging stacks...",flush=True)
    tic0 = time.time()
    X_EDMD = torch.stack([exp_gt(i, j, sam_torch) for i in range(-(N-1), N) for j in range(-(N-1), N)]).T
    Y_EDMD = torch.stack([exp_gt(i,j, phi_torch)*exp_torch for i in range(-(N-1), N) for j in range(-(N-1), N)]).T
    print('Converting time = {} sec'.format(time.time() - tic0))
    
    print("Learing ZK operator",flush=True)
    tic1 = time.time()
    X_TX = (X_EDMD.T)@ X_EDMD
    X_TY = (X_EDMD.T) @ Y_EDMD
    with mp.Pool(processes=num_processes) as pool:
        pseudo_inverse = pool.apply(compute_pseudo_inverse, (X_TX,))
    T = pseudo_inverse @ X_TY
    print('ZK operator learning time = {} sec'.format(time.time() - tic1))
    
    torch.save(T, torch_filename)
    print('ZK operator Data saved.')


def eig_op_generator(tensor):
    print("Calculating ZK eigens",flush=True)
    tic2 = time.time()
    result_queue = mp.Queue()
    # Initialize a process to compute the eigenvalues and eigenvectors
    process = mp.Process(target=compute_eigen, args=(tensor, result_queue))
    # Start the process
    process.start()
    # Wait for the process to finish
    process.join()
    # Retrieve the results from the queue
    e, v = result_queue.get()
    print('ZK eigens calculation time = {} sec'.format(time.time() - tic2))
    print('Eigenvalues of the learned ZK opeartor = ', e)
    
    torch.save(e, eig_filename)
    print('Eigenvalues of ZK operator saved.')
    

if __name__ == '__main__':
    #Set path for the learning data
    span = 2
    torch_filename = f'{model_name}_learned_ZK_{M}_samples_{basis_num}_basis_ylim_[{low},{high}].pt'
    #Set path for the eigenvalues
    eig_filename = f'{model_name}_eigenvalues_ZK_{M}_samples_{basis_num}_basis_ylim_[{low},{high}].pt'
    #Load ODE data
    filename = f'{model_name}_train_data_{M}_samples_ylim_[{low},{high}]_span_{span}.npy'
    print("Loading data...",flush=True)
    data = np.load(filename)
    sample, phi, exp = data[:, 0:2], data[:, 2:4], data[:, -1]
    print("Data Loaded.",flush=True)

    sam_torch = torch.FloatTensor(sample)
    phi_torch = torch.FloatTensor(phi)
    exp_torch = torch.FloatTensor(exp)
    #n_pairs = [(m, n) for m in range(-(N-1), N) for n in range(-(N-1), N)]
    
    
    # Initialize multiprocessing pool
    num_processes = mp.cpu_count()  # Number of CPU cores

    if os.path.exists(torch_filename):
        # Path exists, ask the user if they want to overwrite
        overwrite = input("The file " + "'" + torch_filename + "'" + " already exists. Do you want to overwrite it? (yes/no): ")
        if overwrite.lower() == "no":
        # User does not want to overwrite, do nothing
            print("Pass.")
        else:
            learned_op_generator(sam_torch, phi_torch, exp_torch)
    else:
        learned_op_generator(sam_torch, phi_torch, exp_torch)
        
        
    if os.path.exists(eig_filename):
        # Path exists, ask the user if they want to overwrite
        overwrite = input("The file " + "'" + eig_filename + "'" + " already exists. Do you want to overwrite it? (yes/no): ")
        check_eig = input("Do you want to check the eigenvalues of the learned operator? (yes/no): ")
        if overwrite.lower() == "no":
            if check_eig.lower() == "no":
                print("Pass.")
            else:
                eig = torch.load(eig_filename)
                print("Eigenvalues of the learned ZK opeartor = ", eig)
        else:
            T = torch.load(torch_filename)
            eig_op_generator(T)
    else:
        T = torch.load(torch_filename)
        eig_op_generator(T)

    


