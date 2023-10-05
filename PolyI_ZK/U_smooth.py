#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 21:23:20 2023

@author: ym
"""


import numpy as np
import torch
from torch import nn
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import DataCollection as dc
import SolveODE as ss
#from mpl_toolkits.mplot3d import Axes3D
import time
import multiprocessing
import torch.multiprocessing as mp
import os
import random
from torch.utils.data import DataLoader, TensorDataset
from matplotlib.lines import Line2D

model_name = ss.model_name
N = 50
basis_num = (2*N-1)**2
#Specify ROI
low = ss.low
high = ss.high
ROI = ss.ROI

def quad_func(x1, x2):
    return 1.75*x1**2 + 0.5*x1*x2 + 0.75*x2**2

def reverse_poly(t, z):
    x1, x2 = z
    dx1_dt = - x2
    dx2_dt = 2.0*x1 - 1.0/3.0*x1**3 + x2
    return [dx1_dt, dx2_dt]

# Define paramters for data generation, network model, and optimization
#n_samples = 300**2  # number of samples
ylim = 3.5          # |x_2|<=ylim
lr = 0.001          # learning rate
n_epochs = 300      # number of epochs
layer = 2          # number of hiddent layers
width = 30          # number of neurons in each hidden layer
err = 1e-10          # terminating condition for loss
bach_size = 32

#Generate sample points within ROI using random samples or uniform grids
M_for_1d = 300
M = M_for_1d**2

M_test_for_1d = 500
M_test = M_test_for_1d**2

U_filename = f'{model_name}_predicted_U_at_{M_test}_samples_ylim_[{low},{high}]_{M}.npy'
print("Loading data...",flush=True)
data = np.load(U_filename)
x_train, y_train = data[:, 0:2], data[:, -1]

ind_0 = np.where(y_train<0)
y_train[ind_0] = 0
print("Data Loaded.",flush=True)

mesh_size = len(x_train)
side_length = int(np.sqrt(mesh_size))
x_mesh = x_train[:, 0].reshape(side_length, side_length)
y_mesh = x_train[:, 1].reshape(side_length, side_length)


x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1)
print(x_train)


train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=bach_size, shuffle=True)


# Solve the van der Pol system to get the limit cycle as ROA boundary
x_init1 = [np.sqrt(6), 1e-9]
x_init2 = [np.sqrt(6), -1e-9]
x_init3 = [-np.sqrt(6), 1e-9]
x_init4 = [-np.sqrt(6), -1e-9]
T_max = 10.0
t_eval = np.linspace(0, T_max, 1000)  #
vsol1 = solve_ivp(reverse_poly, [0, T_max], x_init1, rtol=1e-6, atol=1e-9, t_eval=t_eval)
vsol2 = solve_ivp(reverse_poly, [0, T_max], x_init2, rtol=1e-6, atol=1e-9, t_eval=t_eval)
vsol3 = solve_ivp(reverse_poly, [0, T_max], x_init3, rtol=1e-6, atol=1e-9, t_eval=t_eval)
vsol4 = solve_ivp(reverse_poly, [0, T_max], x_init4, rtol=1e-6, atol=1e-9, t_eval=t_eval)

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self, num_layers, width):
        super(Net, self).__init__()
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(2, width))
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(width, width))
        
        self.fc = nn.Linear(width, 1)
        self.activation = nn.Tanh()
        
    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.fc(x)
        return x
    
    
    
# Check if a model with same hyperparameters exists
model_file = f"{model_name}_NN_Lyap_layer_{layer}_width_{width}_samples_{M_test}_lr_{lr}_epoch_{n_epochs}_{M}.pt"


# Check if model file exists
if os.path.isfile(model_file):
    # Load the model from the file
    print("Loading model...",flush=True)
    net = torch.load(model_file)
else:
    # Create an instance of the neural network
    net = Net(num_layers=layer, width=width)
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in the model: {num_params}")

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #optimizer = pbSGD(net.parameters(), lr=lr, gamma=0.7)
    # Train the network
    losses = []
    start_time = time.time()
    print("Training network...",flush=True)
    for epoch in range(n_epochs):
        epoch_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = net(x_batch)
            loss = criterion(y_pred, y_batch) 
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        losses.append(epoch_loss / len(train_loader))
        if epoch % 1 == 0:
            print(f"Epoch {epoch}: Loss = {epoch_loss / len(train_loader):.5f}", flush=True)
        if epoch_loss / len(train_loader) < err:
            print(f"Stopping training after epoch {epoch} because loss is less than {err}", flush=True)
            break


    torch.save(net, model_file)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Model saved. Total training time: {total_time:.2f} seconds")
    print('Final loss:', losses[-1])
    

    
# Create a grid of points for testing the network
x1_vals, x2_vals = np.meshgrid(np.linspace(-5.5, 5.5, 500), np.linspace(-6, 6, 500))
x_test = np.vstack((x1_vals.reshape(-1), x2_vals.reshape(-1))).T
x_test = torch.from_numpy(x_test).float()

#net_dd = torch.load('Van_der_Pol_dd_Lyap_layer_2_width_15_samples_40000_lr_0.001_epoch_500.pt')
#learned_dd = torch.load('Van_der_Pol_dd_Lyap_layer_2_width_15_samples_62500_lr_0.001_epoch_500.pt')

# Evaluate the network on the test points
with torch.no_grad():
    y_test = net(x_test)
    #y_test_dd = net_dd(x_test)
    zero = torch.FloatTensor(np.array([[0., 0.]]))
    y_test = y_test/net(zero)
    #y_dd_veri = learned_dd(x_test)

y_test = y_test.numpy()
y_vals = y_test.reshape(x1_vals.shape)
"""
# Reshape the output to match the input grid

y_test_dd = y_test_dd.numpy()
y_vals_dd = y_test_dd.reshape(x1_vals.shape)
#y_test_dd_veri = y_dd_veri.numpy()
#y_vals_dd_veri = y_test_dd_veri.reshape(x1_vals.shape)
"""

"""
# Generate plots
# Plot the training data
fig = plt.figure(figsize=(6,5))

ax = fig.add_subplot(1, 3, 1)
ax.scatter(x_test[:,0], x_test[:,1], c=1-y_test, cmap='GnBu', s=2)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('Training data')

# Plot the predicted function
ax = fig.add_subplot(1, 3, 2,  projection='3d')
ax.plot_surface(x1_vals, x2_vals, 1-y_vals)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('W')
ax.set_title('Learned Lyapunov function')
"""

"""
plt.figure(2, figsize=(6,5))
indices = np.where(0.003<= y_test)
plt.xlim(-2.5, 2.5)  # Set the x-axis range
plt.ylim(-3, 3)
plt.scatter(x_test[indices, 0], x_test[indices, 1], color = (0.7, 0., 0.), marker='o',  s=1)


indices_new = np.where(0.00004<= y_train)
plt.xlim(-2.5, 2.5)  # Set the x-axis range
plt.ylim(-3, 3)
plt.scatter(x_train[indices_new, 0], x_train[indices_new, 1], color=(0.7, 0.9, 0.7), marker='o',   s=1) #
"""

fig2 = plt.figure(figsize=(6,5)) #figsize=(6,5))
ax = fig2.add_subplot()
levels = [0.29]
#ax.contour(x1_vals, x2_vals, quad_func(x1_vals, x2_vals), levels=levels, colors='r', linewidths=2, linestyles='--')
levels2 = [0.022]
ax.contour(x_mesh, y_mesh, y_train.reshape(x_mesh.shape), colors='#00008B', levels=[0.0210], linewidths=1.5, linestyles='-.', label='Predicted ROA boundary')
#ax.contour(x_mesh, y_mesh, y_train.reshape(x_mesh.shape), colors='#00008B', levels=[0.0260], linewidths=1.5, label='Verified ROA boundary')
ax.contour(x1_vals, x2_vals, y_vals, levels=levels2, colors='#00008B',linewidths=2, label='Verified boundary')
#ax.contour(x1_vals, x2_vals, y_vals_dd, colors='green', levels=[0.00275], linewidths=1.5, linestyles='-.', label='DD-Predicted ROA boundary')
#ax.contour(x1_vals, x2_vals, y_vals_dd, colors='green', levels=[0.004], linewidths=1.5, label='DD-Predicted ROA boundary')
ax.plot(vsol1.y[0], vsol1.y[1], color='#AA0000', linewidth=2, label='Approximate ROA boundary')
ax.plot(vsol2.y[0], vsol2.y[1], color='#AA0000', linewidth=2)
ax.plot(vsol3.y[0], vsol3.y[1], color='#AA0000', linewidth=2)
ax.plot(vsol4.y[0], vsol4.y[1], color='#AA0000', linewidth=2)
ax.set_xlim(-6,6)
ax.set_ylim(-6,6)
# Plot the level sets
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
#ax.clabel(cs, inline=1, fontsize=10)#  fmt=manual_labels)
#ax.plot(vsol.y[0], vsol.y[1], color='red', linewidth=2, label='Limit cycle (ROA boundary)')


legend_handles = [
    #Line2D([0], [0], color='#AA0000', linewidth=1, linestyle='--', label='Local'),
    Line2D([0], [0], color='#00008B', linewidth=1.5, linestyle='-.', label='ZK-Predicted boundary'),
    Line2D([0], [0], color='#00008B', linewidth=1.5, label='ZK-verified'),    
    #Line2D([0], [0], color='green', linewidth=1.5, linestyle='-.', label='Data-Driven Predicted boundary'),
    #Line2D([0], [0], color='green', linewidth=1.5, label='Data-Driven verified'),
    Line2D([0], [0], color='#AA0000', linewidth=1.5, label='ROA boundary'),
]

ax.legend(handles=legend_handles, loc='lower right')
ax.set_title('Prediction and Verification of the Region of Attraction')
plt.text(2.8, 5.25, '$M = 500^2$', fontsize=12, color='black', va='top', ha='left')
plt.text(2.8, 4.25, '$Width = 30$', fontsize=12, color='black', va='top', ha='left')
plt.show()
plt.tight_layout()
fig2.savefig('poly3.png', dpi=300) 

    
    