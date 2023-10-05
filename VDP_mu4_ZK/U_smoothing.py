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
import ZK_Learning as zk
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

# Define the quadratic Lyapunov function
def quad_func(x1, x2):
    return 2.25*x1**2 - x1*x2 + 0.25*x2**2

# Define paramters for data generation, network model, and optimization
#n_samples = 300**2  # number of samples
ylim = 6.5          # |x_2|<=ylim
lr = 0.001          # learning rate
n_epochs = 500      # number of epochs
layer = 2          # number of hiddent layers
width = 15          # number of neurons in each hidden layer
err = 1e-8          # terminating condition for loss
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



train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=bach_size, shuffle=True)

# Define the van der Pol system
def van_der_pol(t, xy, mu=4.0):
    x, y = xy[0], xy[1]
    return [y, mu * (1.0 - x ** 2) * y - x]

# Solve the van der Pol system to get the limit cycle as ROA boundary
x_init = [0.1, 0.1]
T_max = 100.0
t_eval = np.linspace(T_max-14.14, T_max, 1000)  # Evaluate only over the last ~14.14 second (one period)
vsol = solve_ivp(van_der_pol, [0, T_max], x_init, rtol=1e-6, atol=1e-9, t_eval=t_eval)

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
x1_vals, x2_vals = np.meshgrid(np.linspace(-4, 4, 500), np.linspace(-ylim, ylim, 500))
x_test = np.vstack((x1_vals.reshape(-1), x2_vals.reshape(-1))).T
x_test = torch.from_numpy(x_test).float()

#net_dd = torch.load('Van_der_Pol_dd_Lyap_layer_2_width_15_samples_10000_lr_0.001_epoch_300.pt')
#learned_dd = torch.load('Van_der_Pol_dd_Lyap_layer_2_width_15_samples_62500_lr_0.001_epoch_500.pt')

# Evaluate the network on the test points
with torch.no_grad():
    y_test = net(x_test)
    #y_test_dd = net_dd(x_test)
    zero = torch.FloatTensor(np.array([[0., 0.]]))
    y_test = y_test/net(zero)
    #y_dd_veri = learned_dd(x_test)


# Reshape the output to match the input grid
y_test = y_test.numpy()
y_vals = y_test.reshape(x1_vals.shape)
#y_test_dd = y_test_dd.numpy()
#y_vals_dd = y_test_dd.reshape(x1_vals.shape)
#y_test_dd_veri = y_dd_veri.numpy()
#y_vals_dd_veri = y_test_dd_veri.reshape(x1_vals.shape)


# Generate plots
# Plot the training data
fig = plt.figure(figsize=(10,5))

ax = fig.add_subplot(1, 2, 1)
ax.scatter(x_test[:,0], x_test[:,1], c=1-y_test, cmap='GnBu', s=2)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('Training data')

# Plot the predicted function
ax = fig.add_subplot(1, 2, 2,  projection='3d')
ax.plot_surface(x1_vals, x2_vals, 1-y_vals)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('U')
ax.set_title('Learned Lyapunov function')



indices = np.where(0.05<= y_test)

fig2 = plt.figure(figsize=(6,5))
ax = fig2.add_subplot()
levels = [1.16]
#ax.contour(x1_vals, x2_vals, quad_func(x1_vals, x2_vals), levels=levels, colors='r', linewidths=2, linestyles='--')
levels2 = [ 0.3, 0.5, 0.7, 0.9]
ax.contour(x_mesh, y_mesh, y_train.reshape(x_mesh.shape), colors='#00008B', levels=[0.05], linewidths=1.5, linestyles='-.', label='Predicted ROA boundary')
#cs = ax.contour(x1_vals, x2_vals, y_vals, levels=levels2, linewidths=1.5, label='Level sets')
ax.plot(vsol.y[0], vsol.y[1], color='#AA0000', linewidth=1.5, label='Limit cycle (ROA boundary)')
filtered_x_test = x_test[indices[0]]
filtered_y_test = y_test[indices[0]]
ax.scatter(filtered_x_test[:,0], filtered_x_test[:,1], c=1-filtered_y_test, cmap='tab20c', s=2)

ax.set_xlim(-3, 3)
ax.set_ylim(-6.5, 6.5)
# Plot the level sets
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
#ax.clabel(cs, inline=1, fontsize=10)#  fmt=manual_labels)
#ax.plot(vsol.y[0], vsol.y[1], color='red', linewidth=2, label='Limit cycle (ROA boundary)')


legend_handles = [
    #Line2D([0], [0], color='#AA0000', linewidth=1, linestyle='--', label='Local'),
    Line2D([0], [0], color='#00008B', linewidth=1.5, linestyle='-.', label='ZK-Predicted boundary'),
    Line2D([0], [0], color='#AA0000', linewidth=1.5, label='ROA boundary'),
]

ax.legend(handles=legend_handles, loc='lower right')
ax.set_title('Prediction of the Region of Attraction')
plt.text(2.2, 3.25, '$\mu = 4$', fontsize=12, color='black', va='top', ha='left')

plt.show()
fig2.savefig('mu4.png', dpi=300) 

    
    