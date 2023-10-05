#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 19:46:08 2023

@author: ym
"""

"""
Description: use dReal to verify a level set of a neural Lyapunov function 
as a region of attraction. 
"""

import numpy as np
import torch
import torch.nn as nn


from dreal import *
import timeit 

import sys

if len(sys.argv) < 4:
    print("Please provide a filename as a command line argument")
    sys.exit()

model_file = sys.argv[1]
c2 = float(sys.argv[2])
c1 = float(sys.argv[3])

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

model = torch.load(model_file)
layers = len(model.layers) 






x1 = Variable("x1")
x2 = Variable("x2")
dreal_vars = [x1,x2]
config = Config()
config.use_polytope_in_forall = True
config.use_local_optimization = True
config.precision = 1e-2
epsilon = 1e-04
xlim = [4.5, 5.9]

f_ex = [x2, -2.0*x1 + 1.0/3.0*x1**3 - x2]

# Extract V from neural network model
layers = len(model.layers) 
# Extract weights for each layer
weights = [layer.weight.data.cpu().numpy() for layer in model.layers]
biases = [layer.bias.data.cpu().numpy() for layer in model.layers]

# Extract weights and biases for the final layer separately
final_layer_weight = model.fc.weight.data.cpu().numpy()
final_layer_bias = model.fc.bias.data.cpu().numpy()

# Calculate h for each layer
h = dreal_vars
for i in range(layers):
    z = np.dot(h, weights[i].T) + biases[i]
    h = [tanh(z[j]) for j in range(len(weights[i]))]

V_learn = (np.dot(h, final_layer_weight.T) + final_layer_bias)[0]

print(V_learn)

def CheckLyapunov_c(x, f, V, c1, c2, config, epsilon):    
    x_bound = Expression(0)
    lie_derivative_of_V = Expression(0)
    
    for i in range(len(x)):
        lie_derivative_of_V += f[i]*V.Differentiate(x[i])  
    vars_in_bound = logical_and(c1 <= V, V <= c2)
    
    x1_bound = logical_and(x[0]<=xlim[0], x[0]>=-xlim[0])
    x2_bound = logical_and(x[1]<=xlim[1], x[1]>=-xlim[1])
    x12_bound = logical_and(x1_bound, x2_bound)
    x_bound = logical_and(x12_bound, vars_in_bound)

    x_boundary = logical_or(x[0] == xlim[0], x[0] == -xlim[0])
    x_boundary = logical_or(x[1] == xlim[1], x_boundary)
    x_boundary = logical_or(x[1] == -xlim[1], x_boundary)

    set_inclusion = logical_imply(x_bound, logical_not(x_boundary))
    stability = logical_imply(logical_and(x12_bound, V >= c2), 
    								1.75*x[0]**2 + x[0]*x[1] + 0.7 * x[1]**2 <= 1.05)
    reach = logical_imply(x_bound, lie_derivative_of_V >= epsilon) 

    condition = logical_and(logical_and(reach, stability),set_inclusion)

    return CheckSatisfiability(logical_not(condition),config)

start_ = timeit.default_timer() 

result= CheckLyapunov_c(dreal_vars, f_ex, V_learn, c1, c2, config, epsilon) 

stop_ = timeit.default_timer() 
t = stop_ - start_

print(f"Verification results for {model_file}.")

if (result): 
  print(f"Not a Lyapunov function on U_NN >= {c1}. Found counterexample: ")
  print(result)
else:
  print("Satisfy conditions with beta = ", epsilon)
  print(V_learn, f" is a Lyapunov function on U_NN >= {c1}.")

print("Verification time =", t)