#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 22:07:49 2023

@author: ym
"""

import numpy as np
import torch
import torch.nn as nn

import sys



model_file = 'poly_NN_Lyap_layer_2_width_30_samples_250000_lr_0.001_epoch_300_90000.pt'
c1 = 0.022

# Define the neural network architecture and load the model
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

net = torch.load(model_file)

## Load data file augment the PINNS training
filename = "poly_predicted_U_at_250000_samples_ylim_[-6,6]_90000.npy"
data = np.load(filename)
x_data, y_data = data[:, :-1], data[:, -1]

outputs = net(torch.Tensor(x_data)).squeeze().detach().numpy()
print(f"The size of 'outputs' is {len(outputs)}")

num_cases = np.count_nonzero(y_data > 0.021)
print(f"The size of 'winning set' is {num_cases}")
count = 0

for i in range(len(outputs)):
    if outputs[i] >= c1:
        count += 1

ratio = count / num_cases * 100

print(f"The ratio of cases where the model output is less than or equal to {c1} and y_data > 0 is {ratio:.2f}%")
