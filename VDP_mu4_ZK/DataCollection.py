#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 11:25:41 2023

@author: ym
"""

import numpy as np
import matplotlib.pyplot as plt




def id_out(data, ROI):
    # Create a boolean mask of out-of-domain elements
    if len(ROI) == 1:
        domain = ROI[0]
        out_of_domain_index = np.argmax((data < domain[0]) | (data > domain[1]))
        return out_of_domain_index 
    
    else: 
        out_of_domain_mask = np.zeros(data.shape[0], dtype=bool)
        for i, (min_val, max_val) in enumerate(ROI):
            out_of_domain_mask |= (data[:, i] < min_val) | (data[:, i] > max_val)
    
        # Find the indices of the first out-of-domain element
        out_of_domain_indices = np.argwhere(out_of_domain_mask)
        return out_of_domain_indices[0][0] if out_of_domain_indices.size > 0 else None


def where_intersect(points, ROI):
    boundaries = {
    "left": (ROI[0][0], None),
    "right": (ROI[0][1], None),
    "top": (None, ROI[1][1]),
    "bottom": (None, ROI[1][0])}
    
    x1, y1 = points[0]
    x2, y2 = points[1]
    
    x_min = min(x1, x2)
    y_min = min(y1, y2)
    x_max = max(x1, x2)
    y_max = max(y1, y2)
    
    line_func = lambda x: (y2-y1)/(x2-x1) * x + (y1 - (y2-y1)/(x2-x1)*x1) if x1 != x2 else y1
    
    intersections = []
    for boundary, (x_val, y_val) in boundaries.items():
        if boundary in ["left", "right"]:
            y = line_func(x_val)
            if y_min <= y <= y_max and x_min <= x_val <= x_max:
                intersections.append((x_val, y))
        else:
            if x1 != x2:
                x = (y_val - (y1 - (y2-y1)/(x2-x1)*x1)) / ((y2-y1)/(x2-x1))
                if y_min <= y_val <= y_max and x_min <= x <= x_max:
                    intersections.append((x, y_val))
            else:
                if y_min <= y_val <= y_max: 
                    intersections.append((x1, y_val)) 
    if intersections == []:
        return [(x2, y2)]
    else:
        return intersections


def data_mod(data, out_of_domain_index, ROI):
    n = len(ROI)
    index = int(out_of_domain_index)
    pre = int(index-1)
    points = np.array([data[pre], data[index]])
    #print(points)
    if n ==1:
        intersect = [ROI[0][1] if points[0] <= ROI[0][1] <= points[1] else  ROI[0][0]]
    else:
        intersect = where_intersect(points, ROI)
    
    data[index:] = intersect[0] 
    
    return data, intersect


def integral_mod(integral_data, mod_data, out_of_domain_index, eta, span):
    index = int(out_of_domain_index)
    pre = int(index - 1)
    size = integral_data.shape[0] 
    value_pre = integral_data[pre]
    new_integrand = eta(mod_data[index])
    new_size = int(size - index)
    discrete_interval = span / size
    rear_integral = np.array([value_pre + new_integrand*discrete_interval*(i+1) for i in range(0, new_size)])
    integral_data[index:] = rear_integral
    return integral_data

def plot_2d(ROI, intersect, data, points=None):

    plt.plot([ROI[0][0], ROI[0][0]], [ROI[1][0], ROI[1][1]], color='k', linestyle='--')  # Left boundary
    plt.plot([ROI[0][1], ROI[0][1]], [ROI[1][0], ROI[1][1]], color='k', linestyle='--')  # Right boundary
    plt.plot([ROI[0][0], ROI[0][1]], [ROI[1][0], ROI[1][0]], color='k', linestyle='--')  # Bottom boundary
    plt.plot([ROI[0][0], ROI[0][1]], [ROI[1][1], ROI[1][1]], color='k', linestyle='--')  # Top boundary

    plt.xlim(1.25* ROI[0][0], 1.25* ROI[0][1])
    plt.ylim(1.25* ROI[1][0], 1.25* ROI[1][1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Random Line and Boundary Intersections')
    plt.grid(True)
    plt.legend()
    
    if points:
        x1, y1 = points[0]
        x2, y2 = points[1]
        line_func = lambda x: (y2-y1)/(x2-x1) * x + (y1 - (y2-y1)/(x2-x1)*x1) if x1 != x2 else y1
        x_span = np.linspace(x1, x2, 100)
        y_span = line_func(x_span) if x1!=x2 else np.linspace(y1, y2, 100)
        plt.plot(x_span, y_span, label='Line connecting random points', color='blue')
        plt.scatter(points[:, 0], points[:, 1], color='red')
        
    if intersect:
        plt.scatter(*zip(*intersect), color='green')


    array1 = data[:, 0]  # Get all rows of the first column
    array2 = data[:, 1] 
    plt.plot(array1, array2)
    
    plt.show()

    

    
    

if __name__ == '__main__':
    ROI=[[-1, 1], [-1, 1]]
    data = np.array([[0.7,0.7], [0.8, 0.8], [0.9, 0.9], [1.0, 1.0], [1.3, 1.3]])

    ROI2= [[-1, 1]]
    data2 = np.array([0.5, -1.2, 0.9, 1.1, -0.8, 0.7])
    
    index = id_out(data2, ROI2)
    if index:
        print('index=', index)
    else:
        print("No data is out of domain")
        
    mod_data, intersect = data_mod(data2, index, ROI2)
       
    #plot_2d(ROI2, intersect, data_mod)
    print(mod_data)
    
    integral = np.array([i/2 for i in range(0, 10)])
    print(integral)
    span = 10
    num = 11
    eta = lambda x: x*x
    integral_data = integral_mod(integral, mod_data, index, eta, span)
    
    print(integral_data)
    
    
    
    


