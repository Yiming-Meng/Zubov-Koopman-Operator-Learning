#!/bin/bash

# Execute the first Python file
echo "Solving ODE..."
python3 SolveODE.py

# Execute the second Python file
echo "Learing Zubov-Koopman operator..."
python3 ZK_learning.py

# Execute the third Python file
echo "Predicting U..."
python3 U_pred.py

# Script completed
echo "U learning stage ends."
