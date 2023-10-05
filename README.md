# Zubov-Koopman-Operator-Learning
This programming aims to learn the solutions to Zubov's equations for unknown dynamical systems using Zubov-Koopman approach.

# Brief Introduction
There has been a recent surge of interest in estimating the solution to the Zubov’s Equation, whose non-trivial sub-level sets 
form the exact ROA. We propose a lifting approach to map observable data into an infinite-dimensional function space, which 
generates a flow governed by our proposed operators named as the ‘Zubov-Koopman’ operators. By learning a Zubov-Koopman operator 
over a fixed time interval, we can indirectly approximate the solution to Zubov’s Equation through iterative application of the 
learned operator on the identity function. We also demonstrate that a transformation of such an approximator can be readily utilized 
as a near-maximal Lyapunov function. 

We provide examples for a 1-dim dynamical system, Van der Pol oscillator, a polynomial system, and stiff Van der Pol oscillators with
mu = 4 and 6. Other examples can be easily adjusted based on the settings from the provided examples.

# Sourse files
For the 1-dim system, we use a jupyter note to explain the procedure and phenomena.
For the others, excecute 'exe.sh' to obtain the prediction $U_{ZK}$ of the solution to the Zubov's Dual Equation. One can optionally 
execute 'U_smoothing.py' to utilize a neural network for modifications. 

The 'exe.sh' consecutively excecute 'SolveODE.py', 'ZK_Learning.py', and 'U_prediction.py'. The ZK-operator learning utilizes the 
well-known extended dynamic mode decomposition (EDMD) approach, whereas U_prediction is by iterating the learned ZK operator on the 
chozen dictionary functions. 
