# importing required libraries
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

#loading data
data = np.load('data/A04plotregres.npz')

x = data['x']
y = data['y']

# Manhattan norm
c = np.array([0, 0] + [1] * len(x))      # objective function vector, two zeros stand for betas
A = np.matrix([[1] * len(x), x]).transpose()  
I = np.identity(len(x))     # Identity matrix
bounds = [(None, None), (None, None)] + [(0, None) for _ in range(len(x))] # bounds for variables, the first two tuples set betas real, other variables are positive
A_ub = np.block([[-A,-I], [A,-I]])      # creating a block matrix 
b_ub = np.concatenate([-y, y])          # right side vector
solve = linprog(c, A_ub, b_ub, bounds = bounds)     # solving the linear programming problem with solver from scipy
beta0 = solve.x[0]          # extracting betas
beta1 = solve.x[1] 
fig, ax = plt.subplots()    # creating a plot
ax.plot(x, y, 'o')          # plotting given points

# Chebysev's norm
c_inf = np.array([0, 0, 1])       # objective function vector
A_inf = np.matrix([[1] * len(x), x]).transpose() 
i_inf = np.array([[1] * len(x)]).transpose()    # vector of ones
A_ub_inf = np.block([[-A_inf, -i_inf], [A_inf, -i_inf]])    # creating a block matrix
b_ub_inf = np.concatenate([-y, y])      # right side vector
bounds = [(None, None), (None, None), (0, None)]
solve_inf = linprog(c_inf, A_ub_inf, b_ub_inf, bounds=bounds)   # solving the problem
beta0_inf = solve_inf.x[0]      # extracting betas
beta1_inf = solve_inf.x[1]

x = list(range(36)) + list(x) + list(range(119, 150))   # making x range from zero to maximal given value
values = [i*beta1 + beta0 for i in x]       # list of values predicted using Manhattan norm
ax.plot(x, values, label = 'Manhattan norm')        # plotting a line 
ax.set_xlim([0, max(x) + 5])    # setting limits of x
ax.set_ylim([0, max(y) + 50])   # setting limits of y (price starts at 0 units)

values_inf = [i*beta1_inf + beta0_inf for i in x]   # list of values predicted using Chebysev's norm
ax.plot(x, values_inf, label = 'Chebysev\'s norm')  # plotting a line
ax.legend()     # showing the legend
plt.grid(True, linestyle='--', alpha=0.7)   # plotting a grid
ax.spines['top'].set_visible(False)         # hiding the right and top axis
ax.spines['right'].set_visible(False)
plt.title("The comparison between norms")   # setting a title
plt.show()      # showing the whole plot
