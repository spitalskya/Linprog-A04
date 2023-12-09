import numpy as np
from scipy.optimize import linprog
import pandas as pd
#importing data and creating dataframes
data = pd.read_csv('data\A04wine.csv')

# Separate dependent variable (y) and independent variables (x)
y = data['Price']
x = data[['WinterRain','AGST', 'HarvestRain', 'Age', 'FrancePop']]
# Calculate the number of variables (features) plus 1 for the intercept term
numberOfVariablesBeta = x.shape[1] + 1

#---------------------------------------------------------------------------------------------------------------------------------------

# Formulating the linear programming problem for l1 norm (minimizing the sum of absolute values of coefficients)
c = np.array([0]*numberOfVariablesBeta + [1] * len(x.values)) # Objective function coefficients

ALeft = np.matrix([ [1] * len(x.values)]).transpose() # Coefficients for beta0
ARight = np.matrix(x.values) # Coefficients for other independent variables beta
A = np.block([ALeft, ARight]) # Concatenate coefficients of variables into one matrix

I = np.identity(len(x.values)) # Identity matrix for intercept term

# Formulate inequality constraints for l1 norm
A_ub = np.block([[-A, -I], [A, -I]])
b_ub = np.concatenate([-y, y])

# Solve the linear programming problem for l1 norm
solve = linprog(c, A_ub, b_ub, bounds = [(None,None)]*numberOfVariablesBeta +[(0, None)] * len(x.values))

# extract variables(betas) and print them out
betas = solve.x[:numberOfVariablesBeta]
print(betas)

#---------------------------------------------------------------------------------------------------------------------------------------

# Formulating the linear programming problem for l-inf norm (minimizing the maximum of absolute values of coefficients)
c_inf = np.array([0]*numberOfVariablesBeta + [1]) # Objective function coefficients

A_inf = np.block([ALeft, ARight]) # Coefficients for independent variables for l-inf norm

i_inf = np.array([ [1] * len(x.values)]).transpose() # Coefficients for the intercept term for l-inf norm

# Formulate inequality constraints for l-inf norm
A_ub_inf = np.block([[-A_inf,-i_inf],[A_inf,-i_inf]])
b_ub_inf = np.concatenate([-y, y]) 

# Solve the linear programming problem for l-inf norm
solve_inf = linprog(c_inf, A_ub_inf, b_ub_inf, bounds=[(None,None)]*numberOfVariablesBeta+[(0,None)])

# extract betas and print them out
betas_inf = solve_inf.x[:numberOfVariablesBeta]
print(betas_inf)
