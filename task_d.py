import numpy as np
from scipy.optimize import linprog
import pandas as pd


# load data
data = pd.read_csv('data/A04wine.csv')
y = data['Price']
x = data[['WinterRain', 'AGST', 'HarvestRain', 'Age', 'FrancePop']]


#---------------------------------------------------------------------------------------------------------------------------------------
# we solve for beta coefficients for both LP problems in the same way as in task_c.py

numberOfVariablesBeta = x.shape[1] + 1
c = np.array([0]*numberOfVariablesBeta + [1] * len(x.values))
ALeft = np.matrix([ [1] * len(x.values)]).transpose()
ARight = np.matrix(x.values)

A = np.block([ALeft, ARight])
I = np.identity(len(x.values))
A_ub = np.block([[-A, -I], [A, -I]])
b_ub = np.concatenate([-y, y])
solve = linprog(c, A_ub, b_ub, bounds = [(None,None)]*numberOfVariablesBeta +[(0, None)] * len(x.values))
betas = solve.x[:numberOfVariablesBeta]

c_inf = np.array([0]*numberOfVariablesBeta + [1])
A_inf = np.block([ALeft, ARight])

i_inf = np.array([ [1] * len(x.values)]).transpose()
A_ub_inf = np.block([[-A_inf,-i_inf],[A_inf,-i_inf]])
b_ub_inf = np.concatenate([-y, y]) 
solve_inf = linprog(c_inf, A_ub_inf, b_ub_inf, bounds=[(None,None)]*numberOfVariablesBeta+[(0,None)])
betas_inf = solve_inf.x[:numberOfVariablesBeta]

#---------------------------------------------------------------------------------------------------------------------------------------


# define function for calculating R^2 coefficient

def r_squared(x: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    # calculate y-hat and mean of y vector
    y_hat = beta[0] + x @ beta[1:]
    y_mean = np.mean(y)

    res1 = 0    # partial result for the numerator in the formula
    res2 = 0    # partial result for the denominator in the formula

    # calculate the sums
    res1 = np.sum((y - y_hat)**2)
    res2 = np.sum((y - y_mean)**2)

    # calculate the R^2 coefficient
    result = 1 - (res1 / res2)
    return result


# calculate R-squared coefficients for both regressions
print(f'R-squared for L1 regression on wine data: {r_squared(x, y, betas)}')         # 0.78813 
print(f'R-squared for LInf regression on wine data: {r_squared(x, y, betas_inf)}')     # 0.80649
