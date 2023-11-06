import numpy as np
from scipy.optimize import linprog
import pandas as pd

data = pd.read_csv('data\A04wine.csv')

y = data['Price']
x = data[['WinterRain','AGST', 'HarvestRain', 'Age', 'FrancePop']]
numberOfVariablesBeta = x.shape[1] + 1

#l1 norm
c = np.array([0]*numberOfVariablesBeta + [1] * len(x.values))
ALeft = np.matrix([ [1] * len(x.values)]).transpose()
ARigth = np.matrix(x.values)
A = np.block([ALeft, ARigth])
I = np.identity(len(x.values))

A_ub = np.block([[-A, -I], [A, -I]])
b_ub = np.concatenate([-y, y])
solve = linprog(c, A_ub, b_ub)

bethas = solve.x[:numberOfVariablesBeta]
print(bethas)

#l inf norm
c_inf = np.array([0]*numberOfVariablesBeta + [1])
A_inf = np.block([ALeft, ARigth])
i_inf = np.array([ [1] * len(x.values)]).transpose()

A_ub_inf = np.block([[-A_inf,-i_inf],[A_inf,-i_inf]])
b_ub_inf = np.concatenate([-y, y])
solve_inf = linprog(c_inf, A_ub_inf, b_ub_inf)

bethas_inf = solve_inf.x[:numberOfVariablesBeta]
print(bethas_inf)