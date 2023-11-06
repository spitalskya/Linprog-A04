import numpy as np
from scipy.optimize import linprog
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data_vino.csv')
#print(data)

data2 = np.load('data2_vino.npz')
# print(data2['x'])
# print('oddelenie')
# print(data2['y'])


x = data2['x']
y = data2['y']
c = np.array([0,0] + [1] * len(x))
A = np.matrix([[1] * len(x), x]).transpose()
I = np.identity(len(x))
A_ub = np.block([[-A,-I], [A,-I]])
#print(A_ub)
b_ub = np.concatenate([-y, y])

solve = linprog(c, A_ub, b_ub)
#print(solve.x)
beta1 = solve.x[0]
beta2 = solve.x[1]
values = [i*beta2 + beta1 for i in x]

plt.plot(x,y, 'o')
plt.plot(x, values)
plt.show()
