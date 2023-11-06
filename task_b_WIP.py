import numpy as np
from scipy.optimize import linprog
import pandas as pd
import matplotlib.pyplot as plt

data = np.load('data/A04plotregres.npz')


x = data['x']
y = data['y']

# l1 norm
c = np.array([0,0] + [1] * len(x))
A = np.matrix([[1] * len(x), x]).transpose()
I = np.identity(len(x))
A_ub = np.block([[-A,-I], [A,-I]])
b_ub = np.concatenate([-y, y])
solve = linprog(c, A_ub, b_ub)
beta0 = solve.x[0]
beta1 = solve.x[1]
values = [i*beta1 + beta0 for i in x]
fig, ax = plt.subplots()


ax.plot(x, values, label = 'Manhattan')
ax.set_xlim([0, max(x) + 5])
ax.set_ylim([0, max(y) + 5])


# l inf norm
c_inf = np.array([0,0,1])
A_inf = np.matrix([[1] * len(x), x]).transpose()
i_inf = np.array([[1] * len(x)]).transpose()
A_ub_inf = np.block([[-A_inf, -i_inf], [A_inf, -i_inf]])
b_ub_inf = np.concatenate([-y, y])
solve_inf = linprog(c_inf, A_ub_inf, b_ub_inf)
beta0_inf = solve_inf.x[0]
beta1_inf = solve_inf.x[1]
values_inf = [i * beta1_inf + beta0_inf for i in x]
ax.plot(x, values_inf, label = 'infinity')
ax.plot(x,y, 'o')
ax.legend()
plt.show()


