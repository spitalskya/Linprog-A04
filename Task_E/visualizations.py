import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from normModels import Model, L1Model, LInfModel
from scipy.optimize import linprog

def reg_plot2d(x, y, solved_model, ax):
    ax.scatter(x, y)
    beta0, beta1 = solved_model.beta
    ax.set_title(np.corrcoef(x, y))
    reg = [i * beta1 + beta0 for i in x]
    ax.plot(x, reg)

if __name__ == '__main__':
    data = pd.read_csv('..\data\A04wine.csv')
    data_combinations = list(itertools.combinations(data.columns, 2))
    fig, axes = plt.subplots(len(data_combinations), 2, figsize=(20, 80))

    for row, combination in enumerate(data_combinations):
        x_axis, y_axis = combination
        x = np.array(data[x_axis])
        y = np.array(data[y_axis])

        model1 = L1Model(y, np.array([x]))
        model1.solve()
        reg_plot2d(x, y, model1, axes[row, 0])

        model2 = LInfModel(y, np.array([x]))
        model2.solve()
        reg_plot2d(x, y, model2, axes[row, 1])

    plt.savefig('compare_models.jpg')




