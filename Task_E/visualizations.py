import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
from normModels import L1Model, LInfModel
from DataGenerator import DataGenerator


def data_2d(start=10, finish=200, size=40, noise=(100, 500)):
    generator = DataGenerator()
    return generator.gen_2d_data(start=start, finish=finish, size=size, noise=noise)


def data_3d(start=10, finish=200, size=40, noise=(100, 500)):
    generator = DataGenerator()
    return generator.gen_3d_data(start=start, finish=finish, size=size, noise=noise)


def reg_plot2d(x, y, solved_model, ax):
    # ax.scatter(x, y)
    beta0, beta1 = solved_model.beta
    reg = [i * beta1 + beta0 for i in x]
    ax.plot(x, reg)

def reg_plot3d(x1, x2, y, solved_model, ax):
    ax.scatter(x1, x2, y, c='r')
    beta0, beta1, beta2 = solved_model.beta
    x1, x2 = np.meshgrid(x1, x2)
    reg = beta0 + x1 * beta1 + x2 * beta2

    ax.plot_surface(x1, x2, reg, cmap='Blues')

def random_l1_2dplot(ax, data):
    x, y = data
    model1 = L1Model(y, np.array([x]))
    model1.solve()
    return reg_plot2d(x, y, model1, ax)


def random_linf_2dplot(ax, data):
    x, y = data
    model1 = LInfModel(y, np.array([x]))
    model1.solve()
    return reg_plot2d(x, y, model1, ax)

def random_l1_3dplot(ax, data):
    x1, x2, y = data
    model1 = L1Model(y, np.array([x1, x2]))
    model1.solve()
    return reg_plot3d(x1, x2, y, model1, ax)


if __name__ == '__main__':
    # fig, axes = plt.subplots(2, sharex=True, sharey=True)
    # for i in range(2):
    #     data = data_2d()
    #     random_l1_2dplot(axes[i], data)
    #     random_linf_2dplot(axes[i], data)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    data = data_3d()
    random_l1_3dplot(ax, data)
    plt.show()





