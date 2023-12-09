import numpy as np
import matplotlib.pyplot as plt
from models import L1Model, LInfModel, L1LInfModel
from DataGenerator import DataGenerator


def data_2d(start=10, finish=200, size=40, noise=(100, 500)):
    generator = DataGenerator()
    return generator.gen_2d_data(start=start, finish=finish, size=size, noise=noise)


def data_3d(start=10, finish=200, size=40, noise=(100, 500)):
    generator = DataGenerator()
    return generator.gen_3d_data(start=start, finish=finish, size=size, noise=noise)


def comparison(x, y, model1, model2, text='', save_loc=''):
    fig, axes = plt.subplots()
    axes.scatter(x, y)

    axes.set_xlim((0, None))
    axes.set_ylim((0, None))

    beta0, beta1 = model1.beta
    left, right = axes.get_xlim()
    reg = [left * beta1 + beta0, right * beta1 + beta0]
    axes.plot([left, right], reg, label=f'{model1.model_type}')

    beta0, beta1 = model2.beta
    left, right = axes.get_xlim()
    reg = [left * beta1 + beta0, right * beta1 + beta0]
    axes.plot([left, right], reg, label=f'{model2.model_type}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)

    plt.title(text)
    if save_loc == '':
        plt.show()
    else:
        plt.savefig(save_loc)


def generate_complots():
    np.random.seed(20)
    data = data_2d(start=10, finish=600, size=40, noise=(10, 50))
    # small noise
    model1 = L1Model(data[0], np.array([data[1]]))
    model1.solve()

    model2 = LInfModel(data[0], np.array([data[1]]))
    model2.solve()

    comparison(data[1], data[0], model1, model2, 'Noise ranging from 10 to 50', 'comparison_plots/low_noise.png')
    # -----------------------------------------------------------------
    # bigger noise
    data = data_2d(start=10, finish=600, size=40, noise=(100, 500))

    model1 = L1Model(data[0], np.array([data[1]]))
    model1.solve()

    model2 = LInfModel(data[0], np.array([data[1]]))
    model2.solve()

    comparison(data[1], data[0], model1, model2, 'Noise ranging from 100 to 500', 'comparison_plots/big_noise.png')
    # ------------------------------------------------------------------
    # adding outliers
    data = data_2d(start=10, finish=600, size=40, noise=(10, 50))

    x = np.append(data[1], [max(data[1] + 7000)])
    y = np.append(data[0], [1000])

    model1 = L1Model(y, np.array([x]))
    model1.solve()
    model2 = LInfModel(y, np.array([x]))
    model2.solve()

    comparison(x, y, model1, model2, 'Adding outlier', 'comparison_plots/outlier.png')
    print('generated 2d plots')


def plot_3d():
    data = data_3d()
    model1 = L1Model(data[2], np.array([data[0], data[1]]))
    model1.solve()
    model1.visualize('comparison_plots/3D_L1.png')

    model2 = LInfModel(data[2], np.array([data[0], data[1]]))
    model2.solve()
    model2.visualize('comparison_plots/3D_Linf.png')
    print('generated 3D plots')


def linear_with_outlier():
    # plot graph of linear data with one outlier with L1 regression line
    data = [[x for x in range(20, 100, 10)] + [100], 
            [2*y for y in range(20, 100, 10)] + [300]]
    
    model1 = L1Model(np.array(data[1]), np.array([data[0]]))
    model1.solve()
    model1.visualize('models/behavior/linear_with_outlier.png')

def random_with_colinear():
    # plot graph of random data with some colinear data with L1 regression line
    data = data_2d()
    
    x = list(data[0]) + [i for i in range(min(data[0]), max(data[0]), (max(data[0]) - min(data[0])) // 15)]
    y = list(data[1]) + [2*i + min(data[1]) for i in range(min(data[0]), max(data[0]), (max(data[0]) - min(data[0])) // 15)]
    
    model1 = LInfModel(np.array(y), np.array([x]))
    model1.solve()
    model1.visualize('models/behavior/random_with_colinear.png')


def minimizing_both_norms():
    data = [[x for x in range(20, 100, 10)] + [100], 
        [2*y for y in range(20, 100, 10)] + [300]]


    for i in range(0, 11):
        model1 = L1LInfModel(np.array(data[1]), np.array([data[0]]))
        model1.solve(omega=i / 100 + 0.3)
        model1.visualize(f'models/behavior/minimizing_both_linear_with_outlier{i}.png')

def plot_behavior():
    # linear_with_outlier()
    # random_with_colinear()
    minimizing_both_norms()
    
if __name__ == '__main__':
    plot_behavior()
    """
    generate_complots()
    plot_3d()
    """