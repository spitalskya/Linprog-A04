import numpy as np
import matplotlib.pyplot as plt
from models import L1Model, LInfModel
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


if __name__ == '__main__':
    generate_complots()
    plot_3d()
