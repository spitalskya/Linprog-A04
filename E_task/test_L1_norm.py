from norm_l1 import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    if __name__ == '__main__':
        data = np.load('../data/A04plotregres.npz')
        x = data['x']
        y = data['y']
        model = L1Model(y, np.array([x]))
        model.solve()
        print('Beta values are:', model.beta)
        fig, ax = plt.subplots()

        ax.set_xlim([0, max(x) + 5])
        ax.set_ylim([0, max(y) + 5])
        reg_values = [i * model.beta[1] + model.beta[0] for i in x]
        ax.plot(x, reg_values, label='Manhattan')
        ax.plot(x, y, 'o')
        ax.legend()
        plt.show()
