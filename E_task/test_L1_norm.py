from norm_l1 import *
import pandas as pd

if __name__ == '__main__':
    data = np.load('../data/A04plotregres.npz')
    x = data['x']
    y = data['y']
    model = L1Model(y, np.array([x]))
    model.solve()
    print('Beta values are:', model.beta)

    data = pd.read_csv('..\data\A04wine.csv')
    y = data['Price']
    x = data[['WinterRain', 'AGST', 'HarvestRain', 'Age', 'FrancePop']]
    x = x.to_numpy().transpose()
    model = L1Model(y, x)
    print(model.solve())

