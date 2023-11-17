from normModels import *
import pandas as pd
import unittest


class TestNorms(unittest.TestCase):
    def test_2dim_L1(self) -> None:
        data = np.load('../data/A04plotregres.npz')
        x = data['x']
        y = data['y']
        model = L1Model(y, np.array([x]))
        model.solve()
        self.assertEqual(model.beta[0], 0)
        self.assertEqual(model.beta[1], 1.9492753623188408)

    def test_2dim_Linfty(self) -> None:
        data = np.load('../data/A04plotregres.npz')
        x = data['x']
        y = data['y']
        model = LInfModel(y, np.array([x]))
        model.solve()
        self.assertEqual(model.beta[0], 15.454545454545439)
        self.assertEqual(model.beta[1], 1.7045454545454546)

    def test_6dim_L1(self) -> None:
        data = pd.read_csv('..\data\A04wine.csv')
        y = data['Price']
        x = data[['WinterRain', 'AGST', 'HarvestRain', 'Age', 'FrancePop']]
        x = x.to_numpy().transpose()
        model = L1Model(y, x)
        model.solve()
        self.assertEqual(model.beta[0], 0)
        self.assertEqual(model.beta[1], 0.0006015209106611832)
        self.assertEqual(model.beta[2], 0.3800601798041318)
        self.assertEqual(model.beta[3], 0)
        self.assertEqual(model.beta[4], 0.023086012257242494)
        self.assertEqual(model.beta[5], 0)

    def test_6dim_LInf(self) -> None:
        data = pd.read_csv('..\data\A04wine.csv')
        y = data['Price']
        x = data[['WinterRain', 'AGST', 'HarvestRain', 'Age', 'FrancePop']]
        x = x.to_numpy().transpose()
        model = LInfModel(y, x)
        model.solve()
        self.assertEqual(model.beta[0], 0)
        self.assertEqual(model.beta[1], 0.0013845110530383996)
        self.assertEqual(model.beta[2], 0.3774389084213748)
        self.assertEqual(model.beta[3], 0)
        self.assertEqual(model.beta[4], 0.0015202385175206966)
        self.assertEqual(model.beta[5], 0)
