import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import unittest
from models import L1Model, LInfModel


class TestModels2D(unittest.TestCase):

    def setUp(self) -> None:
        data = np.load('../data/A04plotregres.npz')
        self.l1 = L1Model(data['y'], np.array([data['x']]))
        self.linf = LInfModel(data['y'], np.array([data['x']]))

    def test_not_solved(self) -> None:
        self.assertEqual([], self.l1.beta)
        self.assertEqual([], self.linf.beta)

    def test_visualize(self) -> None:
        self.l1.solve()
        self.linf.solve()

        self.assertTrue(self.l1.visualize())


class TestModelsWine(unittest.TestCase):

    def setUp(self) -> None:
        data = pd.read_csv('..\data\A04wine.csv')
        y = data['Price']
        x = data[['WinterRain', 'AGST', 'HarvestRain', 'Age', 'FrancePop']]
        x = x.to_numpy().transpose()
        self.l1 = L1Model(y, x)
        self.linf = LInfModel(y, x)

    def test_not_solved(self) -> None:
        self.assertEqual([], self.l1.beta)
        self.assertEqual([], self.linf.beta)

    def test_visualize(self) -> None:
        self.l1.solve()
        self.linf.solve()

        self.assertFalse(self.l1.visualize('comparison_plots/testWine.png'))

