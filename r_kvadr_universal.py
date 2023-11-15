from typing import Union
import numpy as np
import pandas as pd
from Task_E.normModels import L1Model, LInfModel

class Model:
    def r_kvadr(self, x, y, norm: Union[L1Model, LInfModel]) -> float:        
        model = norm(y, x)
        betas = model.solve()
        
        # x su riadky
        # y je stlpec

        y_hat = betas[0] + np.dot(x.transpose(), betas[1:])
        y_mean = np.mean(y)

        print(y - y_hat)

        res1 = 0
        res2 = 0

        for i in range(len(y)):
            res1 += (y[i] - y_hat[i]) ** 2
            res2 += (y[i] - y_mean) ** 2

        '''res1 = np.sum((y - y_hat) ** 2)
        res2 = np.sum((y - y_mean) ** 2)'''
        result = 1 - (res1 / res2)
        return result

'''
xvectors = ['Year', 'WinterRain', 'AGST', 'HarvestRain', 'Age', 'FrancePop']
model = Model()
r2_l1 = model.r_kvadr('data/A04wine.csv', L1Model, xvectors)
r2_linf = model.r_kvadr('data/A04wine.csv', LInfModel, xvectors)
print(f'R-squared L1: {r2_l1}')
print(f'R-squared LInf: {r2_linf}')
'''

diabetes = pd.read_csv('data/diabetes.csv')
x_names = ['Pregnancies','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
x = diabetes[x_names].to_numpy().transpose()
y = diabetes['Glucose']
model = Model()

r2_l1 = model.r_kvadr(x, y, L1Model)
r2_linf = model.r_kvadr(x, y, LInfModel)

print(f'R-squared L1: {r2_l1}')
print(f'R-squared LInf: {r2_linf}')