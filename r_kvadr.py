from typing import Union
import numpy as np
import pandas as pd
from models.models import L1Model, LInfModel

class Model:
    def r_kvadr(self, path: str, norm: Union[L1Model, LInfModel], xcols: list) -> float:
        data = pd.read_csv('Linprog-A04/'+path)
        
        y = np.array(data['Price'])
        x = data[xcols].to_numpy().transpose()
        
        model = norm(y, x)
        betas = model.solve()

        y_hat = betas[0] + np.dot(x.transpose(), betas[1:])
        y_mean = np.mean(y)

        res1 = np.sum((y - y_hat) ** 2)
        res2 = np.sum((y - y_mean) ** 2)
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
x_names = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
