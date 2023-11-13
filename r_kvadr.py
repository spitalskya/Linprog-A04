from typing import Union
import numpy as np
import pandas as pd
from Task_E.normModels import L1Model, LInfModel

def r_kvadr(path: str, norm: Union[L1Model, LInfModel]) -> int:
    data = np.load(path)
    x = data['x']
    y = data['y']
    model = norm(y, np.array([x]))
    betas = model.solve()
    y_hat = [i * betas[1] + betas[0] for i in x]
    y_mean = np.mean(y)
    res1 = np.sum((np.array(y) - np.array(y_hat)) ** 2)
    res2 = np.sum((np.sum((np.array(y) - y_mean) ** 2)))
    result = 1 - (res1 / res2)
    return result



print(r_kvadr('data/A04plotregres.npz', L1Model))