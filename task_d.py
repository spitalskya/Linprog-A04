import pandas as pd
from models.models import L1Model, LInfModel


# load data
data = pd.read_csv('data/A04wine.csv')
y = data['Price']
x = data[['WinterRain', 'AGST', 'HarvestRain', 'Age', 'FrancePop']]
x = x.to_numpy().transpose()

# utilize developed L1 and LInf regression model classes
l1_model = L1Model(y, x)
linf_model = LInfModel(y, x)

# solve LP problems
l1_model.solve()
linf_model.solve()

# calculate R-squared coefficient
print(f'R-squared for L1 regression on wine data: {l1_model.r2()}')
print(f'R-squared for LInf regression on wine data: {linf_model.r2()}')
