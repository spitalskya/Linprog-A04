import pandas as pd
import numpy
df = pd.read_csv('data/A04wine.csv')
df = df.to_numpy()
numpy.save('data/A04wine', df)