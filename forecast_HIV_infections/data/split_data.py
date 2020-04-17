import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('main_data.csv', index_col=0)

X = data.drop(columns='HIVincidence').values
y = data['HIVincidence'].values

X_train, X_test, y_train, y_test = train_test_split(X,y)

data.columns.to_series().to_csv('column_names.csv', index=0)

pd.DataFrame(X_train, columns=data.columns.drop('HIVincidence')).to_csv('x_train.csv')
pd.DataFrame(X_test, columns=data.columns.drop('HIVincidence')).to_csv('x_test.csv')
pd.DataFrame(y_train, columns=['HIVincidence']).to_csv('y_train.csv')
pd.DataFrame(y_test, columns=['HIVincidence']).to_csv('y_test.csv')
