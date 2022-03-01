import pandas as pd
import numpy as np
from sklearn.feature_selection import f_regression


# read the data in to dataframe:
df = pd.read_csv("D:/UCLA/156/Math_156_project/data/train.csv")
df.head()

x = df.drop(['Id', 'SalePrice'], axis=1)
y = df['SalePrice']

x.head()
y.head()

x_num = df.select_dtypes(include=np.number)

(F, pval) = f_regression(x_num, y)
display(F, pval)