import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('D:/UCLA/156/Math_156_project/data/train.csv')
pd.set_option('display.max_columns', None)
df.head()

# split data into numerical and categorical:
df_num = df.select_dtypes(include='int64')
df_num.head()


df_cat = df.select_dtypes(include='object')
df_cat['SalePrice'] = df['SalePrice']
df_cat.head()

# plot correlation:
plt.figure()
sns.heatmap(df_num.corr(), cmap = 'coolwarm')
plt.show()

df_num.columns.values