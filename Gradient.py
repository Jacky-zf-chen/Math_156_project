from asyncio import transports
from cProfile import label
from os import confstr_names
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# read df_imputed data
df = pd.read_csv("D:/UCLA/156/Math_156_project/data/df_imputed.csv")

def vectorXwith0 (X):
    newX = np.c_[np.ones((len(X),1)), X]
    return newX

def a_init(X):
    A = np.random.randn(len(X[0]) + 1, 1)
    return A

def MLR (X, Y, learning_rate, iteration):
    newY = np.reshape(Y, (len(Y),1))
    cost = []
    newX = vectorXwith0(X)
    A = a_init(X)
    m = len(X)
    total = 0
    for i in range(iteration):
        gradient = 1/m * newX.T.dot(newX.dot(A) - newY)
        A = A - learning_rate * gradient
        Y_pre = newX.dot(A)
        cost_value = np.linalg.norm(Y_pre - Y)
        total += cost_value
        cost.append(total)
    plt.semilogy(np.asarray(range(0,iteration)), cost, color = 'red', linewidth = 3, label = 'cost')
    plt.legend()
    plt.show()
    return A



x = df[['Neighborhood','Condition2','OverallQual','OverallCond','YearBuilt','RoofMatl','BsmtExposure','LowQualFinSF','BsmtFullBath','Functional']]
x

# convert categorical variables to numerical:
num = x.select_dtypes(['number']).columns
num
x[num]
cat = x.drop(num,axis =1)
cat = cat.columns
x[cat] = x[cat].astype('category')
x[cat] =x[cat].apply(lambda x: x.cat.codes)
x
y = df[['SalePrice']]
y

# scale the predictors:
sc = StandardScaler()
X = sc.fit_transform(x)
x

# Apply with gradient descend:
A = MLR(X, y, 0.03, 30000)
A.T

# check with the sklearn linear regression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_





















