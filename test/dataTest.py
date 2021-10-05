import numpy as np
from sklearn.linear_model import LinearRegression
from model.LinearRegression import LinearRegressionModel

X = np.asarray([[1,2,3,4,5,6],
                [2,3,4,5,6,7]])
X = np.transpose(X)
y = np.asarray([[2,1,4,3,6,5]])
y = np.transpose(y)

m = LinearRegression()
m.fit(X,y)

m_my = LinearRegressionModel()
m_my.fit(X,y)

print(m_my.coef_,'\n')
print(m.coef_)
