import numpy as np
import model.LinearRegression as lr

from sklearn.linear_model import LinearRegression

X = np.asarray([[230.1,37.9,69.2],
                [44.5,39.3,69.2],
                [17.2,45.9,69.3],
                [151.5,41.3,58.5],
                [190.9,10.8,58.4]])
#X = np.transpose(X)
y = np.asarray([[22.1,10.4,9.3,18.5,12.9]])
y = np.transpose(y)

m_sk = LinearRegression()
m_sk.fit(X,y)

m = lr.LinearRegressionModel()
m.fit(X,y)


print(m_sk.coef_)
ytest = m.predict([X[0,...]])
ytestsk = m_sk.predict([X[0,...]])
print(ytest)
print(ytestsk)