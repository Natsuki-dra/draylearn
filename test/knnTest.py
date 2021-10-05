import numpy as np
import model.KNNClassifier as knn
import matplotlib.pyplot as plt
X = np.asarray([[1,23],
                [2,25],
                [56,3],
                [34,3]])
y = np.asarray([[1],
                [1],
                [0],
                [0]])

c = knn.KNNClassifier(3)
c.fit(X,y)

plt.scatter(X[...,0],X[...,1],s=y[...,0],c='g')
plt.scatter(X[...,0],X[...,1],s=np.abs(1-y[...,0]),c='b')
plt.show()
c.predict([3,34])
