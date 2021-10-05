from model.KNNClassifier import KDTreeNode
import model.KNNClassifier as knn
import numpy as np
import matplotlib.pyplot as plt

def printKDTree2d(r,xmin=0,xmax=100,ymin=0,ymax=100):
    if r == None:
        return
    x = r.coordinate[0]
    y = r.coordinate[1]

    if r.axis % 2 == 0:
        plt.vlines(x,ymin=ymin,ymax=ymax)
        printKDTree2d(r.left,xmin=xmin,xmax=x,ymin=ymin,ymax=ymax)
        printKDTree2d(r.right,xmin=x,xmax=xmax,ymin=ymin,ymax=ymax)
    else :
        plt.hlines(y,xmin=xmin,xmax=xmax)
        printKDTree2d(r.left, xmin=xmin, xmax=xmax, ymin=ymin, ymax=y)
        printKDTree2d(r.right, xmin=xmin, xmax=xmax, ymin=y, ymax=ymax)

    plt.scatter(x, y)


n_sample = 11
X = np.random.randint(0,100,size=(n_sample,2))
#plt.scatter(X[...,0],X[...,1])
#plt.vlines(X[...,0],ymin=X[...,1],ymax=100)
#plt.show()

#排序
#sorted = X[np.argsort(X[...,0])]
#print(sorted)

root = knn.construct(X)
printKDTree2d(root)
plt.show()



#print(X)