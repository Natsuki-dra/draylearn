import math

import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self,k):
        print('KNNClassifier')
        self.X = None
        self.y = None
        self.k = k

    def fit(self,X,y):
        print('train')
        self.X = X
        self.y = y

    def predict(self,xn):
        #gather distance to x

        dists = []

        for i in range(self.X.shape[0]):
            xi = self.X[i,...]  #extract a xi
            #calc the distance
            dist = xi - xn
            dist = dist * dist
            dist = np.sum(dist)
            yi = self.y[i,0]
            dists.append([dist,yi])


        #sort the dists by dist
        dists.sort(key=lambda t:t[0])

        dists = np.asarray(dists)
        k_nearest = dists[0:self.k,:]

        #count the frequency
        stat = Counter(k_nearest[:,1])

        #get the most common class
        ret = stat.most_common(1)[0][0]

        return ret





    def score(self,X,y):
        print('score')



class KNNClassifierKDTree:
    def __init__(self):
        print('init')

    def fit(self):
        print('construct kd space')

    def predict(self):
        print('predict')



def construct(X,axis=0):
    row, dimension = X.shape

    # 设置终止条件
    if row <= 1:
        return

    # 排序，找到中位数
    sorted = X[np.argsort(X[..., axis % dimension])]
    #print('sorted\n',sorted)
    target_line = math.floor(row / 2)

    # log
    #print(sorted[target_line], axis)
    # 递归创建树
    root = KDTreeNode(sorted[target_line], axis)
    root.left = construct(sorted[0:target_line,...], axis=axis + 1)
    root.right = construct(sorted[target_line + 1:row,...], axis=axis + 1)

    return root





class KDTreeNode:
    def __init__(self,coordinate,axis,left=None,right=None):
        self.coordinate = coordinate
        self.axis = axis
        self.left = left
        self.right = right

    def __str__(self):
        return '{ coordinate:'+self.coordinate.__str__() + ',axis:'+self.axis.__str__()+'}'

    def midTraverse(self):
        if self == None:
            return
        if self.left != None:
            self.left.midTraverse()
        print(self)
        if self.right != None:
            self.right.midTraverse()
