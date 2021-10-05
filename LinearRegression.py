import numpy as np


class LinearRegressionModel:
    def __init__(self):
        print('This is a LinearRegression Model')
        self.X = None
        self.y = None
        self.coef_ = None
        self.intercept_ = None


    def fit(self,X_train,y_train):
        """
        :param X_train: 2d Matrix   shape:(n,m)
        :param y_train: vector  shape:(n,1)
        :return: None
        """
        #print('train the dataset')

        X = X_train
        Xt = np.transpose(X_train)

        y = y_train
        #self.X = X
        #self.y = y
        n = X_train.shape[0]
        m = X_train.shape[1]
        #print('n : ',n,' m : ',m)

        # the first item    XT·X
        A = Xt.dot(X)
        #print('A',A)
        # the second item   n·xaT·xa
        xa = np.mean(X,axis=0).reshape(1,m)
        xaT = np.transpose(xa)
        B = xaT.dot(xa) * n
        #print('B',B)

        # the third item
        C = Xt.dot(y)
        #print('C',C)
        # the fourth item
        ya = np.mean(y)
        D = xaT * ya * n
        #print('D',D)
        #solve the linear formula
        amb = A-B
        cmd = C-D
        #print(amb,'\n', cmd)

        #prevent irreversible
        row,col = np.diag_indices_from(amb)
        #diag = np.diagonal(amb,axis=0)
        amb[row,col] += 1e-6

        k = np.linalg.solve(amb,cmd)
        self.coef_ = k
        b = ya - xa.dot(k)
        self.intercept_ = b
        #print('k is ',k,' b is ',b)
        return self


    def predict(self,xi):
        #varify the model validation
        if self.coef_ == None:
            print('your model hasn\'t been trained! please use model.fit(X,y) ')
            return

        return xi.dot(self.coef_) + self.intercept_

    def score(self,test_X,test_y):
        if self.coef_ == None:
            print('your model hasn\'t been trained! please use model.fit(X,y) ')
            return

        y = test_X.dot(self.coef_) + self.intercept_
        dist = y - test_y
        square = dist * dist
        e = np.sum(square)
        return e