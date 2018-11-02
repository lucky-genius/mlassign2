# usage：
# from recommendation import *
# R_train, R_test = make_R(filename = 'movie_ratings.csv', col_user = 'userId', col_item = 'movieId', col_rating = 'rating')
# mf = matrix_factorization(R_train, R_test)
# mf.train(k = 300, alpha = 0.01, _lambda = 0.01, iterations = 20)
# mf.result
# mf.mse()
# rs = correlation_similarity(R_train, R_test)
# rs.test(3)

import numpy as np
import time
from astropy.io import ascii
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

def make_R(filename, col_user, col_item, col_rating, fraction = 0.8):
    '''create R matrix from a file that contains a table of user IDs, item IDs and ratings
    parameters:
        filename: str; the filename of the table
        col_user: str; the column name of user IDs
        col_item: str; the column name of item IDs
        col_rating: str; the column name of the ratings
        fraction: float; the fraction of data to be used for training
    returns:
        R: 2D numpy array; R matrix in with r_{i, j} is the rating that user i given to item j'''
    t = ascii.read(filename)
    nData = np.size(t) # total number of data points(ratings)
    nUser = np.max(t[col_user]) # total number of users
    nitem = np.max(t[col_item]) # total number of items
    
    idxData = np.arange(nData) # an array of idx of data randomized
    np.random.shuffle(idxData)
    
    nTrain = int(np.rint(fraction * nData)) # number of training data
    print(nTrain)
    idxTrain, idxTest = idxData[:nTrain], idxData[nTrain:]
    
    idxUser = t.index_column(col_user)
    idxItem = t.index_column(col_item)
    idxRating = t.index_column(col_rating)
    
    rTrain = np.zeros(shape = (nUser, nitem))
    rTest = np.zeros(shape = (nUser, nitem))
    for nrow in idxTrain:
        i = t[nrow][idxUser] - 1 # the userId
        j = t[nrow][idxItem] - 1 # the itemId
        rTrain[i][j] = t[nrow][idxRating] # the rating
    
    for nrow in idxTest:
        i = t[nrow][idxUser] - 1 # the userId
        j = t[nrow][idxItem] - 1 # the itemId
        rTest[i][j] = t[nrow][idxRating] # the rating
    
    return rTrain, rTest


class matrix_factorization():

    def __init__(self, R_train, R_test):
        '''R_train: 2D numpy array; R matrix in with r_{i, j} is the rating that user i given to item j'''
        self.R_train = R_train
        self.R_test = R_test
        self.nUser, self.nItem = R_train.shape
        self.result = None

    def train(self, k, alpha, _lambda, iterations):
        '''perform matrix factorization to predict empty entries in a matrix.
        parameters:
            k: int; dimensions of u and v vector
            alpha: float; learning rate
            _lambda: float; regularization parameter'''
        self.k = k
        self.alpha = alpha
        self.iterations = iterations
        self._lambda = _lambda
        
        def update():
            for i, j, r in self.samples:
                # Computer prediction and error
                if _lambda is not None:
                    prediction = self.b + self.b_u[i] + self.b_i[j] + self.U[i, :].dot(self.V[j, :].T)
                    e = (r - prediction)
                    # Update biases
                    self.b_u[i] += self.alpha * (e - self._lambda * self.b_u[i])
                    self.b_i[j] += self.alpha * (e - self._lambda * self.b_i[j])

                    # Update user and item latent feature matrices
                    self.U[i, :] += self.alpha * (e * self.V[j, :] - self._lambda * self.U[i,:])
                    self.V[j, :] += self.alpha * (e * self.U[i, :] - self._lambda * self.V[j,:])
                else:
                    prediction = self.U[i, :].dot(self.V[j, :].T)
                    e = (r - prediction)
                    # Update user and item latent feature matrices
                    self.U[i, :] += self.alpha * e * self.V[j, :]
                    self.V[j, :] += self.alpha * e * self.U[i, :]
                    
        start_time = time.time()
        # Initialize user and item latent feature matrice
        self.U = np.random.normal(scale = 1./self.k, size = (self.nUser, self.k))
        self.V = np.random.normal(scale = 1./self.k, size = (self.nItem, self.k))
        
        if _lambda is not None:
            # Initialize the biases
            self.b_u = np.zeros(self.nUser)
            self.b_i = np.zeros(self.nItem)
            self.b = np.mean(self.R_train[np.where(self.R_train != 0)])

        # Create a list of training samples
        self.samples = [
            (i, j, self.R_train[i, j])
            for i in range(self.nUser)
            for j in range(self.nItem)
            if self.R_train[i, j] > 0
        ]

        # Perform stochastic gradient descent for number of iterations
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            update()
                
        if _lambda is not None:
            self.result = self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.U.dot(self.V.T)
        else:
            self.result = self.U.dot(self.V.T)
        
        sigma_train, sigma_test = self.sigma()
        print('training is complete! it took %.2f s' % (time.time() - start_time))
        print("train error = %.4f; test error = %.4f" % (sigma_train, sigma_test))
        return self.result
    
    def sigma(self):
        '''Compute the total mean square error for training and testing data'''
        iTrain, jTrain = self.R_train.nonzero()
        iTest, jTest = self.R_test.nonzero()
        trainError, testError = 0, 0
        trainCount, testCount = 0, 0
        for i, j in zip(iTrain, jTrain):
            trainError += (self.R_train[i, j] - self.result[i, j])**2
            trainCount += 1
            
        for i, j in zip(iTest, jTest):
            testError += (self.R_test[i, j] - self.result[i, j])**2
            testCount += 1
        
        trainSigma = np.sqrt(trainError)/trainCount
        testSigma = np.sqrt(testError)/testCount
        return trainSigma, testSigma
        
class correlation_similarity():
    def __init__(self, R_train, R_test):
        '''initialize
        R_train: 2D numpy array; R matrix in with r_{i, j} is the rating that user i given to item j
        R_test: 2D numpy array; R matrix in with r_{i, j} is the rating that user i given to item j'''
        self.R_train = R_train
        self.R_test = R_test
        self.nUser, self.nItem = R_train.shape
        self.result = None
        self.S = cosine_similarity(sparse.csr_matrix(R_train)) # similarity 
        self.avg = np.mean(R_train)
        
    def predict(self, i, j, k):
        '''predict the rating for user i of item j
        i: int; the user number
        j: int; the item number
        k: int; use k nearest neighbors'''
        if R_train[i][j] != 0:
            return R_train[i][j]
        else:
            idxS_user = np.argsort(self.S[i])
            idx_n = np.where(self.R_train[:, j] > 0)[0]

            if np.size(idx_n) < k:
                if np.size(idx_n) == 0: 
#                     idxS_item = np.argsort(self.S_item[j]) # similarity between items
                    v_j = self.R_train[:, j]
                    prediction = v_j[v_j != 0].mean()
                else:
                    prediction = self.R_train[idx_n, j].mean()
            else:
                idx_knn = idxS_user[np.isin(idxS_user, idx_n, assume_unique = True)][-k - 1:-1] # idxS with ratings
                prediction = self.R_train[idx_knn, j].mean()
            if np.isnan(prediction):
                prediction = self.avg
            return prediction

    def sigma2(self, i, j, k):
        if R_test[i][j] == 0:
            print('No true value!')
            return None
        else:
            r_hat = self.predict(i, j, k)
            return (R_test[i][j] - r_hat)**2
        
    def test(self, k):
        start_time = time.time()
        iTest, jTest = self.R_test.nonzero()
        totalError = 0
        testCount = 0
        for i, j in zip(iTest, jTest):
            totalError += self.sigma2(i, j, k)
            testCount += 1
        sigma = np.sqrt(totalError)/testCount         
        print('testing is complete! it took %.2f s' % (time.time() - start_time))
        print('the test error is %.4f' % sigma)
        return sigma