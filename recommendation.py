# usage：
# from recommendation import *
# R_train, R_test = make_R(filename = 'movie_ratings.csv', col_user = 'userId', col_item = 'movieId', col_rating = 'rating')
# mf = recommendation_system(R_train, R_test, k = 300, alpha = 0.01, _lambda = 0.01, iterations = 20)
# mf.train()
# mf.full_matrix()
# mf.mse()
# i, j = 45, 500
# mf.get_rating(i, j)

import numpy as np
from astropy.io import ascii

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


class recommendation_system():

    def __init__(self, R_train, R_test, k, alpha, _lambda, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        parameters:
            R: 2D numpy array; R matrix in with r_{i, j} is the rating that user i given to item j
            k: int; dimensions of u and v vector
            alpha (float) : learning rate
            _lambda (float)  : regularization parameter
        """

        self.R_train = R_train
        self.R_test = R_test
        self.num_users, self.num_items = R_train.shape
        self.k = k
        self.alpha = alpha
        self.iterations = iterations
        self._lambda = _lambda

    def train(self):
        start_time = time.time()
        # Initialize user and item latent feature matrice
        self.U = np.random.normal(scale = 1./self.k, size = (self.num_users, self.k))
        self.V = np.random.normal(scale = 1./self.k, size = (self.num_items, self.k))
        
        if self.regularization() == True:
            # Initialize the biases
            self.b_u = np.zeros(self.num_users)
            self.b_i = np.zeros(self.num_items)
            self.b = np.mean(self.R_train[np.where(self.R_train != 0)])

        # Create a list of training samples
        self.samples = [
            (i, j, self.R_train[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R_train[i, j] > 0
        ]

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse_train, mse_test = self.mse()
            training_process.append((i, mse_train, mse_test))
            if (i + 1) % 10 == 0:
                print("Iteration: %d ; train error = %.4f; test error = %.4f" % (i + 1, mse_train, mse_test))
                
        print('training is complete! it took %.2f s' % (time.time() - start_time))
        return training_process
    
    def regularization(self):
        '''If we want to regularize the result'''
        return False if self._lambda is None else True

    def mse(self):
        """Compute the total mean square error for training and testing data
        """
        xTrain, yTrain = self.R_train.nonzero()
        xTest, yTest = self.R_test.nonzero()
        predicted = self.full_matrix()
        trainError, testError = 0, 0
        for x, y in zip(xTrain, yTrain):
            trainError += pow(self.R_train[x, y] - predicted[x, y], 2)
            
        for x, y in zip(xTest, yTest):
            testError += pow(self.R_test[x, y] - predicted[x, y], 2)
            
        return np.sqrt(trainError), np.sqrt(testError)

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)
            
            if self.regularization() == True:
                # Update biases
                self.b_u[i] += self.alpha * (e - self._lambda * self.b_u[i])
                self.b_i[j] += self.alpha * (e - self._lambda * self.b_i[j])

                # Update user and item latent feature matrices
                self.U[i, :] += self.alpha * (e * self.V[j, :] - self._lambda * self.U[i,:])
                self.V[j, :] += self.alpha * (e * self.U[i, :] - self._lambda * self.V[j,:])
            else:
                # Update user and item latent feature matrices
                self.U[i, :] += self.alpha * e * self.V[j, :]
                self.V[j, :] += self.alpha * e * self.U[i, :]

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        if self.regularization() == True:
            prediction = self.b + self.b_u[i] + self.b_i[j] + self.U[i, :].dot(self.V[j, :].T)
        else:
            prediction = self.U[i, :].dot(self.V[j, :].T)
        return prediction

    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        if self.regularization() == True:
            return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.U.dot(self.V.T)
        else:
            return self.U.dot(self.V.T)