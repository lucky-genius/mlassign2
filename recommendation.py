# usage：
# from recommendation import *
# R = make_R(filename = 'movie_ratings.csv', col_user = 'userId', col_item = 'movieId', col_rating = 'rating')
# mf = recommendation_system(R, K = 300, alpha = 0.01, _lambda = 0.01, iterations = 20)
# mf.train()
# mf.full_matrix()
# mf.mse()
# i, j = 45, 500
# mf.get_rating(i, j)

import numpy as np
from astropy.io import ascii

def make_R(filename, col_user, col_item, col_rating):
    '''create R matrix from a file that contains a table of user IDs, item IDs and ratings
    parameters:
        filename: str; the filename of the table
        col_user: str; the column name of user IDs
        col_item: str; the column name of item IDs
        col_rating: str; the column name of the ratings
    returns:
        R: 2D numpy array; R matrix in with r_{i, j} is the rating that user i given to item j'''
    t = ascii.read(filename)
    idx_user = t.index_column(col_user)
    idx_item = t.index_column(col_item)
    idx_rating = t.index_column(col_rating)
    
    nUser = np.max(t[col_user]) # total number of users
    nitem = np.max(t[col_item]) # total number of items
    
    R = np.zeros(shape = (nUser, nitem))
    for nrow in np.arange(np.size(t)):
        i = t[nrow][idx_user] - 1 # the userId
        j = t[nrow][idx_item] - 1 # the itemId
        R[i][j] = t[nrow][idx_rating] # the rating
    
    return R

class recommendation_system():

    def __init__(self, R, K, alpha, _lambda, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        parameters:
            R: 2D numpy array; R matrix in with r_{i, j} is the rating that user i given to item j
            K (int)       : number of latent dimensions
            alpha (float) : learning rate
            _lambda (float)  : regularization parameter
        """

        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha

        self.iterations = iterations
        self._lambda = _lambda

    def train(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
        
        if self.regularization() == True:
            # Initialize the biases
            self.b_u = np.zeros(self.num_users)
            self.b_i = np.zeros(self.num_items)
            self.b = np.mean(self.R[np.where(self.R != 0)])

        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            if (i+1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i+1, mse))

        return training_process
    
    def regularization(self):
        '''If we want to regularize the result'''
        return False if self._lambda is None else True

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

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
                self.P[i, :] += self.alpha * (e * self.Q[j, :] - self._lambda * self.P[i,:])
                self.Q[j, :] += self.alpha * (e * self.P[i, :] - self._lambda * self.Q[j,:])
            else:
                # Update user and item latent feature matrices
                self.P[i, :] += self.alpha * e * self.Q[j, :]
                self.Q[j, :] += self.alpha * e * self.P[i, :]

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        if self.regularization() == True:
            prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        else:
            prediction = self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        if self.regularization() == True:
            return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)
        else:
            return self.P.dot(self.Q.T)