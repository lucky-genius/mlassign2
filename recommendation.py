# usage：
# from recommendation import *
# R_train, R_test = make_R(filename = 'movie_ratings.csv', col_user = 'userId', col_item = 'movieId', col_rating = 'rating')
# mf = recommendation_system(R_train, R_test, k = 300, alpha = 0.01, _lambda = 0.01, iterations = 20)
# mf.matrix_factorization()
# or mf.correlation_similarity(2)
# mf.result
# mf.mse()

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

    def __init__(self, R_train, R_test):
        '''R_train: 2D numpy array; R matrix in with r_{i, j} is the rating that user i given to item j'''
        self.R_train = R_train
        self.R_test = R_test
        self.nUser, self.nItem = R_train.shape
        self.result = None

    def matrix_factorization(self, k, alpha, _lambda, iterations):
        '''perform matrix factorization to predict empty entries in a matrix.
        parameters:
            k: int; dimensions of u and v vector
            alpha: float; learning rate
            _lambda: float; regularization parameter'''
        self.k = k
        self.alpha = alpha
        self.iterations = iterations
        self._lambda = _lambda
        
        def sgd():
            '''stochastic graident descent'''
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
            sgd()
#             mse_train, mse_test = self.mse()
            if (i + 1) % 10 == 0:
                print("Iteration: %d " % (i + 1))
#                 print("Iteration: %d ; train error = %.4f; test error = %.4f" % (i + 1, mse_train, mse_test))
                
        if _lambda is not None:
            self.result = self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.U.dot(self.V.T)
        else:
            self.result = self.U.dot(self.V.T)
        
        mse_train, mse_test = self.mse()
        print('training is complete! it took %.2f s' % (time.time() - start_time))
        print("train error = %.4f; test error = %.4f" % (mse_train, mse_test))
        return self.result
    
    def mse(self):
        '''Compute the total mean square error for training and testing data'''
        xTrain, yTrain = self.R_train.nonzero()
        xTest, yTest = self.R_test.nonzero()
        trainError, testError = 0, 0
        for x, y in zip(xTrain, yTrain):
            trainError += pow(self.R_train[x, y] - self.result[x, y], 2)
            
        for x, y in zip(xTest, yTest):
            testError += pow(self.R_test[x, y] - self.result[x, y], 2)
            
        return np.sqrt(trainError), np.sqrt(testError)
        
    def correlation_similarity(self, k):
        start_time = time.time()
        self.result = self.R_train.copy()
#         R_normal = (R.T - np.nanmean(R, axis = 1)).T # normalizing with the average
        
        R_sparse = sparse.csr_matrix(self.R_train)
        S = cosine_similarity(R_sparse) # similarity 
        for row in np.arange(self.nUser): # for all users
            idxS = np.argsort(S[row])
#             for col in np.where(R[row] == 0)[0]: # for item(col) that doesn't have a rating
            for col in np.arange(self.nItem): # for all items
#                 print(row, col, R[row][col])
                idx_n = np.where(self.R_train[:, col] > 0)[0]# the idx of users who rated this item
#                 print(idx_n)
                if np.size(idx_n) < k:
                    self.result[row][col] = self.R_train[idx_n, col].mean()
                else:
                    idx_knn = idxS[np.isin(idxS, idx_n, assume_unique = True)][-k-1:-1] # idxS with ratings
#                     idx_knn = idx_n[np.argsort(S[row][idx_n])][-k:]# idx of k-nearest neighbors
#                 print(idx_knn)
                    self.result[row][col] = self.R_train[idx_knn, col].mean() # mean of the knn rating for this item
        print('training is complete! it took %.2f s' % (time.time() - start_time))
        return self.result