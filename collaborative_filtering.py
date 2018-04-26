import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import model_selection as cv
from sklearn.metrics import mean_squared_error
from math import sqrt

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #use np.newaxis so that mean_user_rating has same format as ratings
        ratings_difference = (ratings - mean_user_rating[:, np.newaxis])
        prediction = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_difference) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'book':
        prediction = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return prediction


def rmse(prediction, truth):
    prediction = prediction[truth.nonzero()].flatten()
    truth = truth[truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction,truth))





data = pd.read_csv('goodbooks-10k-master/ratings.csv', sep=',')
users_size = data.user_id.unique().shape[0]
books_size = data.book_id.unique().shape[0]

#seperate training data and testing data 3:1
training_data, testing_data = cv.train_test_split(data, test_size=0.25)


#Create two user-item matrices, one for training and another for testing
train_data_matrix = np.zeros((users_size, books_size))
for row in training_data.itertuples():
    train_data_matrix[row[1]-1, row[2]-1] = row[3]

test_data_matrix = np.zeros((users_size, books_size))
for row in testing_data.itertuples():
    test_data_matrix[row[1]-1, row[2]-1] = row[3]

#take only the first 10000 users and 10000 books since data too large
train_data_matrix=train_data_matrix[:5000,:10000]
test_data_matrix=test_data_matrix[:5000,:10000]

#calculate the pairwise distance for all users and books
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
book_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

#predict the ratings
book_prediction = predict(train_data_matrix, book_similarity, type='book')
user_prediction = predict(train_data_matrix, user_similarity, type='user')

#calculate the errors
print ('User-based Root mean square error: ' + str(rmse(user_prediction, test_data_matrix)))
print ('Item-based Root mean square error: ' + str(rmse(book_prediction, test_data_matrix)))
book_prediction=book_prediction[test_data_matrix.nonzero()]
test_data_matrix = test_data_matrix[test_data_matrix.nonzero()]
print(np.amax(book_prediction))
print(test_data_matrix)
print(len(test_data_matrix))
