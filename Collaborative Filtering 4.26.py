import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import model_selection as cv
from sklearn.metrics import mean_squared_error
from math import sqrt

######## read data 

data = pd.read_csv('D:ratings.csv', sep=',')

users_size = data.user_id.unique().shape[0]

books_size = data.book_id.unique().shape[0]

######### separate data

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
train_data_matrix=train_data_matrix[:8000,:8000]

test_data_matrix=test_data_matrix[:8000,:8000]

#calculate the pairwise distance for all users and books
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')

book_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

######### function of prediction (user_based CF)

def predict_user(user_item,similarity):
    
    change1 = np.zeros((len(train_data_matrix), len(train_data_matrix[0])))
    
    for i in range(len(train_data_matrix)):
        
        for j in range(len(train_data_matrix[0])):
            
            if train_data_matrix[i][j]!=0:
                
                change1[i][j]=1
        
    multi = user_similarity.dot(train_data_matrix)
    
    multi1 = user_similarity.dot(change1)
    
    for i in range(len(multi1)):
        
        for j in range(len(multi1[0])):
            
            if multi1[i][j]==0:
                
                multi1[i][j]=1
    
    result = multi/multi1
    
    return result

######### function of prediction (item_based CF)

def predict_item(user_item,similarity):
    
    change1 = np.zeros((len(train_data_matrix), len(train_data_matrix[0])))
    
    for i in range(len(train_data_matrix)):
        
        for j in range(len(train_data_matrix[0])):
            
            if train_data_matrix[i][j]!=0:
                
                change1[i][j]=1
        
    multi = train_data_matrix.dot(user_similarity)
    
    multi1 = change1.dot(user_similarity)
    
    for i in range(len(multi1)):
        
        for j in range(len(multi1[0])):
            
            if multi1[i][j]==0:
                
                multi1[i][j]=1
    
    result = multi/multi1
    
    return result
    

######### function for calculating the accuracy

def rmse(prediction, truth):
    
    prediction = prediction[truth.nonzero()].flatten()
    
    truth = truth[truth.nonzero()].flatten()
    
    return sqrt(mean_squared_error(prediction,truth))



######### testing

result_user=predict_user(train_data_matrix,user_similarity)
result_item=predict_item(train_data_matrix,book_similarity)
print ('User-based Root mean square error: ' + str(rmse(result_user, test_data_matrix)))
print ('Item-based Root mean square error: ' + str(rmse(result_item, test_data_matrix)))


   