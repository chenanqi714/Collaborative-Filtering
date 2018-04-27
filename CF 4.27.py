# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 11:26:27 2018

@author: Yisu.Tian, Anqi.Chen, Xinhe.Chen
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import model_selection as cv
from sklearn.metrics import mean_squared_error
from math import sqrt

######## read data 

data = pd.read_csv('H:ratings.csv', sep=',')

def train_test(data,train_test_size):
    
    users_size = data.user_id.unique().shape[0]

    books_size = data.book_id.unique().shape[0]
    
    #seperate training data and testing data 
    training_data, testing_data = cv.train_test_split(data, test_size=train_test_size)

    #Create two user-item matrices, one for training and another for testing
    train_data_matrix = np.zeros((users_size, books_size))

    for row in training_data.itertuples():
    
        train_data_matrix[row[1]-1, row[2]-1] = row[3]

    test_data_matrix = np.zeros((users_size, books_size))

    for row in testing_data.itertuples():
    
        test_data_matrix[row[1]-1, row[2]-1] = row[3]

    #take only the first 10000 users and 10000 books since data too large
    train_data_matrix=train_data_matrix[:10000,:10000]

    test_data_matrix=test_data_matrix[:10000,:10000]
    
    return (train_data_matrix, test_data_matrix)


######### function of prediction (user_based CF)

def predict_user(user_item,similarity):
    
    change1 = np.zeros((len(user_item), len(user_item[0])))
    
    for i in range(len(user_item)):
        
        for j in range(len(user_item[0])):
            
            if user_item[i][j]!=0:
                
                change1[i][j]=1
        
    multi = similarity.dot(user_item)
    
    multi1 = similarity.dot(change1)
    
    for i in range(len(multi1)):
        
        for j in range(len(multi1[0])):
            
            if multi1[i][j]==0:
                
                multi1[i][j]=1
    
    result = multi/multi1
    
    return result

######### function of prediction (item_based CF)

def predict_item(user_item,similarity):
    
    change1 = np.zeros((len(user_item), len(user_item[0])))
    
    for i in range(len(user_item)):
        
        for j in range(len(user_item[0])):
            
            if user_item[i][j]!=0:
                
                change1[i][j]=1
        
    multi = user_item.dot(similarity)
    
    multi1 = change1.dot(similarity)
    
    for i in range(len(multi1)):
        
        for j in range(len(multi1[0])):
            
            if multi1[i][j]==0:
                
                 multi1[i][j]=1
    
    result = multi/multi1
    
    return result
    

## function for calculating the accuracy

def rmse(prediction, truth):
    
    prediction = prediction[truth.nonzero()].flatten()
    
    truth = truth[truth.nonzero()].flatten()
    
    return sqrt(mean_squared_error(prediction,truth))
 

## function to create and test Collaborative Filtering
def CF(data, train_test_size, similarity_function):
    
    T = train_test(data, train_test_size)
    
    train_data_matrix = T[0]
    
    test_data_matrix = T[1]
    
    # similarity function: Cosine similarity
    
    if similarity_function == 'Cosine':
        
        user_similarity = pairwise_distances(train_data_matrix, metric='cosine')

        book_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')
     
    # similarity function: Pearson similarity    
        
    elif similarity_function == 'Pearson':
        
        user_similarity = np.corrcoef(train_data_matrix)*0.5+0.5  
        
        book_similarity = np.corrcoef(train_data_matrix.T)*0.5+0.5  
        
        user_similarity[np.isnan(user_similarity)] = 0
        
        book_similarity[np.isnan(book_similarity)] = 0
        
    # similarity function: Jaccard similarity       
        
    elif similarity_function == 'Jaccard':
        
        user_similarity = pairwise_distances(train_data_matrix, metric='jaccard')

        book_similarity = pairwise_distances(train_data_matrix.T, metric='jaccard')
        
        user_similarity[np.isnan(user_similarity)] = 0
        
        book_similarity[np.isnan(book_similarity)] = 0
        
    result_user = predict_user(train_data_matrix, user_similarity)
    
    result_item = predict_item(train_data_matrix, book_similarity)
    
    print ('User-based Root mean square error: ' + str(rmse(result_user, test_data_matrix)))
    
    print ('Item-based Root mean square error: ' + str(rmse(result_item, test_data_matrix)))


