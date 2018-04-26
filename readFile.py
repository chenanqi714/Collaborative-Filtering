import numpy as np
import pandas as pd

data = pd.read_csv('goodbooks-10k-master/ratings.csv', sep=',')
users = data.user_id.unique().shape[0]
items = data.book_id.unique().shape[0]
print ('Number of users = ' + str(users) + ' | Number of movies = ' + str(items))