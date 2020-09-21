# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 16:08:50 2020

@author: kishe
"""

#impoting libraries
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

#%%
#import dataset
train       = pd.read_csv('sales_train.csv')
#test        = pd.read_csv('test.csv')
#submission  = pd.read_csv('sample_submission.csv')
items       = pd.read_csv('items.csv')
#item_cats   = pd.read_csv('item_categories.csv')
#shops       = pd.read_csv('shops.csv')
#%%
#X = train.iloc[:, :-1].values
#Y = train.iloc[:,-1].values

#Down casts the data entries from int64 to int32 and float64 to float32
#This reduces the size of the records by almost half. (From 134mb to 61mb)
def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df

#%%
#Calls the downcasting function
train = downcast_dtypes(train)
items = downcast_dtypes(items)

#%%
# Manual Feature engineering
# grouped for visual representation

# group data by month and shop_id, return sum of items sold per shop per month
month_group = pd.DataFrame(train.groupby(['date_block_num', 'shop_id'])['item_cnt_day'].sum().reset_index())

# added the item_category into sales_train
merged = pd.merge(train, items[['item_id', 'item_category_id']], on = 'item_id')

# group data by month and category_id, return sum of items sold per category per month
category_group = pd.DataFrame(merged.groupby(['date_block_num', 'item_category_id'])['item_cnt_day'].sum().reset_index())

#%%
# grouping for training model

# added the item_category into sales_train
merged2 = pd.merge(train, items[['item_id', 'item_category_id']], on = 'item_id')

# group data by month and shop_id, return sum of items sold per shop per month
# grouped by price as specials could result in higher sales
month_group2 = pd.DataFrame(merged2.groupby(['date_block_num', 'shop_id', 'item_category_id', 'item_id', 'item_price'])['item_cnt_day'].sum().reset_index())

#%%
month_group2['mon_lag_1'] = 0
month_group2['mon_lag_2'] = 0
month_group2['mon_lag_3'] = 0
month_group2['mon_lag_4'] = 0
month_group2['mon_lag_5'] = 0

#%%
# create leg data showing previous 5 months sale of that product at that store

for i in range(1085000, len(month_group2['shop_id'])):
  new_pd = month_group2.loc[month_group2['date_block_num'] == month_group2['date_block_num'][i]-1].loc[month_group2['shop_id'] == month_group2['shop_id'][i]].loc[month_group2['item_id'] == month_group2['item_id'][i]]
  new_pd2 = month_group2.loc[month_group2['date_block_num'] == month_group2['date_block_num'][i]-2].loc[month_group2['shop_id'] == month_group2['shop_id'][i]].loc[month_group2['item_id'] == month_group2['item_id'][i]]
  new_pd3 = month_group2.loc[month_group2['date_block_num'] == month_group2['date_block_num'][i]-3].loc[month_group2['shop_id'] == month_group2['shop_id'][i]].loc[month_group2['item_id'] == month_group2['item_id'][i]]
  new_pd4 = month_group2.loc[month_group2['date_block_num'] == month_group2['date_block_num'][i]-4].loc[month_group2['shop_id'] == month_group2['shop_id'][i]].loc[month_group2['item_id'] == month_group2['item_id'][i]]
  new_pd5 = month_group2.loc[month_group2['date_block_num'] == month_group2['date_block_num'][i]-5].loc[month_group2['shop_id'] == month_group2['shop_id'][i]].loc[month_group2['item_id'] == month_group2['item_id'][i]]
  #print(len(new_pd['date_block_num']))
  if len(new_pd['shop_id']) > 0:
    month_group2['mon_lag_1'][i] = month_group2['item_cnt_day'][new_pd.index[0]]
    print(i)

  if len(new_pd2['shop_id']) > 0:
    month_group2['mon_lag_2'][i] = month_group2['item_cnt_day'][new_pd2.index[0]]

  if len(new_pd3['shop_id']) > 0:
    month_group2['mon_lag_3'][i] = month_group2['item_cnt_day'][new_pd3.index[0]]

  if len(new_pd4['shop_id']) > 0:
    month_group2['mon_lag_4'][i] = month_group2['item_cnt_day'][new_pd4.index[0]]

  if len(new_pd5['shop_id']) > 0:
    month_group2['mon_lag_5'][i] = month_group2['item_cnt_day'][new_pd5.index[0]]
#%%
month_group2.to_csv('C:/Users/kishe/Desktop/AI/5214/project 1/CSCE5214-jupyter/Sales_Forecasting/month_group5.csv', encoding='utf-8', index=False)
#%%
# nominal intergers can not be converted to binary encoding, convert to string
month_group2['date_block_num'] = [('month ' + str(i)) for i in month_group2['date_block_num']]
month_group2['shop_id'] = [('shop ' + str(i)) for i in month_group2['shop_id']]
month_group2['item_category_id'] = [('item_category ' + str(i)) for i in month_group2['item_category_id']]
month_group2['item_id'] = [('item ' + str(i)) for i in month_group2['item_id']]
#%%
# break into X and Y, where X is the inputs and Y is our output
X = month_group2.iloc[:, :-1].values
Y = month_group2.iloc[:,-1].values

#%%
# Encoding categorical data
# used to provide a value to data that can be then used in equations, eg. Friday = 1 etc.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# the only variable which is not categorical is item_price, hence all other variable will be converted
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0, 1, 2, 3])], remainder = 'passthrough')
#X = np.array(ct.fit_transform(X), dtype=object)
X = ct.fit_transform(X)
# convert back to 2-D representation of the matrix from sparse matrix representation 
#X = X.todense()

#%%
# split data set into training set and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size = 0.2, random_state = 0)

#%%
# fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() 
regressor.fit(X_train, Y_train)

#%%
# Cross validation
from sklearn.model_selection import cross_val_score as cvs
reg_score = cvs(regressor, X_train, Y_train, scoring = "neg_mean_squared_error", cv = 10)
reg_score = np.sqrt(-reg_score)

#%%
def display_scores(scores):
    print("rmse Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
#%%
from sklearn.linear_model import Ridge
#ridge_reg = Ridge(alpha=1, solver="auto")
ridge_reg = Ridge()
#ridge_reg.fit(X_train, Y_train)

#%%
ridge_score = cvs(ridge_reg, X_train, Y_train, scoring = "neg_mean_squared_error", cv = 10)
ridge_score = np.sqrt(-ridge_score)
display_scores(reg_score)
display_scores(ridge_score)

#%%
# Grid search to find good hyperparameters
from sklearn.model_selection import GridSearchCV

# get parameters for hyperparameter tuning
#print(regressor.get_params().keys())
print(ridge_reg.get_params().keys())


#param_grid_reg = [{'fit_intercept': [True, False], 'normalize': [True, False]}]

param_grid_ridge = [{'alpha': [1e-3, 1e-2, 1e-1, 1]
                     , 'fit_intercept': [False]
                     , 'normalize': [True, False]
                     , 'solver': ['cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
                    , {'alpha': [1e-3, 1e-2, 1e-1, 1]
                     , 'fit_intercept': [True]
                     , 'normalize': [True, False]
                     , 'solver': ['sparse_cg', 'sag']}]

#grid_search_reg = GridSearchCV(regressor, param_grid_reg, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search_ridge = GridSearchCV(ridge_reg, param_grid_ridge, cv=5, scoring='neg_mean_squared_error', return_train_score=True)

#grid_search_reg.fit(X_test, Y_test)
grid_search_ridge.fit(X_test, Y_test)

#%%
cvres = grid_search_ridge.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
#%%
Y_pred = regressor.predict(X_test)

#%%
score = regressor.score(X_test, Y_test)
#%%
#z = ['month 0', 'shop 0', 'item_category 2', 'item 5572', 1322]


from scipy import stats
# def auto_predictor():
#     global ct
month = input('Please enter the month you want to predict. Enter the digit of the month, eg. Jan = 0 : ')
shop = input('Please enter the shop ID, eg. 2 : ')
item = input("Please enter the item ID you wish to predict, eg. for item 55, enter '55' : ")
item_cat = (items.loc[items['item_id'] == int(item), ['item_category_id']].values)[0][0]
prices = train.loc[train['item_id'] == int(item), ['item_price']].values
price = (stats.mode(prices))[0][0][0]

z = ['month ' + str(month), 'shop ' + str(shop), 'item_category ' + str(item_cat), 'item ' + str(item), price]
z = np.array(z, dtype = object).reshape(1, -1)
z = ct.transform(z)
z_pred = regressor.predict(z)
    
    # return(z_pred[0])
    

