# Tune and Optimize kNN in Python Using scikit-learn
# Improving kNN Performances in scikit-learn Using GridSearchCV
# Until now, you’ve always worked with k=3 in the kNN algorithm,
# but the best value for k is something that you need to find 
# empirically for each dataset.
# When you use few neighbors, you have a prediction that will 
# be much more variable than when you use more neighbors

# If you use too many neighbors, the prediction
# of each point risks being very close. Let’s say that you use all 
# neighbors for a prediction. In that case, every prediction would be the same.
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from math import sqrt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
# To find the best value for k, you’re going to 
# use a tool called GridSearchCV.This is a tool 
# that is often used for tuning hyperparameters 
# of machine learning models. In your case, it 
# will help by automatically finding the best 
# value of k for your dataset.

abalone = pd.read_csv("abalone.data")
abalone.columns =  ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight",  "Shell weight", "Rings" ]
abalone = abalone.drop("Sex", axis=1)
X = abalone.drop("Rings", axis=1)
X = X.values
y = abalone["Rings"]
y = y.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)
parameters = {"n_neighbors": range(1, 50)}
gridsearch = GridSearchCV(KNeighborsRegressor(), parameters)
gridsearch.fit(X_train, y_train)

# In short, GridSearchCV repeatedly fits kNN 
# regressors on a part of the data and tests 
# the performances on the remaining part of 
# the data. Doing this repeatedly will yield 
# a reliable estimate of the predictive 
# performance of each of the values for k. 
# In this example, you test the values from 1 to 50.

gridsearch.best_params_
# In this code, you print the parameters that 
# have the lowest error score. 
train_preds_grids = gridsearch.predict(X_train)
train_mse = mean_squared_error(y_train, train_preds_grids)
train_rmse = sqrt(train_mse)
print(train_rmse)

test_preds_grid = gridsearch.predict(X_test)
test_mse = mean_squared_error(y_test, test_preds_grid)
test_rmse = sqrt(test_mse)
print(test_rmse)
# With this code, you fit the model on the training 
# data and evaluate the test data. You can see that
#  the training error is worse than before, but the
# test error is better than before. This means that
# your model fits less closely to the training data. 
# Using GridSearchCV to find a value for k has 
# reduced the problem of overfitting on the training data.