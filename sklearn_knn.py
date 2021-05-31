# Fit kNN in Python Using scikit-learn
#  When using a train-test split for model 
# evaluation, you split the dataset into two parts:
# Training data is used to fit the model. For kNN, 
# this means that the training data will be used as neighbors.
# Test data is used to evaluate the model. It means that you’ll
# make predictions for the number of rings of each of the abalones
# in the test data and compare those results to the known tru
# number of rings.
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import numpy as np 
import pandas as pd 

abalone = pd.read_csv("abalone.data")
abalone.columns = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight",  "Shell weight", "Rings" ]
abalone = abalone.drop("Sex", axis=1)
X = abalone.drop("Rings", axis=1)
X = X.values

y = abalone["Rings"] 
y = y.values 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)
# The test_size refers to the number of observations 
# that you want to put in the training data and the 
# test data. If you specify a test_size of 0.2, your
# test_size will be 20 percent of the original data, 
# therefore leaving the other 80 percent as training 
# data. The random_state is a parameter that allows 
# you to obtain the same results every time the code 
# is run. he choice of value in random_state is arbitrary.


# Fitting a kNN Regression in scikit-learn to the Abalone Dataset
knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(X_train, y_train)

# At this point, knn_model contains everything that’s 
# needed to make predictions on new abalone data points.
#  That’s all the code you need for fitting a kNN 
# regression using Python! 