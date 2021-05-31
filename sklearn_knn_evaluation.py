import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

abalone = pd.read_csv("abalone.data")
abalone.columns = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight",  "Shell weight", "Rings" ]
abalone = abalone.drop("Sex", axis=1)
X = abalone.drop("Rings", axis=1)
X = X.values
y = abalone["Rings"]
y = y.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)
knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(X_train, y_train)
train_preds = knn_model.predict(X_train)
mse = mean_squared_error(y_train, train_preds)
rmse = sqrt(mse)
print(rmse)
# For a more realistic result, you should evaluate the
# performances on data that aren’t included in the model.
# This is why you kept the test set separate for now. 
# You can evaluate the predictive performances on the
# test set with the same function as before:
test_preds = knn_model.predict(X_test)
mse = mean_squared_error(y_test, test_preds)
rmse = sqrt(mse)
print(rmse)
# This more-realistic RMSE is slightly higher than before. 