# Further Improving on kNN in scikit-learn With Bagging
# As a third step for kNN tuning, you can use bagging. 
# agging is an ensemble method, or a method that takes
# a relatively straightforward machine learning model 
# and fits a large number of those models with slight 
# variations in each fit. Bagging often uses decision 
# trees, but kNN works perfectly as well.
# One model can be wrong from time to time, but the 
# average of a hundred models should be wrong less often.
# The errors of different individual models are likely 
# to average each other out, and the resulting prediction 
# will be less variable.
import pandas as pd 
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor

abalone = pd.read_csv("abalone.data")
abalone.columns = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight",  "Shell weight", "Rings" ]
abalone = abalone.drop("Sex", axis=1)
X = abalone.drop("Rings", axis=1)
X = X.values
y = abalone["Rings"]
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)

parameters = {
    "n_neighbors": range(1, 50),
    "weights": ["uniform", "distance"]
}
gridsearch = GridSearchCV(KNeighborsRegressor(), parameters)
gridsearch.fit(X_train, y_train)
gridsearch.best_params_
best_k = gridsearch.best_params_["n_neighbors"]
best_weights = gridsearch.best_params_["weights"]
bagged_knn = KNeighborsRegressor(n_neighbors=best_k, weights=best_weights)
bagging_model = BaggingRegressor(bagged_knn, n_estimators=100)
bagging_model.fit(X_train,y_train)
test_preds_grid = bagging_model.predict(X_test)
test_mse = mean_squared_error(y_test, test_preds_grid)
test_rmse = sqrt(test_mse)
print(test_rmse)