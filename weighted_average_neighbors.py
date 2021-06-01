# Adding Weighted Average of Neighbors Based on Distance
# This means that neighbors that are further away will 
# less strongly influence the prediction.
# However, setting this weighted average could have an 
# impact on the optimal value of k. 

import pandas as pd 
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split    
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

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
    "weights": ["uniform", "distance"],
}
gridsearch = GridSearchCV(KNeighborsRegressor(), parameters)
gridsearch.fit(X_train, y_train)
gridsearch.best_params_
test_preds_grid = gridsearch.predict(X_test)
test_mse = mean_squared_error(y_test, test_preds_grid)
test_rmse = sqrt(test_mse)
print(test_rmse)