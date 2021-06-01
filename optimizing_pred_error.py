# You can observe a relatively large difference between
# the RMSE on the training data and the RMSE on the test
# data. This means that the model suffers from overfitting 
# on the training data: It does not generalize well. 
# you’ll see how to optimize the prediction error or test 
# error using various tuning methods.


# Plotting the Fit of Your Model
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

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
test_preds = knn_model.predict(X_test)

# ##################### predicted values ##################
cmap = sns.cubehelix_palette(as_cmap=True)
f, ax = plt.subplots()
points = ax.scatter(X_test[:,0], X_test[:,1], c=test_preds, s=50,cmap=cmap)
f.colorbar(points)
plt.show()



# you use Seaborn to create a scatter plot of the first
#  and second columns of X_test by subsetting the arrays 
# X_test[:,0] and X_test[:,1]. 
# Remember from before that the first two columns are 
# Length and Diameter. They are strongly correlated, 
# as you’ve seen in the correlations table.They are
# strongly correlated, as you’ve seen in 
# the correlations table.
# 
# You use c to specify that the predicted values
# (test_preds) shouldbe used as a colorbar. 
# The argument s is used to specify the size
# of the points in the scatter plot.
# You use cmap to specify the cubehelix_palette color map. 
# On this graph, each point is an abalone from the test set,
# with its actual length and actual diameter on the X- and Y-axis,
# respectively. The color of the point reflects the predicted age. 
# You can see that the longer and larger an abalone is,
# the higher its predicted age.. This is logical, and it’s a 
# positive sign. It means that your model is learning something 
# that seems correct.

# ##################### real values ##################
cmap = sns.cubehelix_palette(as_cmap=True)
f, ax = plt.subplots()
points = ax.scatter(X_test[:,0], X_test[:,1], c=y_test, s=50, cmap=cmap)
f.colorbar(points)
plt.show()
# This confirms that the trend your model is learning does indeed make sense.