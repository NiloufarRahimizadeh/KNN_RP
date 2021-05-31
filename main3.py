#Voting or Averaging of Multiple Neighbors
import numpy as np 
import pandas as pd 


abalone = pd.read_csv("abalone.data")
abalone.columns = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight",  "Shell weight", "Rings" ]
abalone = abalone.drop("Sex", axis=1)
X = abalone.drop("Rings", axis=1)
X = X.values
y = abalone["Rings"]
y = y.values
new_data_point = np.array([
     0.569552,
     0.446407,
     0.154437,
     1.016849,
     0.439051,
     0.222526,
     0.291208,
])
distances= np.linalg.norm(X-new_data_point, axis=1)
k=3
nearest_neighbors_ids = distances.argsort()[:k]
# print(X[nearest_neighbors_ids])
nearest_neighbor_rings = y[nearest_neighbors_ids]
print(nearest_neighbor_rings)
# Now that you have the values for those three neighbors,
# you’ll combine them into a prediction for your new data
# point.
# Average for Regression
# You combine multiple neighbors into one prediction by 
# taking the average of their values of the target variable. 
# You can do this as follows:

prediction = nearest_neighbor_rings.mean()
# You’ll get a value of 10 for prediction. 
# This means that the 3-Nearest Neighbor 
# prediction for your new data point is 10. 