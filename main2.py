import pandas as pd 
import numpy as np 

# Now, to find the nearest neighbors in NumPy.
#  As you’ve seen, you need to define distances
#  on the vectors of the independent variables, 
# so you should first get your pandas DataFrame
#  into a NumPy array using the .values attribute 
abalone = pd.read_csv("abalone.data")
abalone.columns = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight",  "Shell weight", "Rings" ]
abalone = abalone.drop("Sex", axis=1)

# This code block generates two objects that now contain your data: X and y
#  X is the independent variables and y is the dependent variable of your model.
# Note that you use a capital letter for X but a lowercase letter for y. This is 
# often done in machine learning code because mathematical notation generally
#  uses a capital letter for matrices and a lowercase letter for vectors.
X = abalone.drop("Rings", axis=1)
X = X.values
y = abalone["Rings"]
y = y.values
# You can create the NumPy array for this data point as follows:
new_data_point = np.array([
     0.569552,
     0.446407,
     0.154437,
     1.016849,
     0.439051,
     0.222526,
     0.291208,
])
# The next step is to compute the distances between
#  this new data point and each of the data points in
#  the Abalone Dataset using the following code:
distances = np.linalg.norm(X-new_data_point, axis=1)
k = 3
nearest_neighbor_ids = distances.argsort()[:k] #indices
print(X[nearest_neighbor_ids]) #their valuse