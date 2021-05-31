# Mode for Classification
# In classification problems, the target variable
# is categorical. As discussed before, you can’t 
# take averages on categorical variables. For 
# example, what would be the average of three 
# predicted car brands?You can’t apply an average 
# on class predictions.
# Instead, in the case of classification, you take the mode.
# The mode is the value that occurs most often.
# This means that you count the classes of all the neighbors, 
# and you retain the most common class. The prediction is
# the value that occurs most often among the neighbors.
# If there are multiple modes, there are multiple possible 
# solutions. You could select a final winner randomly from 
# the winners. You could also make the final decision based 
# on the distances of the neighbors, in which case the mode 
# of the closest neighbors would be retained.
import scipy.stats
import numpy as np

class_neighbors = np.array(["A", "B", "B","C"])
mode_class = scipy.stats.mode(class_neighbors)
print(mode_class)
