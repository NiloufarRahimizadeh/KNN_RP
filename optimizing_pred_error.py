# You can observe a relatively large difference between
# the RMSE on the training data and the RMSE on the test
# data. This means that the model suffers from overfitting 
# on the training data: It does not generalize well. 
# you’ll see how to optimize the prediction error or test 
# error using various tuning methods.


# Plotting the Fit of Your Model
import seaborn as sns


# , you use Seaborn to create a scatter plot of the first
#  and second columns of X_test by subsetting the arrays 
# X_test[:,0] and X_test[:,1]. 
# Remember from before that the first two columns are 
# Length and Diameter. They are strongly correlated, 
# as you’ve seen in the correlations table.