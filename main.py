import pandas as pd 
import matplotlib.pyplot as plt 

url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data" )


abalone = pd.read_csv(url, header=None)


abalone.columns = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight",  "Shell weight", "Rings" ]


# You should remove the Sex column.
#  The goal of the current exercise 
# is to use physical measurements to
#  predict the age of the abalone.

abalone = abalone.drop("Sex", axis=1)

# The target variable of this exercise is Rings
# A histogram will give you a quick and useful 
# overview of the age ranges that you can expect

abalone["Rings"].hist(bins=15)
plt.show()

# Too few bins can hide certain patterns, while
# too many bins can make the histogram lack 
# smoothness.
# The histogram shows that most abalones in the 
# dataset have between five and fifteen rings, 
# but that it’s possible to get up to twenty-five 
# rings
# print(abalone.head(10))
