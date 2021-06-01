import pandas as pd 


url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data")
abalone = pd.read_csv(url, header=None)
abalone.columns = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight",  "Shell weight", "Rings" ]
abalone = abalone.drop("Sex", axis=1)

correlation_matrix = abalone.corr()
# You can conclude that there’s at 
# least some correlation between physical 
# measurements of adult abalones and their 
# age, yet it’s also not very high.
print(correlation_matrix["Rings"])