import numpy as np 

a = np.array([2,2])
b = np.array([4,4])
c = np.linalg.norm(a-b)
# This way, you directly obtain the distance 
# between two multidimensional points.
print(c)