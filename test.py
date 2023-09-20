import numpy as np

vec1 = np.array([1, 2, 3])
vec2 = np.array([4, 5, 6])
print(np.dot(vec1.reshape((3, 1)), vec2.reshape(1, 3)))