import numpy as np
arr = ["a", "aa", "aaa"]

def func(x):
    return len(x)

vec_func = np.vectorize(func)

result = vec_func(arr)
print(result)