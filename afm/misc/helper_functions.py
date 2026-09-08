import numpy as np

def normalize(arr):
    arr = np.array(arr, dtype=float)
    value_range = arr.max() - arr.min()

    if value_range == 0:
        return np.zeros_like(arr)
    return (arr - arr.min()) / value_range