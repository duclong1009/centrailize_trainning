import numpy as np

def softmax(x, axis=None):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# Create a sample 3D NumPy array
array_3d = np.array([[[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0],
                     [7.0, 8.0, 9.0]],
                    [[9.0, 8.0, 7.0],
                     [6.0, 5.0, 4.0],
                     [3.0, 2.0, 1.0]]])

# Apply the softmax function along a specified axis (e.g., axis=2 for the third dimension)
result = softmax(array_3d, axis=2)
breakpoint()
print(result)
