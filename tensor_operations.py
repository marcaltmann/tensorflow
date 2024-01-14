import numpy as np
import time

x = np.random.random((20, 100))
y = np.random.random((20, 100))

def naive_relu(x):
    "Implements a naive rectified linear unit function for 2D tensors."
    assert len(x.shape) == 2
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)

    return x


def naive_add(x, y):
    "Implements a naive add function for 2D tensors."
    assert len(x.shape) == 2
    assert x.shape == y.shape
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]

    return x


t0 = time.time()
for _ in range(1000):
    z = x + y
    z = np.maximum(z, 0.)

print("Took: {0:.2f} s".format(time.time() - t0))


t0 = time.time()
for _ in range(1000):
    z = naive_add(x, y)
    z = naive_relu(x)

print("Took: {0:.2f} s".format(time.time() - t0))
