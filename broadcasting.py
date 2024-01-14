import numpy as np

x = np.random.random((32, 10))
y = np.random.random((10,))

def naive_add_matrix_and_vector(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[j]

    return x


z = np.expand_dims(y, axis=0)
Y = np.concatenate([z] * 32, axis=0)


a = np.random.random((64, 3, 32, 10))
b = np.random.random((32, 10))
c = np.maximum(a, b)
