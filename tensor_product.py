import numpy as np

x = np.random.random((5,32))
y = np.random.random((32,5))
z = np.dot(x, y)


def naive_vector_dot(x, y):
    "Calculates the dot product of two vectors."
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]
    z = 0.

    for i in range(x.shape[0]):
        z += x[i] * y[i]

    return z


def naive_matrix_vector_dot(x, y):
    "Calculates the dot product of a matrix and a vector."
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    z = np.zeros(x.shape[0])

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i, j] * y[j]

    return z


def naive_matrix_vector_dot2(x, y):
    """
    Calculates the dot product of a matrix and a vector.
    Reuses another function.
    """
    z = np.zeros(x.shape[0])

    for i in range(x.shape[0]):
        z[i] = naive_vector_dot(x[i, :], y)

    return z


def naive_matrix_dot(x, y):
    "Calculates the dot product two matrices."
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]
    z = np.zeros((x.shape[0], y.shape[1]))

    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            row_x = x[i, :]
            column_y = y[:, j]
            z[i, j] = naive_vector_dot(row_x, column_y)

    return z
