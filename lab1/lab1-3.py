import numpy as np


def L1_norm(X):
    """
    Count ||X||_1
    """
    n = X.shape[0]
    if type(X[0]) == np.ndarray:
        l2_norm = abs(X[0][0])
        for i in range(n):
            for j in range(n):
                l2_norm = max(abs(X[i][j]), l2_norm)
    else:
        l2_norm = abs(X[0])
        for i in range(n):
            l2_norm = max(abs(X[i]), l2_norm)
    return l2_norm


def solve_iterative(A, b, eps):
    """
    Uses iterative method to solve Ax=b
    Returns x and number of iterations
    """
    n = A.shape[0]

    # Step 1. Ax=b -> x = alpha * x + beta
    alpha = np.zeros_like(A, dtype='float')
    beta = np.zeros_like(b, dtype='float')
    for i in range(n):
        for j in range(n):
            if i == j:
                alpha[i][j] = 0
            else:
                alpha[i][j] = -A[i][j] / A[i][i]

        beta[i] = b[i] / A[i][i]

    # Step 2. Iterating
    iterations = 0
    cur_x = np.copy(beta)
    converge = False
    while not converge:
        prev_x = np.copy(cur_x)
        cur_x = alpha @ prev_x + beta
        iterations += 1
        if L1_norm(alpha) < 1:
            converge = L1_norm(alpha) / (1 - L1_norm(alpha)) * L1_norm(cur_x - prev_x) <= eps
        else:
            converge = L1_norm(cur_x - prev_x) <= eps
    return cur_x, iterations


def seidel_multiplication(alpha, x, beta):
    """
    Count alhpa * x + beta for seidel method
    """
    res = np.copy(x)
    c = np.copy(alpha)
    for i in range(alpha.shape[0]):
        res[i] = beta[i]
        for j in range(alpha.shape[1]):
            res[i] += alpha[i][j] * res[j]
            if j < i:
                c[i][j] = 0
    return res, c


def solve_seidel(A, b, eps):
    """
    Uses Seidel method to solve Ax=b
    Returns x and number of iterations
    """
    n = A.shape[0]

    # Step 1. Ax=b -> x = alpha * x + beta
    alpha = np.zeros_like(A, dtype='float')
    beta = np.zeros_like(b, dtype='float')
    for i in range(n):
        for j in range(n):
            if i == j:
                alpha[i][j] = 0
            else:
                alpha[i][j] = -A[i][j] / A[i][i]

        beta[i] = b[i] / A[i][i]

    # Step 2. Iterating
    iterations = 0
    cur_x = np.copy(beta)
    converge = False
    while not converge:
        prev_x = np.copy(cur_x)
        cur_x, c = seidel_multiplication(alpha, prev_x, beta)
        iterations += 1
        if L1_norm(alpha) < 1:
            converge = L1_norm(c) / (1 - L1_norm(alpha)) * L1_norm(cur_x - prev_x) <= eps
        else:
            converge = L1_norm(prev_x - cur_x) <= eps
    return cur_x, iterations


if __name__ == '__main__':
    A = [
        [14, -4, -2, 3],
        [-3, 23, -6, -9],
        [-7, -8, 21, -5],
        [-2, -2, 8, 18]
    ]
    A = np.array(A, dtype='float')
    b = [38, -195, -27, 142]
    eps = 0.000000001

    print('Iteration method')
    x_iter, i_iter = solve_iterative(A, b, eps)
    print(x_iter)
    print('Iterations:', i_iter)
    print()

    print('Seidel method')
    x_seidel, i_seidel = solve_seidel(A, b, eps)
    print(x_seidel)
    print('Iterations:', i_seidel)
