def tridiagonal_solve(A, b):
    """
    Solves Ax=b, where A - tridiagonal matrix
    Returns x
    """
    n = len(A)
    # Step 1. Forward
    v = [0 for _ in range(n)]
    u = [0 for _ in range(n)]
    v[0] = A[0][1] / -A[0][0]
    u[0] = b[0] / A[0][0]
    for i in range(1, n-1):
        v[i] = A[i][i+1] / (-A[i][i] - A[i][i-1] * v[i-1])
        u[i] = (A[i][i-1] * u[i-1] - b[i]) / (-A[i][i] - A[i][i-1] * v[i-1])
    v[n-1] = 0
    u[n-1] = (A[n-1][n-2] * u[n-2] - b[n-1]) / (-A[n-1][n-1] - A[n-1][n-2] * v[n-2])

    # Step 2. Backward
    x = [0 for _ in range(n)]
    x[n-1] = u[n-1]
    for i in range(n-1, 0, -1):
        x[i-1] = v[i-1] * x[i] + u[i-1]
    return x


if __name__ == "__main__":
    # n = int(input('Enter the number of equations: '))
    # print('Enter not null elements of tridiagonal matrix')
    A = [
        [-11, 9, 0, 0, 0],
        [1, -8, 1, 0, 0],
        [0, -2, -11, 5, 0],
        [0, 0, 3, -14, 7],
        [0, 0, 0, 8, 10]
    ]

    b = [
        -114,
        81,
        -8,
        -38,
        144
    ]

    print('Solution')
    x = tridiagonal_solve(A, b)
    print(x)
