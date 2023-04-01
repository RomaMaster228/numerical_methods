import copy


def LU_decompose(A):
    """
    A = LU, where:
    L - lower triangle matrix
    U - upper triangle matrix
    Returns L, U
    """
    n = len(A)
    L = [[0 for _ in range(n)] for _ in range(n)]
    U = copy.deepcopy(A)

    for k in range(1, n):
        for i in range(k - 1, n):
            for j in range(i, n):
                L[j][i] = U[j][i] / U[i][i]

        for i in range(k, n):
            for j in range(k - 1, n):
                U[i][j] = U[i][j] - L[i][k - 1] * U[k - 1][j]

    return L, U


def solve_system(L, U, b):
    """
    Solves system of equations: LUx=b
    Returns x
    """
    # Step 1: Ly = b
    n = len(L)
    y = [0 for _ in range(n)]
    for i in range(n):
        s = 0
        for j in range(i):
            s += L[i][j] * y[j]
        y[i] = (b[i] - s) / L[i][i]

    # Step 2: Ux = y
    x = [0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        s = 0
        for j in range(n - 1, i - 1, -1):
            s += U[i][j] * x[j]
        x[i] = (y[i] - s) / U[i][i]
    return x


def determinant(A):
    _, U = LU_decompose(A)
    det = 1
    for i in range(len(U)):
        det *= U[i][i]
    return det


def inverse_matrix(A):
    """
    Calculate A^(-1)
    """
    n = len(A)
    E = [[float(i == j) for i in range(n)] for j in range(n)]
    L, U = LU_decompose(A)
    A_inv = []
    for e in E:
        inv_row = solve_system(L, U, e)
        A_inv.append(inv_row)
    return transpose(A_inv)


def transpose(X):
    m = len(X)
    n = len(X[0])
    X_T = [[X[j][i] for j in range(n)] for i in range(m)]
    return X_T


def print_matrix(A):
    m = len(A)
    n = len(A[0])
    for i in range(m):
        for j in range(n):
            print(f'%6.2f' % A[i][j], end=' ')
        print()


if __name__ == '__main__':
    A = [
        [-1, -8, 0, 5],
        [6, -6, 2, 4],
        [9, -5, -6, 4],
        [-5, 0, -9, 1]
    ]

    b = [
        -60,
        -10,
        65,
        18
    ]

    print("LU decomposition")
    L, U = LU_decompose(A)
    print('L:')
    print_matrix(L)
    print('U:')
    print_matrix(U)

    print("System solution")
    x = solve_system(L, U, b)
    print('x:', x)

    print("det A =", determinant(A))

    print("A^(-1)")
    print_matrix(inverse_matrix(A))
