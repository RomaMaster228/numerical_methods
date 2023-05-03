import matplotlib.pyplot as plt

from lab1.lab1_1 import LU_decompose, solve_system


def least_squares(x, y, n):
    """
    Count coefficient of polynom (degree = n) for least squares method for approximating tabular function y = f(x)
    Returns arrays of coeffs
    """
    assert len(x) == len(y)
    A = []
    b = []
    for k in range(n + 1):
        A.append([sum(map(lambda x: x ** (i + k), x)) for i in range(n + 1)])
        b.append(sum(map(lambda x: x[0] * x[1] ** k, zip(y, x))))
    L, U = LU_decompose(A)
    return solve_system(L, U, b)


def P(coefs, x):
    """
    Calculate the value of polynomial function at x
    """
    return sum([c * x**i for i, c in enumerate(coefs)])


def sum_squared_errors(x, y, ls_coefs):
    """
    Calculate sum of squared errors
    """
    y_ls = [P(ls_coefs, x_i) for x_i in x]
    return sum((y_i - y_ls_i)**2 for y_i, y_ls_i in zip(y, y_ls))


if __name__ == '__main__':
    x = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    y = [-1.8415, 0.0, 1.8415, 2.9093, 3.1411, 3.2432]
    plt.scatter(x, y, color='r')
    plt.plot(x, y, color='c', label='original')

    print('Least squares method, degree = 1')
    ls1 = least_squares(x, y, 1)
    print(f'P(x) = {ls1[0]} + {ls1[1]}x')
    plt.plot(x, [P(ls1, x_i) for x_i in x], color='b', label='degree = 1')
    print(f'Sum of squared errors = {sum_squared_errors(x, y, ls1)}')

    print('Least squares method, degree = 2')
    ls2 = least_squares(x, y, 2)
    print(f'P(x) = {ls2[0]} + {ls2[1]}x + {ls2[2]}x^2')
    plt.plot(x, [P(ls2, x_i) for x_i in x], color='g', label='degree = 2')
    print(f'Sum of squared errors = {sum_squared_errors(x, y, ls2)}')

    plt.legend()
    plt.show()
