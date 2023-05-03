import numpy as np
import math
import matplotlib.pyplot as plt

from lab1.lab1_2 import tridiagonal_solve
from lab3.lab3_4 import df
from lab4.lab4_1 import runge_kutta_method, runge_romberg_method, mae


def f(x, y, z):
    return (x * z - y) / (x * (x - 1))


def g(x, y, z):
    return z


# Functions for finite difference method
# y'' + p_fd(x)y' + q_fd(x)y = f_fd(x)

def p_fd(x):
    return -1 / (x - 1)


def q_fd(x):
    return 1 / (x * (x - 1))


def f_fd(x):
    return 0


def exact_solution(x):
    return 2 + x + 2 * x * math.log(abs(x))


def get_n(n_prev, n, ans_prev, ans, b, delta, gamma, y1):
    x, y = ans_prev[0], ans_prev[1]
    y_der = df(b, x, y)
    phi_n_prev = delta * y[-1] + gamma * y_der - y1
    x, y = ans[0], ans[1]
    y_der = df(b, x, y)
    phi_n = delta * y[-1] + gamma * y_der - y1
    return n - (n - n_prev) / (phi_n - phi_n_prev) * phi_n


def check_finish(x, y, b, delta, gamma, y1, eps):
    y_der = df(b, x, y)
    return abs(delta * y[-1] + gamma * y_der - y1) > eps


def shooting_method(f, g, alpha, beta, delta, gamma, y0, y1, interval, h, eps):
    a, b = interval[0], interval[1]
    n_prev, n = 1.0, 0.8
    y_der = (y0 - alpha * n_prev) / beta
    x_prev, y_prev = runge_kutta_method(f, g, n_prev, y_der, (a, b), h)
    y_der = (y0 - alpha * n) / beta
    x, y = runge_kutta_method(f, g, n, y_der, (a, b), h)
    iterations = 0

    while check_finish(x, y, b, delta, gamma, y1, eps):
        n, n_prev = get_n(n_prev, n, (x_prev, y_prev), (x, y), b, delta, gamma, y1), n
        x_prev, y_prev = x, y
        y_der = (y0 - alpha * n) / beta
        x, y = runge_kutta_method(f, g, n, y_der, (a, b), h)
        iterations += 1

    return x, y, iterations


def finite_difference_method(p, q, f, y0, yn, alpha, beta, delta, gamma, interval, h):
    A = []
    B = []
    rows = []
    a, b = interval
    x = np.arange(a, b + h, h)
    n = len(x)

    # Creating tridiagonal matrix
    for i in range(n):
        if i == 0:
            rows.append(alpha * h - beta)
        elif i == 1:
            rows.append(beta)
        else:
            rows.append(0)
    A.append(rows)
    B.append(y0 * h)

    for i in range(1, n - 1):
        rows = []
        B.append(f(x[i]) * h ** 2)
        for j in range(n):
            if j == i - 1:
                rows.append(1 - p(x[i]) * h / 2)
            elif j == i:
                rows.append(q(x[i]) * h ** 2 - 2)
            elif j == i + 1:
                rows.append(1 + p(x[i]) * h / 2)
            else:
                rows.append(0)
        A.append(rows)

    rows = []
    B.append(yn * h)
    for i in range(n):
        if i == n - 1:
            rows.append(delta * h + gamma)
        elif i == n - 2:
            rows.append(-gamma)
        else:
            rows.append(0)

    A.append(rows)
    y = tridiagonal_solve(A, B)
    return x, y


def print_task():
    task = """
Given problem:
x * (x - 1) * y'' - x * y' + y  = 0
y'(1) = 3
y(3) - 3 * y'(3) = -4

We will convert this to system of equations with order = 1:
y' = g(x, y, z) = z
z' = f(x, y, z) = (x * y' - y) / (x * (x - 1))
z(1) = 3
y(3) - 3 * z(3) = -4
"""
    print(task)


if __name__ == '__main__':
    interval = (1.000001, 3)  # x in [1; 3]
    y0 = 3
    y1 = -4
    h = 0.1
    eps = 0.001
    alpha, beta, delta, gamma = 0, 1, 1, -3

    print_task()

    x_shooting, y_shooting, iters_shooting = shooting_method(f, g, alpha, beta, delta, gamma, y0, y1, interval, h, eps)
    plt.plot(x_shooting, y_shooting, label=f'shooting method, step={h}')
    x_shooting2, y_shooting2, iters_shooting2 = shooting_method(f, g, alpha, beta, delta, gamma, y0, y1, interval, h / 2, eps)
    plt.plot(x_shooting2, y_shooting2, label=f'shooting method, step={h / 2}')

    x_fd, y_fd = finite_difference_method(p_fd, q_fd, f_fd, y0, y1, alpha, beta, delta, gamma, interval, h)
    plt.plot(x_fd, y_fd, label=f'finite difference method, step={h}')
    x_fd2, y_fd2 = finite_difference_method(p_fd, q_fd, f_fd, y0, y1, alpha, beta, delta, gamma, interval, h / 2)
    plt.plot(x_fd2, y_fd2, label=f'finite difference method, step={h / 2}')

    x_exact = [i for i in np.arange(interval[0], interval[1] + h, h)]
    x_exact2 = [i for i in np.arange(interval[0], interval[1] + h / 2, h / 2)]
    y_exact = [exact_solution(x_i) for x_i in x_exact]
    y_exact2 = [exact_solution(x_i) for x_i in x_exact2]
    plt.plot(x_exact, y_exact, label='exact solution')

    print('Iterations')
    print(f'Step = {h}')
    print('shooting:', iters_shooting)
    print(f'Step = {h / 2}')
    print('shooting:', iters_shooting2)
    print()

    print('Mean absolute errors')
    print(f'Step = {h}')
    print('finite difference:', mae(y_fd, y_exact))
    print(f'Step = {h / 2}')
    print('finite difference:', mae(y_fd2, y_exact2))
    print(f'Step = {h}')
    print('shooting:', mae(y_shooting, y_exact))
    print(f'Step = {h / 2}')
    print('shooting:', mae(y_shooting2, y_exact2))
    print()

    print('Runge-Romberg accuracy')
    print('shooting:', runge_romberg_method(h, h / 2, y_shooting, y_shooting2, 1))
    print('finite difference:', runge_romberg_method(h, h / 2, y_fd, y_fd2, 4))

    plt.legend()
    plt.show()
