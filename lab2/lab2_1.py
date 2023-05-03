import math
import numpy as np


def iteration_method(f, phi, interval, eps):
    """
    Find root of f(x) == 0 at interval using iteration method
    Returns x and number of iterations
    """
    l, r = interval[0], interval[1]
    q = max(abs(dphi(x)) for x in np.linspace(l, r, num=100))
    print(f"q = {q}")
    x_prev = (l + r) * 0.5
    iters = 0
    while True:
        iters += 1
        x = phi(x_prev)
        if q * abs(x - x_prev) / (1 - q) < eps:
            break
        x_prev = x

    return x, iters


def newton_method(f, df, interval, eps):
    """
    Find root of f(x) == 0 at interval using newton method
    Returns x and number of iterations
    """
    l, r = interval[0], interval[1]
    x_prev = (l + r) * 0.5
    iters = 0
    while True:
        iters += 1
        x = x_prev - f(x_prev) / df(x_prev)
        if abs(x - x_prev) < eps:
            break
        x_prev = x

    return x, iters


def f(x):
    return (3 ** x) - 5 * x * x + 1


def phi(x):
    return ((3 ** x) + 1) / (5 * x)


def df(x):
    return (3 ** x) * math.log(3) - 10 * x


def dphi(x):
    return ((3 ** x) * math.log(3) - (3 ** x) - 1) / (5 * x * x)


if __name__ == "__main__":
    l, r = 0.8, 1
    eps = 0.000000000000001

    print('Iteration method')
    x_iter, i_iter = iteration_method(f, phi, (l, r), eps)
    print('x =', x_iter, '; f(x) =', f(x_iter))
    print('Iterations:', i_iter)

    l, r = 0.6, 4
    print('Newton method')
    x_newton, i_newton = newton_method(f, df, (l, r), eps)
    print('x =', x_newton, '; f(x) =', f(x_newton))
    print('Iterations:', i_newton)
