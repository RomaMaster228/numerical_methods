import math

INF = 1e10


def integrate_rectangle_method(f, l, r, h):
    """
    Calculate integral f(x)dx at interval [l; r] using rectangle method with step=h
    """
    result = 0
    cur_x = l
    while cur_x < r:
        result += h * f((cur_x + cur_x + h) * 0.5)
        cur_x += h
    return result


def integrate_rectangle_method2(f, l, r, h):
    """
    Calculate integral f(x)dx at interval [l; r] using rectangle method with step=h
    """
    result = 0
    cur_x = l
    iters = 0
    while cur_x < r:
        result += h * f((cur_x + cur_x + h) * 0.5)
        cur_x += h
        iters += 1
    return result, iters


def f(x):
    """
    Function to integrate
    """
    if x < 0:
        return 0
    else:
        return x * (math.exp(- x / 2) / 4)


def integrate_with_definite_integral(f, l, r, h=0.01, eps=1e-6, max_iters=10000000):
    """
    Calculate improper integral (type 1) transforming to definite integrals
    """

    def f_new(t):
        return (1. / t ** 2) * f(1. / t)

    result = 0
    iters = 0
    if r == INF:
        new_r = max(eps, l)
        try:
            res, i = integrate_rectangle_method2(f_new, eps, 1. / new_r - eps, h)
        except ValueError:
            res = 0
        iters += i
        if iters >= max_iters:
            return "The integral is probably divergent, or slowly convergent"
        result += res
    else:
        new_r = r
    if l == -INF:
        new_l = min(-eps, r)
        try:
            res, i = integrate_rectangle_method2(f_new, 1. / new_l + eps, -eps, h)
        except ValueError:
            res = 0
        iters += i
        if iters >= max_iters:
            return "The integral is probably divergent, or slowly convergent"
        result += res
    else:
        new_l = l
    if new_l < new_r:
        try:
            res, i = integrate_rectangle_method2(f, new_l, new_r, h)
        except ValueError:
            res = 0
        iters += i
        if iters >= max_iters:
            return "The integral is probably divergent, or slowly convergent"
        result += res
    print(iters)
    return result


def integrate_lim(f, l, r, h=0.1, eps=1e-6, max_iters=10000000):
    """
    Calculate improper integral f(x)dx (type 1) using limit transition.
    Returns: integral result, number of iterations
    """
    result = 0
    iters = 0
    if r == INF:
        finish = False
        cur_x = max(l, 0)
        while not finish:
            iters += 1
            try:
                res = h * f((cur_x + cur_x + h) * 0.5)
            except ValueError:
                res = 0
            iters += 1
            if iters >= max_iters:
                return "The integral is probably divergent, or slowly convergent", None
            new_result = result + res
            cur_x += h
            if abs(new_result - result) < eps:
                finish = True
            result = new_result
    else:
        try:
            res = integrate_rectangle_method(f, 0, r, h)
        except ValueError:
            res = 0
        iters += 1
        if iters >= max_iters:
            return "The integral is probably divergent, or slowly convergent", None
        result += res
    if l == -INF:
        finish = False
        cur_x = min(0, r)
        while not finish:
            iters += 1
            try:
                res = h * f((cur_x - h + cur_x) * 0.5)
            except ValueError:
                res = 0
            iters += 1
            if iters >= max_iters:
                return "The integral is probably divergent, or slowly convergent", None
            new_result = result + res
            cur_x -= h
            if abs(new_result - result) < eps:
                finish = True
            result = new_result
    else:
        try:
            res = integrate_rectangle_method(f, l, 0, h)
        except ValueError:
            res = 0
        iters += 1
        if iters >= max_iters:
            return "The integral is probably divergent, or slowly convergent", None
        result += res
    return result, iters


if __name__ == '__main__':
    a = -INF
    b = INF
    h = 0.01
    eps = 0.0001
    print('Transforming to definite integral')

    res_definite = integrate_with_definite_integral(f, a, b, h, eps)
    print('Integral =', res_definite)
    print()

    print('Limit method')
    res_limit, iters_limit = integrate_lim(f, a, b)
    print('Integral =', res_limit)
    print('Iterations:', iters_limit)
    print()


"""
(1 / (2 * math.sqrt(2 * math.pi))) * math.exp(-x * x / 8)
Transforming to definite integral
Integral = 1.0019548111873857

Limit method
Integral = 0.9857144797824032
Iterations: 980

(1 / (3 * math.sqrt(2 * math.pi))) * math.exp(-x * x / 18)
Transforming to definite integral
Integral = 1.0013032096773957

Limit method
Integral = 0.9771939244916249
Iterations: 1366

(1 / (4 * math.sqrt(2 * math.pi))) * math.exp(-x * x / 32)
Transforming to definite integral
Integral = 1.0009774070859054

Limit method
Integral = 0.9682465472841191
Iterations: 1718

1 / (math.pi * (1 + x ** 2))
Transforming to definite integral
Integral = 1.003119415164576

Limit method
Integral = 0.8867112706583545
Iterations: 1112

(1 / (3 * math.sqrt(2 * math.pi))) * math.exp(-(x ** 2) / 72)
Transforming to definite integral
Integral = 2.001303433724044

Limit method
Integral = 1.9997047136103574
Iterations: 4554

(1 / (x * math.sqrt(2 * math.pi))) * math.exp(-(math.log(x) ** 2) / 2)
res_limit, iters_limit = integrate_lim(f, a, b)
Transforming to definite integral
Integral = 1.00000056057111

Limit method
Integral = 0.9997796531421432
Iterations: 411

k = 2, teta = 2
    if x < 0:
        return 0
    else:
        return x * (math.exp(- x / 2) / 4)
Transforming to definite integral
Integral = 1.000012217719231

Limit method
Integral = 1.0000833196941048
Iterations: 270


"""