def f(x):
    return x / (x**3 + 8)


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


def integrate_trapeze_method(f, l, r, h):
    """
    Calculate integral f(x)dx at interval [l; r] using trapeze method with step=h
    """
    result = 0
    cur_x = l
    while cur_x < r:
        result += h * 0.5 * (f(cur_x + h) + f(cur_x))
        cur_x += h
    return result


def integrate_simpson_method(f, l, r, h):
    """
    Calculate integral f(x)dx at interval [l; r] using simpson method with step=h
    """
    result = 0
    cur_x = l + h
    while cur_x < r:
        result += f(cur_x - h) + 4*f(cur_x) + f(cur_x + h)
        cur_x += 2 * h
    return result * h / 3


def runge_romberg_method(h1, h2, integral1, integral2, p):
    """
    Find more accurate value of integral using previous calculations.
    Works if h1 == k * h2
    """
    return integral1 + (integral1 - integral2) / ((h2 / h1)**p - 1)


if __name__ == '__main__':
    l, r = -1, 1  # interval of integrating
    h1, h2 = 0.5, 0.25  # steps

    print('Rectangle method')
    int_rectangle_h1 = integrate_rectangle_method(f, l, r, h1)
    int_rectangle_h2 = integrate_rectangle_method(f, l, r, h2)
    print(f'Step = {h1}: integral = {int_rectangle_h1}')
    print(f'Step = {h2}: integral = {int_rectangle_h2}')

    print('Trapeze method')
    int_trapeze_h1 = integrate_trapeze_method(f, l, r, h1)
    int_trapeze_h2 = integrate_trapeze_method(f, l, r, h2)
    print(f'Step = {h1}: integral = {int_trapeze_h1}')
    print(f'Step = {h2}: integral = {int_trapeze_h2}')

    print('Simpson method')
    int_simpson_h1 = integrate_simpson_method(f, l, r, h1)
    int_simpson_h2 = integrate_simpson_method(f, l, r, h2)
    print(f'Step = {h1}: integral = {int_simpson_h1}')
    print(f'Step = {h2}: integral = {int_simpson_h2}')

    print('Runge Romberg method')
    print(f'More accurate integral by rectangle method = {runge_romberg_method(h1, h2, int_rectangle_h1, int_rectangle_h2, 3)}')
    print(f'More accurate integral by trapeze method = {runge_romberg_method(h1, h2, int_trapeze_h1, int_trapeze_h2, 3)}')
    print(f'More accurate integral by Simpson method = {runge_romberg_method(h1, h2, int_simpson_h1, int_simpson_h2, 3)}')
