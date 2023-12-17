import math
import numpy as np
import pickle
import matplotlib.pyplot as plt

from lab1.lab1_2 import tridiagonal_solve


a = 1

x_begin = 0
x_end = math.pi

t_begin = 0
t_end = 5

h = 0.01
sigma = 0.45


# граничные условия
def phi_0(t, a=a):
    return math.exp(-a * t)


def phi_1(t, a=a):
    return -math.exp(-a * t)


# начальные условия
def psi(x):
    return math.cos(x)


def solution(x, t, a=a):
    return math.exp(-a * t) * math.cos(x)


def get_analytical_solution(
        x_range,  # (x_begin, x_end)
        t_range,  # (t_begin, t_end)
        h=h,  # длина клетки по x
        sigma=sigma,  # коэффициент sigma
        a=a,  # коэффициент a
):
    """
    Аналитическое решение параболического ДУ
    Возвращает матрицу U
    """
    tau = sigma * h ** 2 / a  # длина клетки по t
    x = np.arange(*x_range, h)
    t = np.arange(*t_range, tau)

    res = np.zeros((len(t), len(x)))
    for idx in range(len(x)):
        for idt in range(len(t)):
            res[idt][idx] = solution(x[idx], t[idt], a)

    return res


analytical_solution = get_analytical_solution(
    x_range=(x_begin, x_end),
    t_range=(t_begin, t_end),
    h=h,
    sigma=sigma,
    a=a,
)


solutions = dict()
solutions["analytical solution"] = analytical_solution


def max_abs_error(A, B):
    assert A.shape == B.shape
    return abs(A - B).max()


def mean_abs_error(A, B):
    assert A.shape == B.shape
    return abs(A - B).mean()


def plot_results(
    solutions,  # dict: solutions[method_name] = solution
    time,  # момент времени
    x_range,  # (x_begin, x_end)
    t_range,  # (t_bein, t_end)
    h,  # длина клетки по x
    sigma,  # коэффициент sigma
):
    tau = sigma * h**2 / a  # длина клетки по t
    x = np.arange(*x_range, h)
    times = np.arange(*t_range, tau)
    cur_t_id = abs(times - time).argmin()

    plt.figure(figsize=(15, 9))
    for method_name, solution in solutions.items():
        plt.plot(x, solution[cur_t_id], label=method_name)

    plt.legend()
    plt.grid()
    plt.show()


def plot_errors_from_time(
    solutions,  # dict: solutions[method name] = solution
    analytical_solution_name,  # для сравнения
    t_range,  # (t_bein, t_end)
    h,
    sigma,
):
    """
    max_abs_error = f(time)
    """
    tau = sigma * h**2 / a
    t = np.arange(*t_range, tau)

    plt.figure(figsize=(15, 9))
    for method_name, solution in solutions.items():
        if method_name == analytical_solution_name:
            continue
        max_abs_errors = np.array([
            max_abs_error(solution[i], solutions[analytical_solution_name][i])
            for i in range(len(t))
        ])
        plt.plot(t, max_abs_errors, label=method_name)

    plt.xlabel('time')
    plt.ylabel('Max abs error')

    plt.legend()
    plt.grid()
    plt.show()


def explicit_finite_difference_method(
    x_range, # (x_begin, x_end)
    t_range, # (t_begin, t_end)
    h, # len of cell by x
    sigma, # coefficient sigma
    a=a, # coefficient a
    phi_0=phi_0, # boundary condition 0
    phi_1=phi_1, # boundary condition 1
    psi=psi, # initial condition,
):
    """
    Явная конечно-разностная схема для решения параболического ДУ
    Возвращает матрицу U
    """
    tau = sigma * h**2 / a # len of cell by t
    x = np.arange(*x_range, h)
    t = np.arange(*t_range, tau)

    res = np.zeros((len(t), len(x)))
    # row 0 -> use initial condition
    for col_id in range(len(x)):
        res[0][col_id] = psi(x[col_id])

    for row_id in range(1, len(t)):
        # col 0 -> use boundary condition 0
        res[row_id][0] = phi_0(t[row_id], a)
        # cols 1..n-1 -> use explicit schema
        for col_id in range(1, len(x)-1):
            res[row_id][col_id] = (
                sigma * res[row_id-1][col_id-1]
                + (1 - 2*sigma) * res[row_id-1][col_id]
                + sigma * res[row_id-1][col_id+1]
            )
        # col n -> use boundary condition 1
        res[row_id][-1] = phi_1(t[row_id], a)
    return res


explicit_solution = explicit_finite_difference_method(
    x_range=(x_begin, x_end),
    t_range=(t_begin, t_end),
    h=h,
    sigma=sigma,
)


solutions["explicit schema"] = explicit_solution

print(f'max abs error = {max_abs_error(explicit_solution, analytical_solution)}')
print(f'mean abs error = {mean_abs_error(explicit_solution, analytical_solution)}')


def implicit_finite_difference_method(
    x_range, # (x_begin, x_end)
    t_range, # (t_begin, t_end)
    h, # len of cell by x
    sigma, # coefficient sigma
    a=a, # coefficient a
    phi_0=phi_0, # boundary condition 0
    phi_1=phi_1, # boundary condition 1
    psi=psi, # initial condition,
):
    """
    Неявная конечно-разностная схема для решения параболического ДУ
    Возвращает матрицу U
    """
    tau = sigma * h**2 / a # len of cell by t
    x = np.arange(*x_range, h)
    t = np.arange(*t_range, tau)
    res = np.zeros((len(t), len(x)))

    # row 0 -> use initial condition
    for col_id in range(len(x)):
        res[0][col_id] = psi(x[col_id])

    for row_id in range(1, len(t)):
        A = np.zeros((len(x)-2, len(x)-2))  # first and last elements will be counted with boundary conditions

        # create system of equations for implicit schema
        A[0][0] = -(1 + 2*sigma)
        A[0][1] = sigma
        for i in range(1, len(A) - 1):
            A[i][i-1] = sigma
            A[i][i] = -(1 + 2*sigma)
            A[i][i+1] = sigma
        A[-1][-2] = sigma
        A[-1][-1] = -(1 + 2*sigma)

        # vector b is previous line except first and last elements
        b = -res[row_id-1][1:-1]
        # apply boundary conditions
        b[0] -= sigma * phi_0(t[row_id])
        b[-1] -= sigma * phi_1(t[row_id])

        res[row_id][0] = phi_0(t[row_id])
        res[row_id][-1] = phi_1(t[row_id])
        res[row_id][1:-1] = tridiagonal_solve(A, b)

    return res


implicit_solution = implicit_finite_difference_method(
    x_range=(x_begin, x_end),
    t_range=(t_begin, t_end),
    h=h,
    sigma=sigma,
)

solutions["implicit schema"] = implicit_solution

print(f'max abs error = {max_abs_error(implicit_solution, analytical_solution)}')
print(f'mean abs error = {mean_abs_error(implicit_solution, analytical_solution)}')


def crank_nicolson_method(
    x_range, # (x_begin, x_end)
    t_range, # (t_begin, t_end)
    h, # len of cell by x
    sigma, # coefficient sigma
    a=a, # coefficient a
    phi_0=phi_0, # boundary condition 0
    phi_1=phi_1, # boundary condition 1
    psi=psi, # initial condition,
    theta=0.5, # coefficient theta for combination
):
    """
    Solves parabolic DE using Crank-Nicolson schema.
    Returns matrix U with values of function
    """
    tau = sigma * h**2 / a # len of cell by t
    x = np.arange(*x_range, h)
    t = np.arange(*t_range, tau)
    res = np.zeros((len(t), len(x)))

    # row 0 -> use initial condition
    for col_id in range(len(x)):
        res[0][col_id] = psi(x[col_id])

    for row_id in range(1, len(t)):
        A = np.zeros((len(x)-2, len(x)-2)) # first and last elements will be counted with boundary conditions

        # create system of equations for implicit schema
        A[0][0] = -(1 + 2*sigma*theta)
        A[0][1] = sigma * theta
        for i in range(1, len(A) - 1):
            A[i][i-1] = sigma * theta
            A[i][i] = -(1 + 2*sigma*theta)
            A[i][i+1] = sigma * theta
        A[-1][-2] = sigma * theta
        A[-1][-1] = -(1 + 2*sigma*theta)

        # vector b is previous line except first and last elements
        b = np.array([-(
            res[row_id-1][i] +
            (1-theta) * sigma *
                (res[row_id-1][i-1] - 2*res[row_id-1][i] + res[row_id-1][i+1])
        ) for i in range(1, len(res[row_id-1])-1)])
        # apply boundary conditions
        b[0] -= sigma * theta * phi_0(t[row_id])
        b[-1] -= sigma * theta * phi_1(t[row_id])

        res[row_id][0] = phi_0(t[row_id])
        res[row_id][-1] = phi_1(t[row_id])
        res[row_id][1:-1] = tridiagonal_solve(A, b)

    return res


crank_nicolson_solution = crank_nicolson_method(
    x_range=(x_begin, x_end),
    t_range=(t_begin, t_end),
    h=h,
    sigma=sigma,
)

solutions["crank-nicolson schema"] = crank_nicolson_solution

print(f'max abs error = {max_abs_error(crank_nicolson_solution, analytical_solution)}')
print(f'mean abs error = {mean_abs_error(crank_nicolson_solution, analytical_solution)}')

f = open('solutions.pickle', 'wb')
pickle.dump(solutions, f)
f.close()


plot_results(
    solutions=solutions,
    time=0.5,
    x_range=(x_begin, x_end),
    t_range=(t_begin, t_end),
    h=h,
    sigma=sigma,
)

plot_errors_from_time(
    solutions=solutions,
    analytical_solution_name="analytical solution",
    t_range=(t_begin, t_end),
    h=h,
    sigma=sigma,
)
