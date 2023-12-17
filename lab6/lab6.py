import math
import numpy as np
import matplotlib.pyplot as plt

from lab1.lab1_2 import tridiagonal_solve

x_begin = 0
x_end = math.pi

t_begin = 0
t_end = 5

h = 0.01
sigma = 1


# граничные условия
def phi_0(t):
    return math.sin(2 * t)


def phi_1(t):
    return -math.sin(2 * t)


# начальные условия
def psi_0(x):
    return 0


def psi_1(x):
    return 2 * math.cos(x)


def solution(x, t):
    return math.cos(x) * math.sin(2 * t)


def get_analytical_solution(
        x_range,
        t_range,
        h,
        sigma=sigma,
):
    tau = np.sqrt(sigma * h ** 2)
    x = np.arange(*x_range, h)
    t = np.arange(*t_range, tau)

    res = np.zeros((len(t), len(x)))
    for idx in range(len(x)):
        for idt in range(len(t)):
            res[idt][idx] = solution(x[idx], t[idt])

    return res


analytical_solution = get_analytical_solution(
    x_range=(x_begin, x_end),
    t_range=(t_begin, t_end),
    h=h,
    sigma=sigma,
)


solutions = dict()
solutions["analytical solution"] = analytical_solution


def max_abs_error(A, B):
    """
    Calculate max absolute error of elements of matrices A and B
    """
    assert A.shape == B.shape
    return abs(A - B).max()


def mean_abs_error(A, B):
    """
    Calculate mean absolute error of elements of matrices A and B
    """
    assert A.shape == B.shape
    return abs(A - B).mean()


def plot_results(
    solutions, # dict: solutions[method name] = solution
    time, # moment of time
    x_range, # (x_begin, x_end)
    t_range, # (t_begin, t_end)
    h, # len of cell by x
    sigma, # coefficient sigma
):
    tau = np.sqrt(sigma * h**2) # len of cell by t
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
    solutions, # dict: solutions[method name] = solution
    analytical_solution_name, # for comparing
    t_range, # (t_begin, t_end)
    h, # len of cell by x
    sigma, # coefficient sigma
):
    """
    Plot max_abs_error = f(time)
    """
    tau = np.sqrt(sigma * h**2) # len of cell by t
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
        x_range,  # (x_begin, x_end)
        t_range,  # (t_begin, t_end)
        h,  # len of cell by x
        sigma,  # coefficient sigma
        phi_0=phi_0,  # boundary condition 0
        phi_1=phi_1,  # boundary condition 1
        psi_0=psi_0,  # initial condition 0,
        psi_1=psi_1,  # initial condition 1 (first derivative),
):
    """
    Явная конечно-разностная схема для решения гиперболического ДУ
    Возвращает матрицу U
    """
    tau = np.sqrt(sigma * h ** 2)  # len of cell by t
    x = np.arange(*x_range, h)
    t = np.arange(*t_range, tau)

    res = np.zeros((len(t), len(x)))
    for col_id in range(len(x)):
        res[0][col_id] = psi_0(x[col_id])

    for col_id in range(len(x)):
        res[1][col_id] = psi_0(x[col_id]) + tau * psi_1(x[col_id])

    for row_id in range(2, len(t)):
        res[row_id][0] = phi_0(t[row_id])
        for col_id in range(1, len(x) - 1):
            res[row_id][col_id] = (
                    sigma * (
                    res[row_id - 1][col_id + 1]
                    - 2 * res[row_id - 1][col_id]
                    + res[row_id - 1][col_id - 1]
            )
                    + (2 - 3 * tau ** 2) * res[row_id - 1][col_id]
                    - res[row_id - 2][col_id]
            )
        res[row_id][-1] = phi_1(t[row_id])
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

# Явная схема устойчива при условии: sigma = (tau ^ 2) / (h ^ 2) < 1


def implicit_finite_difference_method(
        x_range,  # (x_begin, x_end)
        t_range,  # (t_begin, t_end)
        h,  # len of cell by x
        sigma,  # coefficient sigma
        phi_0=phi_0,  # boundary condition 0
        phi_1=phi_1,  # boundary condition 1
        psi_0=psi_0,  # initial condition 0,
        psi_1=psi_1,  # initial condition 1 (first derivative),
):
    """
    Неявная конечно-разностная схема для решения параболического ДУ
    Возвращает матрицу U
    """
    tau = np.sqrt(sigma * h ** 2)  # len of cell by t
    x = np.arange(*x_range, h)
    t = np.arange(*t_range, tau)
    res = np.zeros((len(t), len(x)))

    for col_id in range(len(x)):
        res[0][col_id] = psi_0(x[col_id])

    for col_id in range(len(x)):
        res[1][col_id] = psi_0(x[col_id]) + tau * psi_1(x[col_id])

    for row_id in range(2, len(t)):
        A = np.zeros((len(x) - 2, len(x) - 2))  # first and last elements will be counted with boundary conditions

        A[0][0] = -(1 + 2 * sigma + 3 * tau ** 2)
        A[0][1] = sigma
        for i in range(1, len(A) - 1):
            A[i][i - 1] = sigma
            A[i][i] = -(1 + 2 * sigma + 3 * tau ** 2)
            A[i][i + 1] = sigma
        A[-1][-2] = sigma
        A[-1][-1] = -(1 + 2 * sigma + 3 * tau ** 2)

        b = -2 * res[row_id - 1][1:-1] + res[row_id - 2][1:-1]
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