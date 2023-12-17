import pickle
import math
import numpy as np
import matplotlib.pyplot as plt

with open("solutions.pickle", 'rb') as file:
    solutions = pickle.load(file)


a = 1

x_begin = 0
x_end = math.pi

t_begin = 0
t_end = 5

h = 0.01
sigma = 0.45


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


plot_results(
    solutions=solutions,
    time=10,
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
