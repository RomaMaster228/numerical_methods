import math
import pickle
import numpy as np
import matplotlib.pyplot as plt


with open("solutions.pickle", 'rb') as file:
    solutions = pickle.load(file)


x_begin = 0
x_end = math.pi

y_begin = 0
y_end = math.pi

t_begin = 0
t_end = 1

a = 1
mu1 = 1
mu2 = 1

h_x = 0.01
h_y = 0.01
tau = 0.01


def max_abs_error(A, B):
    assert A.shape == B.shape
    return abs(A - B).max()


def mean_abs_error(A, B):
    assert A.shape == B.shape
    return abs(A - B).mean()


def plot_results(
    solutions, # dict: solutions[method name] = solution
    cur_time, # moment of time
    cur_y, # moment by y
    x_range, # (x_begin, x_end)
    y_range, # (y_begin, y_end)
    t_range, # (t_bein, t_end)
    h_x, # len of cell by x
    h_y, # len of cell by y
    tau, # len of cell by t
):
    x = np.arange(*x_range, h_x)
    y = np.arange(*y_range, h_y)
    t = np.arange(*t_range, tau)
    cur_t_id = abs(t - cur_time).argmin()
    cur_y_id = abs(y - cur_y).argmin()

    plt.figure(figsize=(15, 9))
    for method_name, solution in solutions.items():
        plt.plot(x, solution[cur_t_id][:, cur_y_id], label=method_name)

    plt.legend()
    plt.grid()
    plt.show()


def plot_errors_from_time(
    solutions, # dict: solutions[method name] = solution
    analytical_solution_name, # for comparing
    t_range, # (t_begin, t_end)
    tau, # len of cell by t
):
    """
    Plot max_abs_error = f(time)
    """
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
    cur_time=0.5,
    cur_y=0.5,
    x_range=(x_begin, x_end),
    y_range=(y_begin, y_end),
    t_range=(t_begin, t_end),
    h_x=h_x,
    h_y=h_y,
    tau=tau
)

plot_errors_from_time(
    solutions=solutions,
    analytical_solution_name="analytical solution",
    t_range=(t_begin, t_end),
    tau=tau,
)