import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("solutions.pickle", 'rb') as file:
    solutions = pickle.load(file)


x_begin = 0
x_end = 1.05

y_begin = 0
y_end = 1.05

h_x = 0.05
h_y = 0.05


def max_abs_error(A, B):
    assert A.shape == B.shape
    return abs(A - B).max()


def mean_abs_error(A, B):
    assert A.shape == B.shape
    return abs(A - B).mean()


def plot_results(
    solutions, # dict: solutions[method name] = solution
    cur_y, # coord by y
    x_range, # (x_begin, x_end)
    y_range, # (y_begin, y_end)
    h_x, # len of cell by x
    h_y, # len of cell by y
):
    x = np.arange(*x_range, h_x)
    y = np.arange(*y_range, h_y)
    cur_y_id = abs(y - cur_y).argmin()

    plt.figure(figsize=(15, 9))
    for method_name, solution in solutions.items():
        plt.plot(x, solution[:, cur_y_id], label=method_name)

    plt.legend()
    plt.grid()
    plt.show()


def plot_errors_from_y(
    solutions, # dict: solutions[method name] = solution
    analytical_solution_name, # for comparing
    y_range, # (y_begin, y_end)
    h_y, # len of cell by y
):
    """
    Plot max_abs_error = f(y)
    """
    y = np.arange(*y_range, h_y)

    plt.figure(figsize=(15, 9))
    for method_name, solution in solutions.items():
        if method_name == analytical_solution_name:
            continue
        max_abs_errors = np.array([
            max_abs_error(solution[:, i], solutions[analytical_solution_name][:, i])
            for i in range(len(y))
        ])
        plt.plot(y, max_abs_errors, label=method_name)

    plt.xlabel('y')
    plt.ylabel('Max abs error')

    plt.legend()
    plt.grid()
    plt.show()


plot_results(
    solutions=solutions,
    cur_y=0.5,
    x_range=(x_begin, x_end),
    y_range=(y_begin, y_end),
    h_x=h_x,
    h_y=h_y,
)

plot_errors_from_y(
    solutions=solutions,
    analytical_solution_name="analytical solution",
    y_range=(y_begin, y_end),
    h_y=h_y,
)
