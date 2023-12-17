import math
import pickle
import numpy as np
import matplotlib.pyplot as plt

from lab1.lab1_2 import tridiagonal_solve


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


def phi_0(y, t, a=a, mu1=mu1, mu2=mu2):
    return math.cos(mu2*y) * math.exp(-(mu1**2 + mu2**2) * a * t)


def phi_1(y, t, a=a, mu1=mu1, mu2=mu2):
    return (-1)**mu1 * math.cos(mu2*y) * math.exp(-(mu1**2 + mu2**2) * a * t)


def phi_2(x, t, a=a, mu1=mu1, mu2=mu2):
    return math.cos(mu1*x) * math.exp(-(mu1**2 + mu2**2) * a * t)


def phi_3(x, t, a=a, mu1=mu1, mu2=mu2):
    return (-1)**mu2 * math.cos(mu1*x) * math.exp(-(mu1**2 + mu2**2) * a * t)


def psi(x, y, mu1=mu1, mu2=mu2):
    return math.cos(mu1*x) * math.cos(mu2*y)


def solution(x, y, t, a=a, mu1=mu1, mu2=mu2):
    return math.cos(mu1*x) * math.cos(mu2*y) * math.exp(-(mu1**2 + mu2**2) * a * t)


def get_analytical_solution(
        x_range,  # (x_begin, x_end)
        y_range,  # (y_begin, y_end)
        t_range,  # (t_begin, t_end)
        h_x,  # len of cell by x
        h_y,  # len of cell by y
        tau,  # len of cell by t
):
    """
    Get analytical solution of 2D parabolic DE
    Returns tensor U with values of function
    """
    x = np.arange(*x_range, h_x)
    y = np.arange(*y_range, h_y)
    t = np.arange(*t_range, tau)

    res = np.zeros((len(t), len(x), len(y)))
    for idx in range(len(x)):
        for idy in range(len(y)):
            for idt in range(len(t)):
                res[idt][idx][idy] = solution(x[idx], y[idy], t[idt])

    return res


analytical_solution = get_analytical_solution(
    x_range=(x_begin, x_end),
    y_range=(y_begin, y_end),
    t_range=(t_begin, t_end),
    h_x=h_x,
    h_y=h_y,
    tau=tau,
)

solutions = dict()
solutions["analytical solution"] = analytical_solution


def max_abs_error(A, B):
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


def variable_directions_method(
        x_range,  # (x_begin, x_end)
        y_range,  # (y_begin, y_end)
        t_range,  # (t_begin, t_end)
        h_x,  # len of cell by x
        h_y,  # len of cell by y
        tau,  # len of cell by t
        a=a,  # coefficient a
        mu1=mu1,  # coefficient mu1
        mu2=mu2,  # coefficient mu2
        phi_0=phi_0,  # boundary condition 0
        phi_1=phi_1,  # boundary condition 1
        phi_2=phi_2,  # boundary condition 2
        phi_3=phi_3,  # boundary condition 3
        psi=psi,  # initial condition
):
    """
    Solves 2D parabolic DE using variable directions.
    Returns tensor U with values of function
    """
    x = np.arange(*x_range, h_x)
    y = np.arange(*y_range, h_y)
    t = np.arange(*t_range, tau)
    res = np.zeros((len(t), len(x), len(y)))

    # initial condition
    for x_id in range(len(x)):
        for y_id in range(len(y)):
            res[0][x_id][y_id] = psi(x[x_id], y[y_id], mu1, mu2)

    for t_id in range(1, len(t)):
        U_halftime = np.zeros((len(x), len(y)))

        # boundary conditions
        for x_id in range(len(x)):
            res[t_id][x_id][0] = phi_2(x[x_id], t[t_id], a, mu1, mu2)
            res[t_id][x_id][-1] = phi_3(x[x_id], t[t_id], a, mu1, mu2)
            U_halftime[x_id][0] = phi_2(x[x_id], t[t_id] - tau / 2, a, mu1, mu2)
            U_halftime[x_id][-1] = phi_3(x[x_id], t[t_id] - tau / 2, a, mu1, mu2)

        for y_id in range(len(y)):
            res[t_id][0][y_id] = phi_0(y[y_id], t[t_id], a, mu1, mu2)
            res[t_id][-1][y_id] = phi_1(y[y_id], t[t_id], a, mu1, mu2)
            U_halftime[0][y_id] = phi_0(y[y_id], t[t_id] - tau / 2, a, mu1, mu2)
            U_halftime[-1][y_id] = phi_1(y[y_id], t[t_id] - tau / 2, a, mu1, mu2)

        # solving sytem 1
        for y_id in range(1, len(y) - 1):
            A = np.zeros((len(x) - 2, len(x) - 2))
            b = np.zeros((len(x) - 2))

            A[0][0] = 2 * h_x ** 2 * h_y ** 2 + 2 * a * tau * h_y ** 2
            A[0][1] = -a * tau * h_y ** 2
            for i in range(1, len(A) - 1):
                A[i][i - 1] = -a * tau * h_y ** 2
                A[i][i] = 2 * h_x ** 2 * h_y ** 2 + 2 * a * tau * h_y ** 2
                A[i][i + 1] = -a * tau * h_y ** 2
            A[-1][-2] = -a * tau * h_y ** 2
            A[-1][-1] = 2 * h_x ** 2 * h_y ** 2 + 2 * a * tau * h_y ** 2

            for x_id in range(1, len(x) - 1):
                b[x_id - 1] = (
                        res[t_id - 1][x_id][y_id - 1] * a * tau * h_x ** 2
                        + res[t_id - 1][x_id][y_id] * (2 * h_x ** 2 * h_y ** 2 - 2 * a * tau * h_x ** 2)
                        + res[t_id - 1][x_id][y_id + 1] * a * tau * h_x ** 2
                )
            b[0] -= (-a * tau * h_y ** 2) * phi_0(y[y_id], t[t_id] - tau / 2, a, mu1, mu2)
            b[-1] -= (-a * tau * h_y ** 2) * phi_1(y[y_id], t[t_id] - tau / 2, a, mu1, mu2)
            U_halftime[1:-1, y_id] = np.array(tridiagonal_solve(A, b))

        # solving system 2
        for x_id in range(1, len(x) - 1):
            A = np.zeros((len(y) - 2, len(y) - 2))
            b = np.zeros((len(y) - 2))

            A[0][0] = 2 * h_x ** 2 * h_y ** 2 + 2 * a * tau * h_x ** 2
            A[0][1] = -a * tau * h_x ** 2
            for i in range(1, len(A) - 1):
                A[i][i - 1] = -a * tau * h_x ** 2
                A[i][i] = 2 * h_x ** 2 * h_y ** 2 + 2 * a * tau * h_x ** 2
                A[i][i + 1] = -a * tau * h_x ** 2
            A[-1][-2] = -a * tau * h_x ** 2
            A[-1][-1] = 2 * h_x ** 2 * h_y ** 2 + 2 * a * tau * h_x ** 2

            for y_id in range(1, len(y) - 1):
                b[y_id - 1] = (
                        U_halftime[x_id - 1][y_id] * a * tau * h_y ** 2
                        + U_halftime[x_id][y_id] * (2 * h_x ** 2 * h_y ** 2 - 2 * a * tau * h_y ** 2)
                        + U_halftime[x_id + 1][y_id] * a * tau * h_y ** 2
                )
            b[0] -= (-a * tau * h_x ** 2) * phi_2(x[x_id], t[t_id], a, mu1, mu2)
            b[-1] -= (-a * tau * h_x ** 2) * phi_3(x[x_id], t[t_id], a, mu1, mu2)
            res[t_id][x_id][1:-1] = tridiagonal_solve(A, b)
    return res


variable_directions_solution = variable_directions_method(
    x_range=(x_begin, x_end),
    y_range=(y_begin, y_end),
    t_range=(t_begin, t_end),
    h_x=h_x,
    h_y=h_y,
    tau=tau,
)

solutions["variable directions method"] = variable_directions_solution

print(f'max abs error = {max_abs_error(variable_directions_solution, analytical_solution)}')
print(f'mean abs error = {mean_abs_error(variable_directions_solution, analytical_solution)}')


def fractional_steps_method(
        x_range,  # (x_begin, x_end)
        y_range,  # (y_begin, y_end)
        t_range,  # (t_begin, t_end)
        h_x,  # len of cell by x
        h_y,  # len of cell by y
        tau,  # len of cell by t
        a=a,  # coefficient a
        mu1=mu1,  # coefficient mu1
        mu2=mu2,  # coefficient mu2
        phi_0=phi_0,  # boundary condition 0
        phi_1=phi_1,  # boundary condition 1
        phi_2=phi_2,  # boundary condition 2
        phi_3=phi_3,  # boundary condition 3
        psi=psi,  # initial condition
):
    """
    Solves 2D parabolic DE using fractional steps method.
    Returns tensor U with values of function
    """
    x = np.arange(*x_range, h_x)
    y = np.arange(*y_range, h_y)
    t = np.arange(*t_range, tau)
    res = np.zeros((len(t), len(x), len(y)))

    # initial condition
    for x_id in range(len(x)):
        for y_id in range(len(y)):
            res[0][x_id][y_id] = psi(x[x_id], y[y_id], mu1, mu2)

    for t_id in range(1, len(t)):
        U_halftime = np.zeros((len(x), len(y)))

        # boundary conditions
        for x_id in range(len(x)):
            res[t_id][x_id][0] = phi_2(x[x_id], t[t_id], a, mu1, mu2)
            res[t_id][x_id][-1] = phi_3(x[x_id], t[t_id], a, mu1, mu2)
            U_halftime[x_id][0] = phi_2(x[x_id], t[t_id] - tau / 2, a, mu1, mu2)
            U_halftime[x_id][-1] = phi_3(x[x_id], t[t_id] - tau / 2, a, mu1, mu2)

        for y_id in range(len(y)):
            res[t_id][0][y_id] = phi_0(y[y_id], t[t_id], a, mu1, mu2)
            res[t_id][-1][y_id] = phi_1(y[y_id], t[t_id], a, mu1, mu2)
            U_halftime[0][y_id] = phi_0(y[y_id], t[t_id] - tau / 2, a, mu1, mu2)
            U_halftime[-1][y_id] = phi_1(y[y_id], t[t_id] - tau / 2, a, mu1, mu2)

        # solving sytem 1
        for y_id in range(1, len(y) - 1):
            A = np.zeros((len(x) - 2, len(x) - 2))
            b = np.zeros((len(x) - 2))

            A[0][0] = h_x ** 2 + 2 * a * tau
            A[0][1] = -a * tau
            for i in range(1, len(A) - 1):
                A[i][i - 1] = -a * tau
                A[i][i] = h_x ** 2 + 2 * a * tau
                A[i][i + 1] = -a * tau
            A[-1][-2] = -a * tau
            A[-1][-1] = h_x ** 2 + 2 * a * tau

            for x_id in range(1, len(x) - 1):
                b[x_id - 1] = res[t_id - 1][x_id][y_id] * h_x ** 2
            b[0] -= (-a * tau) * phi_0(y[y_id], t[t_id] - tau / 2, a, mu1, mu2)
            b[-1] -= (-a * tau) * phi_1(y[y_id], t[t_id] - tau / 2, a, mu1, mu2)
            U_halftime[1:-1, y_id] = np.array(tridiagonal_solve(A, b))

        # solving system 2
        for x_id in range(1, len(x) - 1):
            A = np.zeros((len(y) - 2, len(y) - 2))
            b = np.zeros((len(y) - 2))

            A[0][0] = h_y ** 2 + 2 * a * tau
            A[0][1] = -a * tau
            for i in range(1, len(A) - 1):
                A[i][i - 1] = -a * tau
                A[i][i] = h_y ** 2 + 2 * a * tau
                A[i][i + 1] = -a * tau
            A[-1][-2] = -a * tau
            A[-1][-1] = h_y ** 2 + 2 * a * tau

            for y_id in range(1, len(y) - 1):
                b[y_id - 1] = U_halftime[x_id][y_id] * h_y ** 2
            b[0] -= (-a * tau) * phi_2(x[x_id], t[t_id], a, mu1, mu2)
            b[-1] -= (-a * tau) * phi_3(x[x_id], t[t_id], a, mu1, mu2)
            res[t_id][x_id][1:-1] = tridiagonal_solve(A, b)
    return res


fractional_steps_solution = fractional_steps_method(
    x_range=(x_begin, x_end),
    y_range=(y_begin, y_end),
    t_range=(t_begin, t_end),
    h_x=h_x,
    h_y=h_y,
    tau=tau,
)

solutions["fractional steps method"] = fractional_steps_solution

print(f'max abs error = {max_abs_error(fractional_steps_solution, analytical_solution)}')
print(f'mean abs error = {mean_abs_error(fractional_steps_solution, analytical_solution)}')


f = open('solutions.pickle', 'wb')
pickle.dump(solutions, f)
f.close()


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