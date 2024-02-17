import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from fcc_qp import FCCQPSolver, FCCQPSolution


SMALL_SIZE = 20
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def load_qp_matrices(test_name):
    data = np.load(f'test_data/id_qp_log_{test_name}.npz', allow_pickle=True)
    return data['qps']


def solve(qp, solver):
    solver.Solve(
        qp['Q'], qp['b'], qp['A_eq'], qp['b_eq'], qp['friction_coeffs'],
        qp['lb'], qp['ub'])
    result = solver.GetSolution()
    return result


@mpl.rc_context({'lines.linewidth': 4})
def plot_variable(name, val, idx, ylabel, ax):
    ax.plot(idx, val)
    ax.set_title(name)
    ax.set_xlabel('QP #')
    ax.set_ylabel(ylabel)


@mpl.rc_context({'lines.linewidth': 3})
def make_plots(results):
    idx = [i for i in range(len(results))]
    z = np.vstack([result.z for result in results])
    ts = [result.details.solve_time for result in results]
    n = [result.details.n_iter for result in results]

    # Dimensions of Cassie OSC problem
    vdot = z[:, :22]
    u = z[:, 22:32]
    lambda_h = z[:, 32:38]
    lambda_c = z[:, 38:]

    fig, ax = plt.subplots(2, 3, figsize=(20, 12))

    plot_variable('U Solution', u, idx, 'u (nm)', ax[0][0])
    plot_variable('Solve Time', ts, idx, 'Solve Time (seconds)', ax[0][1])
    plot_variable('Number of Iterations', n, idx, 'Iterations', ax[0][2])
    plot_variable('Vdot Solution', vdot, idx, 'Vdot Solution', ax[1][0])
    plot_variable('Lambda_h Solution', lambda_h, idx, 'Lambda_h (N or Nm)',
                  ax[1][1])
    plot_variable('Lambda_c Solution', lambda_c, idx, 'lambda_c (N)', ax[1][2])
    plt.show()


def main():
    qps = load_qp_matrices('running')

    # Dimensions of Cassie OSC problem
    solver = FCCQPSolver(50, 38, 12, 38)
    solver.set_rho(1000)
    solver.set_eps(1e-4)

    results = []
    for i in range(len(qps)):
        result = solve(qps[i], solver)
        results.append(result)

    make_plots(results)


if __name__ == '__main__':
    main()