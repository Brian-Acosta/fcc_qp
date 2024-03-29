import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from fcc_qp import FCCQP, FCCQPSolution, FCCQPOptions


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


@mpl.rc_context({'lines.linewidth': 2})
def plot_variable(name, val, idx, ylabel, ax):
    ax.plot(idx, val)
    ax.set_title(name)
    ax.set_xlabel('QP #')
    ax.set_ylabel(ylabel)


@mpl.rc_context({'lines.linewidth': 2})
def make_plots(results):
    idx = [i for i in range(len(results))]
    z = np.vstack([result.z for result in results])
    ts = [result.details.solve_time for result in results]
    n = [result.details.n_iter for result in results]
    fcone_viol = [result.details.friction_cone_viol for result in results]
    bound_viol = [result.details.bounds_viol for result in results]

    # Dimensions of Cassie OSC problem
    vdot = z[:, :22]
    u = z[:, 22:32]
    lambda_h = z[:, 32:38]
    lambda_c = z[:, 38:]

    fig, ax = plt.subplots(2, 4, figsize=(15, 9))

    plot_variable('U Solution', u, idx, 'u (nm)', ax[0][0])
    plot_variable('Solve Time', ts, idx, 'Solve Time (seconds)', ax[0][1])
    plot_variable('Number of Iterations', n, idx, 'Iterations', ax[0][2])
    plot_variable('Fcone Viol', fcone_viol, idx, 'Viol', ax[0][3])
    plot_variable('friction', vdot, idx, 'Vdot Solution', ax[1][0])
    plot_variable('Lambda_h Solution', lambda_h, idx, 'Lambda_h (N or Nm)',
                  ax[1][1])
    plot_variable('Lambda_c Solution', lambda_c, idx, 'lambda_c (N)', ax[1][2])
    plot_variable('Bounds Viol', bound_viol, idx, 'viol', ax[1][3])
    plt.show()


def main():
    dataset = 'walking'
    qps = load_qp_matrices(dataset)

    # Dimensions of Cassie OSC problem
    solver = FCCQP(60, 38, 12, 38)
    options = FCCQPOptions()
    options.rho = 5e-5
    options.eps_fcone = 1e-6
    options.eps_bound = 1e-6
    options.max_iter = 100
    solver.set_options(options)

    results = []
    for i in range(len(qps)):
        solver.set_warm_start(i > 0)
        result = solve(qps[i], solver)
        results.append(result)

    make_plots(results)


if __name__ == '__main__':
    main()