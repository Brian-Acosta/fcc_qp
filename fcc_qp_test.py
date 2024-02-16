import os
import numpy as np
import matplotlib.pyplot as plt
from fcc_qp import FCCQPSolver, FCCQPSolution


def load_qp_matrices():
    data = np.load('test_data/id_qp_log_walking.npz', allow_pickle=True)
    return data['qps']


def solve(qp, solver):
    solver.Solve(
        qp['Q'], qp['b'], qp['A_eq'], qp['b_eq'], qp['friction_coeffs'], qp['lb'], qp['ub'])
    result = solver.GetSolution()
    return result


def main():
    qps = load_qp_matrices()
    solver = FCCQPSolver(50, 38, 12, 38)
    solver.set_rho(100)
    solver.set_eps(1e-4)

    u_sol = []
    idxs = []
    ts = []
    for i in range(len(qps)):
        result = solve(qps[i], solver)
        u_sol.append(result.z[22:32])
        idxs.append(i)
        ts.append(result.solve_time)

    plt.plot(idxs, u_sol)
    plt.title('Input Solution')
    plt.xlabel('QP #')
    plt.ylabel('u (nm)')

    plt.figure()
    plt.plot(idxs, ts)
    plt.title('Solve Times')
    plt.xlabel('QP #')
    plt.ylabel('Solve Time (s)')
    plt.show()


if __name__ == '__main__':
    main()