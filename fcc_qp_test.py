import numpy as np
import matplotlib.pyplot as plt
from fcc_qp import FCCQPSolver, FCCQPSolution

def B():
    return np.array(
        [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
         [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
    )


def load_qp_matrices():
    data = np.load('/Volumes/Extreme SSD/id_qp_log_running.npz', allow_pickle=True)
    return data['arr_0']


def solve(qp_data):
    nv = qp_data['M'].shape[0]
    nu = qp_data['B'].shape[1]
    nh = qp_data['Jh'].shape[0]
    nc = qp_data['Jc'].shape[0]
    n = nv+nu+nh+nc
    nc_active = qp_data['Jc_active'].shape[0]



    lb = np.full((n,), -np.inf)
    ub = np.full((n,), np.inf)
    lb[nv:nv+nu] = -150. * np.ones((nu,))
    ub[nv:nv+nu] = 150. * np.ones((nu,))

    Q = qp_data['Q'][:n, :n]

    A = np.zeros((nv+nh+nc_active, nv+nu+nh+nc))
    A[:nv, :nv] = qp_data['M']
    A[:nv, nv:nv+nu] = - B()  # Temporary workaround for bug in data collection
    A[:nv, nv+nu:nv+nu+nh] = -qp_data['Jh'].T
    A[:nv, nv+nu+nh:nv+nu+nh+nc] = -qp_data['Jc'].T
    A[nv:nv+nh, :nv] = qp_data['Jh']
    A[nv+nh:nv+nh+nc_active, :nv] = qp_data['Jc_active']

    b = qp_data['b'][:n]
    b_eq = np.hstack((-qp_data['bias'], -qp_data['JdotV_h'], -qp_data['JdotV_c']))

    friction_coeffs = [1.0 for _ in range(int(nc / 3))]

    solver = FCCQPSolver(n, A.shape[0], nc, nv+nu+nh)
    result = solver.Solve(Q, b, A, b_eq, friction_coeffs, lb, ub, False)
    return result
def main():
    qps = load_qp_matrices()

    # _, _, _ = solve(qps[320])

    u_sol = []
    idxs = []
    ts = []
    for i in range(len(qps)):
        result = solve(qps[i])
        u_sol.append(result.z[nv:nv+nu])
        idxs.append(i)
        ts.append(result.solve_time)
        #print(f'Input Lim Res: {u_res}, Fiction Cone Res: {lambda_res}, Solve Time: {t}')

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