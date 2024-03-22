#inlcude "linalg.hpp"

namespace fcc_qp {

using Eigen::LDLT;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::CompleteOrthogonalDecomposition;


KKTSolver::KKTSolver(int n, int n_eq) : n_(n), n_eq_(n_eq_) {
  hessian_solver_ = LDLT<MatrixXd>;
  constraint_solver_ = CompleteOrthogonalDecomposition<MatrixXd>(n, n_eq);
}

void KKTSolver::compute(const MatrixXd& Q, const MatrixXd& A_eq) {
  hessian_solver_.compute(Q);
  constraint_solver_.compute(A_eq);
  constraint_kernel_ = constraint_solver_.kernel();
}

}
