#include "linalg.hpp"
#include <iostream>
namespace fcc_qp {

using Eigen::LLT;
using Eigen::LDLT;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::FullPivLU;
using Eigen::ColPivHouseholderQR;
using Eigen::FullPivHouseholderQR;
using Eigen::CompleteOrthogonalDecomposition;
using Eigen::SelfAdjointEigenSolver;


KKTSolver::KKTSolver(int n, int n_eq) : n_(n), n_eq_(n_eq) {
  full_kkt_solver_ = LDLT<MatrixXd>();
  reduced_hessian_solver_ = LDLT<MatrixXd>();
  constraint_solver_ = CompleteOrthogonalDecomposition<MatrixXd>(n_eq, n);
  Q_ = MatrixXd::Zero(n,n);
}

void KKTSolver::compute(const MatrixXd& Q, const MatrixXd& A_eq) {

  constraint_solver_.compute(A_eq);
  long rank = constraint_solver_.rank();
  MatrixXd Z = constraint_solver_.matrixZ() * constraint_solver_
      .colsPermutation().inverse();
  MatrixXd N = Z.bottomRows(n_ - rank).transpose();

  MatrixXd kkt = MatrixXd::Zero(n_ + n_eq_, n_ + n_eq_);
  kkt.topLeftCorner(n_, n_) = Q;
  kkt.topRightCorner(n_, n_eq_) = A_eq.transpose();
  kkt.bottomLeftCorner(n_eq_, n_) = A_eq;
  full_kkt_solver_.compute(kkt);
}

VectorXd KKTSolver::solve(const VectorXd &b, const VectorXd &b_eq) const {
  VectorXd b_kkt = VectorXd::Zero(n_ + n_eq_);
  b_kkt.head(n_) = -b;
  b_kkt.tail(n_eq_) = b_eq;
  return full_kkt_solver_.solve(b_kkt).head(n_);
}

}
