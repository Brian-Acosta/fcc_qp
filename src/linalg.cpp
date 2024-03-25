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
  constraint_kernel_ = Z.bottomRows(n_ - rank).transpose();
  reduced_hessian_solver_.compute(
      constraint_kernel_.transpose() * Q * constraint_kernel_);
  Q_ = Q;
}

VectorXd KKTSolver::solve(const VectorXd &b, const VectorXd &b_eq) const {
  VectorXd x = constraint_solver_.solve(b_eq);
  VectorXd rhs = -constraint_kernel_.transpose() * (Q_ * x + b);
  VectorXd z = reduced_hessian_solver_.solve(rhs);
  return  constraint_kernel_ * z + x;
}

}
