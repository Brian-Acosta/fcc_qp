#pragma once
#include <Eigen/Dense>

namespace fcc_qp {

class KKTSolver {
 public:
  KKTSolver(int n) explicit;

  void compute(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& A_eq);
  void solve(const Eigen::VectorXd& b, const Eigen::VectorXd& b_eq) const;
  void solve(const Eigen::VectorXd& b_kkt) const {
    return solve(b_kkt.head(n_), b_kkt.tail(n_eq_);
  }
 private:
  const int n_;
  const int n_eq_;
  Eigen::LDLT<Eigen::MatrixXd> hessian_solver_;
  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> constraint_solver_;
  Eigen::MatrixXd constraint_kernel_;
};


}