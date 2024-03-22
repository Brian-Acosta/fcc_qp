#pragma once
#include <Eigen/Dense>

namespace fcc_qp {

class KKTSolver {
 public:
  explicit KKTSolver(int n, int n_eq);

  void compute(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& A_eq);
  Eigen::VectorXd solve(const Eigen::VectorXd& b, const Eigen::VectorXd& b_eq) const;
  Eigen::VectorXd solve(const Eigen::VectorXd& b_kkt) const {
    return solve(b_kkt.head(n_), b_kkt.tail(n_eq_));
  }
 private:
  const int n_;
  const int n_eq_;
  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> reduced_hessian_solver_;
  Eigen::FullPivHouseholderQR<Eigen::MatrixXd> constraint_solver_;
  Eigen::MatrixXd constraint_kernel_;
  Eigen::MatrixXd Q_;
};


}