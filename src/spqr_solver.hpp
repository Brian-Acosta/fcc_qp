#include <vector>
#include <memory>
#include <Eigen/Dense>
#include "SuiteSparseQR.hpp"


/// This file is currently unused because either
/// - SuiteSparse QR (even the min2norm solve) is not accurate enough for the
/// fcc_qp demo problems
/// - there is a bug
/// - both

namespace fcc_qp {

/// Wrapper for SuiteSparse workspace for solving KKT matrix
class SPQRSolver {
 public:
  explicit SPQRSolver(int n);
  ~SPQRSolver();

  void compute(const Eigen::MatrixXd& A);
  Eigen::VectorXd solve(const Eigen::VectorXd& b);

 private:
  const int n_;
  const int max_nnz_;
  cholmod_common common_{};
  cholmod_sparse* A_ = nullptr;
  cholmod_dense* b_ = nullptr;
  cholmod_triplet* triplet_ = nullptr;
  SuiteSparseQR_factorization<double>* QR_ = nullptr;

  std::vector<int> triplet_j_;
  std::vector<int> triplet_i_;
  std::vector<double> triplet_val_;
};

}