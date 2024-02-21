#include "spqr_solver.hpp"

namespace fcc_qp {

SPQRSolver::SPQRSolver(int n) : n_(n), max_nnz_(n*n) {
  assert(n >= 0);

  triplet_ = cholmod_allocate_triplet(
      n_, n_, max_nnz_, CHOLMOD_MM_SYMMETRIC, CHOLMOD_REAL, &common_);
  assert(triplet_ != nullptr);

  triplet_i_.reserve(max_nnz_);
  triplet_j_.reserve(max_nnz_);
  triplet_val_.reserve(max_nnz_);

  b_ = cholmod_allocate_dense(n_, 1, 1, CHOLMOD_REAL, &common_);

}

void SPQRSolver::compute(const Eigen::MatrixXd &A) {
  triplet_i_.clear();
  triplet_j_.clear();
  triplet_val_.clear();

  for (int j = 0; j < A.cols(); ++j) {
    for (int i = 0; i < A.rows(); ++i) {
      if (A(i,j) != 0) {
        triplet_val_.push_back(A(i,j));
        triplet_i_.push_back(i);
        triplet_j_.push_back(j);
      } else {
        triplet_i_.push_back(0);
        triplet_j_.push_back(0);
        triplet_val_.push_back(0);
      }
    }
  }

  std::memcpy(triplet_->i, triplet_i_.data(), sizeof(int) * max_nnz_);
  std::memcpy(triplet_->j, triplet_j_.data(), sizeof(int) * max_nnz_);
  std::memcpy(triplet_->x, triplet_val_.data(), sizeof(double) * max_nnz_);

  if (A_ != nullptr) {
    cholmod_free_sparse(&A_, &common_);
  }
  A_ = cholmod_triplet_to_sparse(triplet_, max_nnz_, &common_);

  if (A_factorization_ != nullptr) {
    SuiteSparseQR_free(&A_factorization_, &common_);
  }
  A_factorization_ = SuiteSparseQR_factorize<double>(
      SPQR_ORDERING_AMD, 0, A_, &common_);

}

Eigen::VectorXd SPQRSolver::solve(const Eigen::VectorXd& b) {
  memcpy(b_->x, b.data(), sizeof(double) * n_);
  auto sol = SuiteSparseQR_solve()
}

}