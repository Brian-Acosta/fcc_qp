#include <iostream>
#include "spqr_solver.hpp"


namespace fcc_qp {


SPQRSolver::SPQRSolver(int n) : n_(n), max_nnz_(n * n) {
  assert(n >= 0);

  cholmod_l_start(&common_);

  triplet_i_.reserve(max_nnz_);
  triplet_j_.reserve(max_nnz_);
  triplet_val_.reserve(max_nnz_);

  triplet_ = cholmod_l_allocate_triplet(
      n_, n_, max_nnz_, 0, CHOLMOD_REAL, &common_);


  b_ = cholmod_l_zeros(n_, 1, CHOLMOD_REAL, &common_);
  assert(b_ != nullptr);
}

SPQRSolver::~SPQRSolver() {
  if (A_ != nullptr) {
    cholmod_l_free_sparse(&A_, &common_);
  }
  if (QR_ != nullptr) {
    SuiteSparseQR_free(&QR_, &common_);
  }
  if (b_ != nullptr) {
    cholmod_l_free_dense(&b_, &common_);
  }
  if (triplet_ != nullptr) {
    cholmod_l_free_triplet(&triplet_, &common_);
  }
  cholmod_l_finish(&common_);
}

void SPQRSolver::compute(const Eigen::MatrixXd &A) {
  triplet_i_.clear();
  triplet_j_.clear();
  triplet_val_.clear();

  for (int j = 0; j < A.cols(); ++j) {
    for (int i = 0; i < A.rows(); ++i) {
      if (A(i, j) != 0.0) {
        triplet_val_.push_back(A(i, j));
        triplet_i_.push_back(i);
        triplet_j_.push_back(j);
      }
    }
  }

  int nnz = triplet_val_.size();

  assert(triplet_ != nullptr);

  for (int ind = 0; ind < nnz; ++ind) {
    ((long int *) triplet_->i)[ind] = triplet_i_[ind];
    ((long int *) triplet_->j)[ind] = triplet_j_[ind];
    ((double *) triplet_->x)[ind] = triplet_val_[ind];
  }
  triplet_->nnz = nnz;

  if (A_ != nullptr) {
    cholmod_l_free_sparse(&A_, &common_);
    A_ = nullptr;
  }
  A_ = cholmod_l_triplet_to_sparse(triplet_, nnz, &common_);

  assert(A_ != nullptr);

  if (QR_ != nullptr) {
    SuiteSparseQR_free(&QR_, &common_);
    QR_ = nullptr;
  }
  QR_ = SuiteSparseQR_factorize<double>(
      SPQR_ORDERING_CHOLMOD, 0, A_, &common_);

  assert(QR_ != nullptr);
}

Eigen::VectorXd SPQRSolver::solve(const Eigen::VectorXd &b) {

  memcpy(b_->x, b.data(), sizeof(double) * n_);

  auto aux = SuiteSparseQR_qmult<double>(SPQR_QTX, QR_, b_, &common_);
  assert(aux != nullptr);


  auto sol = SuiteSparseQR_solve<double>(SPQR_RETX_EQUALS_B, QR_, aux,
                                         &common_);


  assert(sol != nullptr);
  assert(sol->x != nullptr);

  double* values = static_cast<double*>(sol->x);

  Eigen::VectorXd out = Eigen::VectorXd::Zero(n_);
  memcpy(out.data(), values, n_ * sizeof(double));

  cholmod_l_free_dense(&aux, &common_);
  cholmod_l_free_dense(&sol, &common_);

  return out;
}

}