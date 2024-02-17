#include <chrono>
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include "fcc_qp_solver.hpp"
#include <cassert>

using std::abs;
using std::sqrt;
using std::max;
using std::min;
using std::vector;

namespace fcc_qp {

using Eigen::VectorXd;
using Eigen::Vector3d;
using Eigen::MatrixXd;
using Eigen::BDCSVD;
using Eigen::CompleteOrthogonalDecomposition;
using Eigen::Ref;

FCCQPSolver::FCCQPSolver(int num_vars, int num_equality_constraints,
                         int nc, int lambda_c_start) :
    n_vars_(num_vars), n_eq_(num_equality_constraints), nc_(nc),
    lambda_c_start_(lambda_c_start) {
  assert(nc_ >= 0);
  assert(nc_ % 3 == 0);
  assert(lambda_c_start <= n_vars_ - nc_);

  // Initialize workspace variables
  P_rho_ = rho_ * MatrixXd::Identity(n_vars_, n_vars_);
  q_rho_ = VectorXd::Zero(n_vars_);

  int N = n_vars_ + n_eq_;
  M_kkt_ = MatrixXd::Zero(N,N);
  M_kkt_pre_ = MatrixXd::Zero(N, N);
  M_kkt_factorization_ = CompleteOrthogonalDecomposition<MatrixXd>(N, N);
  M_kkt_pre_factorization_ = CompleteOrthogonalDecomposition<MatrixXd>(N, N);

  b_kkt_ = VectorXd::Zero(N);
  kkt_sol_ = VectorXd::Zero(N);
  z_ = VectorXd::Zero(n_vars_);
  z_bar_ = VectorXd::Zero(n_vars_);
  z_res_ = VectorXd::Zero(n_vars_);
  mu_z_ = VectorXd::Zero(n_vars_);
  lambda_c_bar_ = VectorXd::Zero(nc_);
  mu_lambda_c_ = VectorXd::Zero(nc_);
  lambda_c_res_ = VectorXd::Zero(nc_);
}

namespace {

VectorXd project_to_friction_cone(
    const VectorXd& f, const vector<double>& friction_coeffs) {
  VectorXd out = VectorXd::Zero(f.rows());
  for (int i = 0; i < f.rows() / 3; ++i) {
    int start = 3*i;
    double fz = max(f(start + 2), 0.);
    double norm_xy = f.segment(start, 2).norm();
    double num = min(
        friction_coeffs.at(i) * fz, norm_xy);
    double ratio = (norm_xy > 0) ? num / norm_xy : 1.0;

    out.segment(start, 2) = ratio * f.segment(start, 2);
    out(start+2) = fz;
  }
  return out;
}

VectorXd project_to_bounds(
    const VectorXd& x, const VectorXd& lb, const VectorXd& ub) {
  VectorXd out(x.rows());
  for (int i = 0; i < x.rows(); ++i) {
    out(i) = max(min(x(i), ub(i)), lb(i));
  }
  return out;
}

}

void FCCQPSolver::Solve(
    const Ref<const MatrixXd>& Q, const Ref<const VectorXd>& b,
    const Ref<const MatrixXd>& A_eq, const Ref<const VectorXd>& b_eq,
    const vector<double>& friction_coeffs, const Ref<const VectorXd>& lb,
    const Ref<const VectorXd>& ub) {

  auto start = std::chrono::high_resolution_clock::now();

  mu_z_.setZero();
  mu_lambda_c_.setZero();

  M_kkt_.setZero();
  M_kkt_pre_.setZero();
  b_kkt_.setZero();

  M_kkt_pre_.topLeftCorner(n_vars_, n_vars_) = Q;
  M_kkt_pre_.bottomLeftCorner(n_eq_, n_vars_) = A_eq;
  M_kkt_pre_.topRightCorner(n_vars_, n_eq_) = A_eq.transpose();
  M_kkt_ = M_kkt_pre_;
  M_kkt_.topLeftCorner(n_vars_, n_vars_) += P_rho_;

  b_kkt_.head(n_vars_) = -b;
  b_kkt_.segment(n_vars_, n_eq_) = b_eq;

  // presolve without rho
  M_kkt_pre_factorization_.compute(M_kkt_pre_);
  z_ = M_kkt_pre_factorization_.solve(b_kkt_).head(n_vars_);

  z_bar_ = z_;
  lambda_c_bar_ = z_.segment(lambda_c_start_, nc_);

  M_kkt_factorization_.compute(M_kkt_);

  for (int iter = 0; iter < max_iter_; ++iter) {
    q_rho_ = -rho_ * (z_bar_ - mu_z_);
    q_rho_.segment(lambda_c_start_, nc_) = -rho_ * (lambda_c_bar_ - mu_lambda_c_);
    b_kkt_.head(n_vars_) = -(b + q_rho_);
    kkt_sol_ = M_kkt_factorization_.solve(b_kkt_);
    z_ = kkt_sol_.head(n_vars_);

    z_bar_ = project_to_bounds(z_ + mu_z_, lb, ub);

    lambda_c_bar_ = project_to_friction_cone(
        z_.segment(lambda_c_start_, nc_) + mu_lambda_c_, friction_coeffs);

    z_res_ = z_ - z_bar_;
    lambda_c_res_ = z_.segment(lambda_c_start_, nc_) - lambda_c_bar_;
    z_res_norm_ = z_res_.norm();
    lambda_c_res_norm_ = lambda_c_res_.norm();

    mu_z_ += z_res_;
    mu_lambda_c_ += lambda_c_res_;

    if (lambda_c_res_norm_ < eps_ and z_res_norm_ < eps_) {
      n_iter_ = iter;
      break;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> solve_time = end - start;
  solve_time_ = solve_time.count();
}

FCCQPSolution FCCQPSolver::GetSolution() const {
  FCCQPSolution out;
  out.details.eps_bounds = z_res_norm_;
  out.details.eps_friction_cone = lambda_c_res_norm_;
  out.details.solve_time = solve_time_;
  out.details.n_iter = n_iter_;
  out.z = z_;
  return out;
}

}