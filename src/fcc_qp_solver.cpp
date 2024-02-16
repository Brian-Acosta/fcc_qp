#include <chrono>
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include "fcc_qp_solver.hpp"
#include <cassert>

namespace py = pybind11;

using std::abs;
using std::sqrt;
using std::max;
using std::min;
using std::vector;

namespace fcc_qp {

using Eigen::VectorXd;
using Eigen::Vector3d;
using Eigen::MatrixXd;
using Eigen::Ref;

FCCQPSolver::FCCQPSolver(int num_vars, int u_start, int nu,
                               int lambda_c_start, int nc,
                               int num_equality_constraints) :
    n_vars_(num_vars), nu_(nu), nc_(nc), u_start_(u_start),
    lambda_c_start_(lambda_c_start), n_eq_(num_equality_constraints) {
  assert(nu_ >= 0);
  assert(nc_ >= 0);
  assert(nc_ % 3 == 0);
  assert(u_start <= n_vars_ - nu_);
  assert(lambda_c_start <= n_vars_ - nc_);

  // Initialize workspace variables
  P_rho_ = MatrixXd::Zero(n_vars_, n_vars_);
  P_rho_.block(u_start_, u_start_, nu_, nu_) =
      rho_ * MatrixXd::Identity(nu_, nu_);
  P_rho_.bottomRightCorner(nc_, nc_) = rho_ * MatrixXd::Identity(nc_, nc_);
  q_ = VectorXd::Zero(n_vars_);
  q_rho_ = VectorXd::Zero(n_vars_);

  A_ = MatrixXd::Zero(n_eq_, n_vars_);

  int N = n_vars_ + A_.rows();
  M_kkt_ = MatrixXd::Zero(N,N);
  b_kkt_ = VectorXd::Zero(N);
  kkt_sol_ = VectorXd::Zero(N);
  z_ = VectorXd::Zero(n_vars_);
  u_bar_ = VectorXd::Zero(nu_);
  mu_u_ = VectorXd::Zero(nu_);
  u_res_ = VectorXd::Zero(nu_);
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

VectorXd project_to_input_bounds(
    const VectorXd& u, const VectorXd& lb, const VectorXd& ub) {
  VectorXd out(u.rows());
  for (int i = 0; i < u.rows(); ++i) {
    out(i) = max(min(u(i), ub(i)), lb(i));
  }
  return out;
}

}

void FCCQPSolver::Solve(
    const Ref<MatrixXd>& A_eq, const Ref<VectorXd>& b_eq,
    const Ref<MatrixXd>& Q, const Ref<VectorXd>& b,
    const vector<double>& friction_coeffs, const Ref<VectorXd>& u_lb,
    const Ref<VectorXd>& u_ub, bool warm_start) {

  auto start = std::chrono::high_resolution_clock::now();

  if (not warm_start) {
    mu_lambda_c_.setZero();
    mu_u_.setZero();
    lambda_c_bar_.setZero();
    u_bar_.setZero();
  }

  M_kkt_.topLeftCorner(n_vars_, n_vars_) = Q + P_rho_;

  M_kkt_.bottomLeftCorner(n_eq_, n_vars_) = A_;
  M_kkt_.topRightCorner(n_vars_, n_eq_) = A_.transpose();
  b_kkt_.segment(n_vars_, n_eq_) = b_eq;

  auto pinv_factorization = (M_kkt_.transpose() * M_kkt_).ldlt();

  for (int iter = 0; iter < max_iter_; ++iter) {
    q_rho_.segment(u_start_, nu_) = -rho_ * (u_bar_ - mu_u_);
    q_rho_.segment(lambda_c_start_, nc_) = -rho_ * (lambda_c_bar_ - mu_lambda_c_);
    b_kkt_.head(n_vars_) = -(b + q_rho_);
    kkt_sol_ = pinv_factorization.solve(M_kkt_.transpose() * b_kkt_);
    z_ = kkt_sol_.head(n_vars_);

    u_bar_ = project_to_input_bounds(
        z_.segment(u_start_, nu_) + mu_u_, u_lb, u_ub);

    lambda_c_bar_ = project_to_friction_cone(
        z_.segment(lambda_c_start_, nc_) + mu_lambda_c_, friction_coeffs);

    u_res_ = z_.segment(u_start_, nu_) - u_bar_;
    lambda_c_res_ = z_.segment(lambda_c_start_, nc_) - lambda_c_bar_;
    u_res_norm_ = u_res_.norm();
    lambda_c_res_norm_ = lambda_c_res_.norm();

    mu_u_ += u_res_;
    mu_lambda_c_ += lambda_c_res_;

    if (lambda_c_res_norm_ < eps_ and u_res_norm_ < eps_) {
      n_iter_ = iter;
      break;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> solve_time = end - start;
  solve_time_ = solve_time.count();
}

}