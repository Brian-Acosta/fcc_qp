#include <chrono>
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include "fcc_qp.hpp"
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

FCCQP::FCCQP(int num_vars, int num_equality_constraints,
                         int nc, int lambda_c_start) :
    n_vars_(num_vars), n_eq_(num_equality_constraints), nc_(nc),
    lambda_c_start_(lambda_c_start) ,
    spqr_M_kkt_(num_vars + num_equality_constraints),
    spqr_M_kkt_pre_(num_vars + num_equality_constraints) {

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

Vector3d project_to_friction_cone(const Vector3d& f, double mu) {
  double norm_fxy = f.head<2>().norm();

  // inside the friction cone, do nothing
  if (mu * f(2) >=  norm_fxy) {
    return f;
  }

  // More than 90 degrees from the side of the cone, closest point is the origin
  if (f(2) < -mu * norm_fxy) {
    return Vector3d::Zero();
  }

  // project up to the side of the friction cone
  double xy_ratio = mu * f(2)  / norm_fxy;
  Vector3d cone_ray(xy_ratio * f(0), xy_ratio * f(1), f(2));
  cone_ray.normalize();
  return cone_ray.dot(f) * cone_ray;

}

VectorXd project_to_friction_cone(
    const VectorXd& f, const vector<double>& friction_coeffs) {
  VectorXd out = VectorXd::Zero(f.rows());
  for (int i = 0; i < f.rows() / 3; ++i) {
    out.segment<3>(3*i) = project_to_friction_cone(
        f.segment<3>(3*i), friction_coeffs.at(i));
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

double calc_friction_cone_violation(
    const VectorXd& f, const vector<double>& friction_coeffs) {
  double violation = 0;

  for (int i = 0; i < f.rows() / 3; ++i) {
    int start = 3*i;
    double fz = f(start + 2);
    double mu = friction_coeffs.at(i);
    violation += max(0., f.segment<2>(start).norm() -mu * fz);
  }
  return violation;
}

double calc_bound_violation(
    const VectorXd& x, const VectorXd& lb, const VectorXd& ub) {
  return (x - project_to_bounds(x, lb, ub)).norm();
}

}

void FCCQP::Solve(
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
  auto fact_start = std::chrono::high_resolution_clock::now();
  spqr_M_kkt_.compute(M_kkt_pre_);
  spqr_M_kkt_pre_.compute(M_kkt_);

  auto fact_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> fact_time = fact_end - fact_start;
  factorization_time_ = fact_time.count();

  z_ = spqr_M_kkt_pre_.solve(b_kkt_).head(n_vars_);

  z_bar_ = z_;
  lambda_c_bar_ = z_.segment(lambda_c_start_, nc_);

  n_iter_ = max_iter_;
  for (int iter = 0; iter < max_iter_; ++iter) {
    q_rho_ = -rho_ * (z_bar_ - mu_z_);
    q_rho_.segment(lambda_c_start_, nc_) = -rho_ * (lambda_c_bar_ - mu_lambda_c_);
    b_kkt_.head(n_vars_) = -(b + q_rho_);
    kkt_sol_ = spqr_M_kkt_.solve(b_kkt_);
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
  bounds_viol_ = calc_bound_violation(z_, lb, ub);
  fricion_con_viol_ = calc_friction_cone_violation(
      z_.segment(lambda_c_start_, nc_), friction_coeffs);

}

FCCQPSolution FCCQP::GetSolution() const {
  FCCQPSolution out;
  out.details.eps_bounds = z_res_norm_;
  out.details.eps_friction_cone = lambda_c_res_norm_;
  out.details.bounds_viol = bounds_viol_;
  out.details.friction_cone_viol = fricion_con_viol_;
  out.details.solve_time = solve_time_;
  out.details.factorization_time = factorization_time_;
  out.details.n_iter = n_iter_;
  out.z = z_;
  return out;
}

}