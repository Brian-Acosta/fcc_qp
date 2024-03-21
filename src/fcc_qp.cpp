#include <chrono>
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
    lambda_c_start_(lambda_c_start) {

  assert(n_vars_ >= 0);
  assert(n_eq_ >= 0);
  assert(nc_ >= 0);
  assert(nc_ % 3 == 0);
  assert(lambda_c_start_ + nc_ <= n_vars_);

  // Initialize workspace variables
  P_rho_ = rho_ * MatrixXd::Identity(n_vars_, n_vars_);
  q_rho_ = VectorXd::Zero(n_vars_);

  int N = n_vars_ + n_eq_;
  M_kkt_ = MatrixXd::Zero(N,N);
  M_kkt_pre_ = MatrixXd::Zero(N, N);
  M_kkt_factorization_ = Eigen::LDLT<MatrixXd>();
  M_kkt_pre_factorization_ = Eigen::LDLT<MatrixXd>();
  M_kkt_pre_factorization_backup_ = CompleteOrthogonalDecomposition<MatrixXd>(N, N);

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

void FCCQP::DoADMM(
    const Ref<const VectorXd> &b, const vector<double> &friction_coeffs,
    const Ref<const VectorXd> &lb, const Ref<const VectorXd> &ub) {

  // Initialize KKT system for primal updates
  M_kkt_ = M_kkt_pre_;
  M_kkt_.topLeftCorner(n_vars_, n_vars_) += P_rho_;

  // Factorize KKT matrix
  auto fact_start = std::chrono::high_resolution_clock::now();
  M_kkt_factorization_.compute(M_kkt_);
  auto fact_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> fact_time = fact_end - fact_start;
  factorization_time_ += fact_time.count();

  // Initialize slack to solution to equality constrained QP
  z_bar_ = z_;
  lambda_c_bar_ = z_.segment(lambda_c_start_, nc_);

  // ADMM iterations
  n_iter_ = max_iter_;
  for (int iter = 0; iter < max_iter_; ++iter) {
    // Update KKT system RHS
    q_rho_ = -rho_ * (z_bar_ - mu_z_);
    q_rho_.segment(lambda_c_start_, nc_) = -rho_ * (lambda_c_bar_ - mu_lambda_c_);
    b_kkt_.head(n_vars_) = -(b + q_rho_);

    // Primal update - solve equality constrained QP
    kkt_sol_ = M_kkt_factorization_.solve(b_kkt_);
    z_ = kkt_sol_.head(n_vars_);

    // Slack update - project to feasible set
    z_bar_ = project_to_bounds(z_ + mu_z_, lb, ub);
    lambda_c_bar_ = project_to_friction_cone(
        z_.segment(lambda_c_start_, nc_) + mu_lambda_c_, friction_coeffs);

    // Calculate residual between primal and slack variables
    z_res_ = z_ - z_bar_;
    lambda_c_res_ = z_.segment(lambda_c_start_, nc_) - lambda_c_bar_;
    z_res_norm_ = abs(max(z_res_.maxCoeff(), -z_res_.minCoeff())) ;
    lambda_c_res_norm_ = abs(max(lambda_c_res_.maxCoeff(), -lambda_c_res_.minCoeff()));

    // dual update
    mu_z_ += z_res_;
    mu_lambda_c_ += lambda_c_res_;

    // check for convergence
    if (lambda_c_res_norm_ < eps_ and z_res_norm_ < eps_) {
      n_iter_ = iter;
      break;
    }
  }


}

void FCCQP::Solve(
    const Ref<const MatrixXd>& Q, const Ref<const VectorXd>& b,
    const Ref<const MatrixXd>& A_eq, const Ref<const VectorXd>& b_eq,
    const vector<double>& friction_coeffs, const Ref<const VectorXd>& lb,
    const Ref<const VectorXd>& ub) {

  auto start = std::chrono::high_resolution_clock::now();

  bool equality_constrained =
      nc_ == 0 and lb.array().isInf().all() and ub.array().isInf().all();

  // re-zero relevant matrices

  if (not warm_start_) {
    mu_z_.setZero();
    mu_lambda_c_.setZero();
  }
  M_kkt_pre_.setZero();
  M_kkt_.setZero();
  b_kkt_.setZero();

  // Assemble KKT system for pre-solve QP
  M_kkt_pre_.topLeftCorner(n_vars_, n_vars_) = Q;
  M_kkt_pre_.bottomLeftCorner(n_eq_, n_vars_) = A_eq;
  M_kkt_pre_.topRightCorner(n_vars_, n_eq_) = A_eq.transpose();
  b_kkt_.head(n_vars_) = -b;
  b_kkt_.segment(n_vars_, n_eq_) = b_eq;

  // Reset solve stats
  factorization_time_ = 0;
  n_iter_ = 0;
  z_res_norm_ = 0;
  lambda_c_res_norm_ = 0;

  // Factorize Equality constrained QP matrix
  if (equality_constrained or not warm_start_) {
    auto fact_start = std::chrono::high_resolution_clock::now();
    M_kkt_pre_factorization_.compute(M_kkt_pre_);

    if (M_kkt_pre_factorization_.info() != Eigen::Success) {
      M_kkt_pre_factorization_backup_.compute(M_kkt_pre_);
    }

    auto fact_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> fact_time = fact_end - fact_start;
    factorization_time_ += fact_time.count();

    // Get initial guess by solving equality constrained QP
    if (M_kkt_pre_factorization_.info() == Eigen::Success) {
      z_ = M_kkt_pre_factorization_.solve(b_kkt_).head(n_vars_);
    } else {
      z_ = M_kkt_pre_factorization_backup_.solve(b_kkt_).head(n_vars_);
    }
  }

  if (not equality_constrained) {
    DoADMM(b, friction_coeffs, lb, ub);
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
  out.details.admm_residual_bounds = z_res_norm_;
  out.details.admm_residual_friction_cone = lambda_c_res_norm_;
  out.details.bounds_viol = bounds_viol_;
  out.details.friction_cone_viol = fricion_con_viol_;
  out.details.solve_time = solve_time_;
  out.details.factorization_time = factorization_time_;
  out.details.n_iter = n_iter_;
  out.z = z_;
  return out;
}

}