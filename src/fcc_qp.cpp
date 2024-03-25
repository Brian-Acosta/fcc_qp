#include <chrono>
#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include <cassert>
#include "fcc_qp.hpp"
#include "constraint_utils.hpp"

using std::abs;
using std::max;
using std::sqrt;
using std::vector;

namespace fcc_qp {

using Eigen::VectorXd;
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
  P_rho_ = options_.rho * MatrixXd::Identity(n_vars_, n_vars_);
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

double calc_bound_violation(
    const VectorXd& x, const VectorXd& lb, const VectorXd& ub) {
  return (x - project_to_bounds(x, lb, ub)).norm();
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
  n_iter_ = options_.max_iter;
  for (int iter = 0; iter < options_.max_iter; ++iter) {
    // Update KKT system RHS
    q_rho_ = -options_.rho * (z_bar_ - mu_z_);
    q_rho_.segment(lambda_c_start_, nc_) = -options_.rho * (lambda_c_bar_ - mu_lambda_c_);
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
    if (lambda_c_res_norm_ < options_.eps_fcone and
        z_res_norm_ < options_.eps_bound) {
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

  assert(validate_bounds(lb, ub));

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

    if (M_kkt_pre_factorization_.info() != Eigen::Success or
        equality_constrained) {
      M_kkt_pre_factorization_backup_.compute(M_kkt_pre_);
    }

    auto fact_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> fact_time = fact_end - fact_start;
    factorization_time_ += fact_time.count();

    // Get initial guess by solving equality constrained QP
    if (M_kkt_pre_factorization_.info() == Eigen::Success and not
        equality_constrained) {
      z_ = M_kkt_pre_factorization_.solve(b_kkt_).head(n_vars_);
    } else {
      z_ = M_kkt_pre_factorization_backup_.solve(b_kkt_).head(n_vars_);
    }
  }

  if (not equality_constrained) {
    DoADMM(b, friction_coeffs, lb, ub);
  }

  bounds_viol_ = calc_bound_violation(z_, lb, ub);
  fricion_cone_viol_ = calc_friction_cone_violation(
      z_.segment(lambda_c_start_, nc_), friction_coeffs);

  if (options_.polish) {
    Polish(b, b_eq, friction_coeffs, lb, ub);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> solve_time = end - start;
  solve_time_ = solve_time.count();
}

void FCCQP::Polish(const Ref<const VectorXd> &b, const Ref<const VectorXd> &b_eq,
                   const vector<double> &friction_coeffs,
                   const Ref<const VectorXd> &lb, const Ref<const VectorXd> &ub) {

  if (fricion_cone_viol_ < options_.eps_fcone and bounds_viol_ < options_.eps_bound) {
    return;
  }

  MatrixXd A_fcone = get_active_set_friction_constraint(
      z_.segment(lambda_c_start_, nc_), friction_coeffs
  );
  std::pair<MatrixXd, VectorXd> Abb = get_active_set_bounds_constraint(z_, lb, ub);

  const MatrixXd& A_b = Abb.first;
  const VectorXd& b_b = Abb.second;

  int n = n_vars_ + n_eq_;
  int n_fc = A_fcone.rows();
  int n_b = A_b.rows();

  MatrixXd M = MatrixXd::Zero(n + n_fc + n_b, n + n_fc + n_b);

  M.topLeftCorner(n, n) = M_kkt_pre_;
  M.topLeftCorner(n_vars_, n_vars_) +=
      options_.delta_polish * MatrixXd::Identity(n_vars_, n_vars_);

  M.block(n, lambda_c_start_, A_fcone.rows(), A_fcone.cols()) = A_fcone;
  M.block(lambda_c_start_, n, A_fcone.cols(), A_fcone.rows()) =
      A_fcone.transpose();
  M.block(n + n_fc, 0, n_b, n_vars_) = A_b;
  M.block(0, n + n_fc, n_vars_, n_b) = A_b.transpose();

  VectorXd b_kkt_ext = VectorXd::Zero(n + n_fc + n_b);
  b_kkt_ext.head(n_vars_) = -b;
  b_kkt_ext.segment(n_vars_, n_eq_) = b_eq;
  b_kkt_ext.tail(n_b) = b_b;
  z_ = M.ldlt().solve(b_kkt_ext).head(n_vars_);
}

FCCQPSolution FCCQP::GetSolution() const {
  FCCQPSolution out;
  out.details.admm_residual_bounds = z_res_norm_;
  out.details.admm_residual_friction_cone = lambda_c_res_norm_;
  out.details.bounds_viol = bounds_viol_;
  out.details.friction_cone_viol = fricion_cone_viol_;
  out.details.solve_time = solve_time_;
  out.details.factorization_time = factorization_time_;
  out.details.n_iter = n_iter_;
  out.z = z_;
  return out;
}

}