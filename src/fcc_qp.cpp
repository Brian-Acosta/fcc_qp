#include <chrono>
#include <vector>
#include <tuple>
#include <cassert>

#include <Eigen/Dense>

#include "fcc_qp.hpp"
#include "constraint_utils.hpp"

using std::abs;
using std::max;
using std::sqrt;
using std::vector;

namespace fcc_qp {

using Eigen::Ref;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::LDLT;
using Eigen::CompleteOrthogonalDecomposition;

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
  M_kkt_factorization_ = LDLT<MatrixXd>();
  M_kkt_pre_factorization_ = LDLT<MatrixXd>();
  M_kkt_pre_factorization_backup_ = CompleteOrthogonalDecomposition<MatrixXd>(N, N);

  b_kkt_ = VectorXd::Zero(N);
  kkt_sol_ = VectorXd::Zero(N);
  x_ = VectorXd::Zero(n_vars_);
  x_bar_ = VectorXd::Zero(n_vars_);
  x_res_ = VectorXd::Zero(n_vars_);
  mu_x_ = VectorXd::Zero(n_vars_);
  lambda_c_bar_ = VectorXd::Zero(nc_);
  mu_lambda_c_ = VectorXd::Zero(nc_);
  lambda_c_res_ = VectorXd::Zero(nc_);
}

void FCCQP::DoADMM(
    const Ref<const VectorXd> &b, const vector<double> &friction_coeffs,
    const Ref<const VectorXd> &lb, const Ref<const VectorXd> &ub) {

  // Initialize KKT system for primal updates
  P_rho_.diagonal() = options_.rho * VectorXd::Ones(n_vars_);
  M_kkt_ = M_kkt_pre_;
  M_kkt_.topLeftCorner(n_vars_, n_vars_) += P_rho_;

  // Factorize KKT matrix
  auto fact_start = std::chrono::high_resolution_clock::now();
  M_kkt_factorization_.compute(M_kkt_);
  auto fact_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> fact_time = fact_end - fact_start;
  factorization_time_ += fact_time.count();

  // Initialize slack to solution to equality constrained QP
  x_bar_ = x_;
  lambda_c_bar_ = x_.segment(lambda_c_start_, nc_);

  // ADMM iterations
  n_iter_ = options_.max_iter;
  for (int iter = 0; iter < options_.max_iter; ++iter) {
    // Update KKT system RHS
    q_rho_ = -options_.rho * (x_bar_ - mu_x_);
    q_rho_.segment(lambda_c_start_, nc_) = -options_.rho * (lambda_c_bar_ - mu_lambda_c_);
    b_kkt_.head(n_vars_) = -(b + q_rho_);

    // Primal update - solve equality constrained QP
    kkt_sol_ = M_kkt_factorization_.solve(b_kkt_);
    x_ = kkt_sol_.head(n_vars_);

    // Slack update - project to feasible set
    x_bar_ = project_to_bounds(x_ + mu_x_, lb, ub);
    lambda_c_bar_ = project_to_friction_cone(
        x_.segment(lambda_c_start_, nc_) + mu_lambda_c_, friction_coeffs);

    // Calculate residual between primal and slack variables
    x_res_ = x_ - x_bar_;
    lambda_c_res_ = x_.segment(lambda_c_start_, nc_) - lambda_c_bar_;
    x_res_norm_ = abs(max(x_res_.maxCoeff(), -x_res_.minCoeff())) ;
    lambda_c_res_norm_ = abs(max(lambda_c_res_.maxCoeff(), -lambda_c_res_.minCoeff()));

    // dual update
    mu_x_ += x_res_;
    mu_lambda_c_ += lambda_c_res_;

    // check for convergence
    if (lambda_c_res_norm_ < options_.eps_fcone and
        x_res_norm_ < options_.eps_bound) {
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
  assert(Q.rows() == Q.cols());
  assert(Q.rows() == n_vars_);
  assert(b.rows() == n_vars_);
  assert(A_eq.cols() == n_vars_);
  assert(A_eq.rows() == b_eq.rows());
  assert(friction_coeffs.size() == nc_ / 3);
  assert(lb.rows() == ub.rows());
  assert(lb.rows() == n_vars_);

  bool equality_constrained =
      nc_ == 0 and lb.array().isInf().all() and ub.array().isInf().all();

  // re-zero relevant matrices
  if (not warm_start_) {
    mu_x_.setZero();
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
  x_res_norm_ = 0;
  lambda_c_res_norm_ = 0;

  // Factorize equality constrained QP matrix without rho
  if (equality_constrained or not warm_start_) {
    auto fact_start = std::chrono::high_resolution_clock::now();
    M_kkt_pre_factorization_.compute(M_kkt_pre_);

    // For equality constrained QPs or if the cost is not positive definite,
    // use a more robust solver
    if (M_kkt_pre_factorization_.info() != Eigen::Success) {
      M_kkt_pre_factorization_backup_.compute(M_kkt_pre_);
      std::cout << "ldlt failed\n";
    }

    auto fact_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> fact_time = fact_end - fact_start;
    factorization_time_ += fact_time.count();

    // Get initial guess by solving equality constrained QP
    if (M_kkt_pre_factorization_.info() == Eigen::Success) {
      x_ = M_kkt_pre_factorization_.solve(b_kkt_).head(n_vars_);
    } else {
      x_ = M_kkt_pre_factorization_backup_.solve(b_kkt_).head(n_vars_);
    }
  }

  if (not equality_constrained) {
    std::cout << "running admm\n";
    DoADMM(b, friction_coeffs, lb, ub);
  }

  bounds_viol_ = calc_bound_violation(x_, lb, ub);
  friction_cone_viol_ = calc_friction_cone_violation(
      x_.segment(lambda_c_start_, nc_), friction_coeffs);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> solve_time = end - start;
  solve_time_ = solve_time.count();
}


FCCQPSolution FCCQP::GetSolution() const {
  FCCQPSolution out;
  out.details.admm_residual_bounds = x_res_norm_;
  out.details.admm_residual_friction_cone = lambda_c_res_norm_;
  out.details.bounds_viol = bounds_viol_;
  out.details.friction_cone_viol = friction_cone_viol_;
  out.details.solve_time = solve_time_;
  out.details.factorization_time = factorization_time_;
  out.details.n_iter = n_iter_;
  out.details.solve_status = (n_iter_ == options_.max_iter) ?
      kMaxIterations : kSuccess;
  out.z = x_;
  return out;
}

}