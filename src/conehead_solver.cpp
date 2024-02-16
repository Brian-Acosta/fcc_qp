#include <chrono>
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include "conehead_solver.hpp"
#include <cassert>

namespace py = pybind11;

using std::abs;
using std::sqrt;
using std::max;
using std::min;
using std::vector;

namespace conehead {

using Eigen::VectorXd;
using Eigen::Vector3d;
using Eigen::MatrixXd;
using Eigen::Ref;

ConeHeadSolver::ConeHeadSolver(int nv, int nu, int nh, int nc, int nc_active) :
  nv_(nv), nu_(nu), nh_(nh), nc_(nc), nc_active_(nc_active),
  n_(nv + nu + nh + nc) {
  assert(nv_ >= 0 );
  assert(nu_ >= 0);
  assert(nh_ >= 0);
  assert(nc_ >= 0);
  assert(nc_ % 3 == 0);

  // Initialize workspace variables
  P_rho_ = MatrixXd::Zero(n_, n_);
  P_rho_.block(nv_, nv_, nu_, nu_) = rho_ * MatrixXd::Identity(nu_, nu_);
  P_rho_.bottomRightCorner(nc_, nc_) = rho_ * MatrixXd::Identity(nc_, nc_);
  q_ = VectorXd::Zero(n_);
  q_rho_ = VectorXd::Zero(n_);
  A_ = MatrixXd::Zero(nv_ + nh_ + nc_active_, n_);

  int N = n_ + A_.rows();
  M_kkt_ = MatrixXd::Zero(N,N);
  b_kkt_ = VectorXd::Zero(N);

  kkt_sol_ = VectorXd::Zero(N);
  z_ = VectorXd::Zero(n_);
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

void ConeHeadSolver::Solve(const Ref<MatrixXd> &M,
                           const Ref<MatrixXd> &B,
                           const Ref<MatrixXd> &Jh,
                           const Ref<MatrixXd> &Jc,
                           const Ref<MatrixXd> &Jc_active,
                           const Ref<VectorXd> &bias,
                           const Ref<VectorXd> &Jh_dotV,
                           const Ref<VectorXd> &Jc_dotV_active,
                           const Ref<MatrixXd> &Q,
                           const Ref<VectorXd> &b,
                           const vector<double> &friction_coeffs,
                           const Ref<VectorXd> &u_lb,
                           const Ref<VectorXd> &u_ub,
                           bool warm_start) {

  auto start = std::chrono::high_resolution_clock::now();

  if (not warm_start) {
    mu_lambda_c_.setZero();
    mu_u_.setZero();
    lambda_c_bar_.setZero();
    u_bar_.setZero();
  }

  M_kkt_.topLeftCorner(n_, n_) = Q + P_rho_;

  A_.block(0, 0, nv_, nv_) = M;
  A_.block(0, nv_, nv_, nu_) = -B;
  A_.block(0, nv_ + nu_, nv_, nh_) = -Jh.transpose();
  A_.block(0, nv_ + nu_ + nh_, nv_, nc_) = -Jc.transpose();
  A_.block(nv_, 0, nh_, nv_) = Jh;
  A_.block(nv_ + nh_, 0, nc_active_, nv_) = Jc_active;

  M_kkt_.bottomLeftCorner(nv_ + nh_ + nc_active_, n_) = A_;
  M_kkt_.topRightCorner(n_, nv_ + nh_ + nc_active_) = A_.transpose();


  b_kkt_.segment(n_, nv_) = -bias;
  b_kkt_.segment(n_+nv_, nh_) = -Jh_dotV;
  b_kkt_.segment(n_+nv_+nh_, nc_active_) = -Jc_dotV_active;

  auto pinv_factorization = (M_kkt_.transpose() * M_kkt_).ldlt();

  for (int iter = 0; iter < max_iter_; ++iter) {
    q_rho_.segment(nv_, nu_) = -rho_ * (u_bar_ - mu_u_);
    q_rho_.segment(nv_+nu_+nh_, nc_) = -rho_ * (lambda_c_bar_ - mu_lambda_c_);
    b_kkt_.head(n_) = -(b + q_rho_);
    kkt_sol_ = pinv_factorization.solve(M_kkt_.transpose() * b_kkt_);
    z_ = kkt_sol_.head(n_);

    u_bar_ = project_to_input_bounds(
        z_.segment(nv_, nu_) + mu_u_, u_lb, u_ub);

    lambda_c_bar_ = project_to_friction_cone(
        z_.tail(nc_) + mu_lambda_c_, friction_coeffs);

    u_res_ = z_.segment(nv_, nu_) - u_bar_;
    lambda_c_res_ = z_.tail(nc_) - lambda_c_bar_;
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