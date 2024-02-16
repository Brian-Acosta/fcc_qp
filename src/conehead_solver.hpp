#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <tuple>

namespace py = pybind11;

namespace conehead {

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::Ref;

using std::vector;
using std::unordered_map;

class ConeHeadSolver {
 public:

  ConeHeadSolver(int nv, int nu, int nh, int nc, int nc_active);

  void set_rho(double rho) {
    assert(rho > 0);
    rho_ = rho;
  }

  /*!
   * Solves the QP associated with the problem data
   * @param M mass matrix (nv x nv), positive definite
   * @param B input matrix (nv x nu)
   * @param Jh stacked holonomic constraint matrix (nh x nv, full rank)
   * @param Jc stacked contact constraint matrix (nc x nv, can have rows of zeros)
   * @param bias dynamics bias (constant) term
   * @param Jh_dotV holonomic constraint bias term
   * @param Jc_dotV contact constraint biases
   * @param Q hessian of the cost on the stacked decision variables
   * @param q linear term of the cost on the staked decision variables
   * @param c constant term in the cost.
   */
  void Solve(
      const Ref<MatrixXd>& M,
      const Ref<MatrixXd>& B,
      const Ref<MatrixXd>& Jh,
      const Ref<MatrixXd>& Jc,
      const Ref<MatrixXd>& Jc_active,
      const Ref<VectorXd>& bias,
      const Ref<VectorXd>& Jh_dotV,
      const Ref<VectorXd>& Jc_dotV_active,
      const Ref<MatrixXd>& Q, const Ref<VectorXd>& b,
      const vector<double>& friction_coeffs, const Ref<VectorXd>& u_lb,
      const Ref<VectorXd>& u_ub, bool warm_start);

 private:

  double rho_ = 10;
  double eps_ = 1e-5;
  int max_iter_ = 100;

  const int nv_;
  const int nu_;
  const int nh_;
  const int nc_;
  const int nc_active_;
  const int n_;

  // Solver workspace variables
  MatrixXd P_rho_; // Hessian of augmented lagrangian term
  VectorXd q_;     // cost function linear term
  VectorXd q_rho_; // augmented lagrangian cost linear term
  MatrixXd A_;     // stacked equality constraint matrix

  MatrixXd M_kkt_; // KKT matrix
  VectorXd b_kkt_; // KKT right hand side

  // Variables
  VectorXd kkt_sol_;      // primal and equality duals from stage 1 primal solve
  VectorXd z_;            // stacked variables z = [dv, u, lambda_h, lambda_c]
  VectorXd u_bar_;        // ADMM copy of u
  VectorXd lambda_c_bar_; // ADMM copy of lambda_c
  VectorXd mu_u_;         // Dual for u = u_bar constraint
  VectorXd mu_lambda_c_;  // Dual for lambda_c = lambda_c_bar constraint

  // residuals
  VectorXd u_res_;
  VectorXd lambda_c_res_;
  double u_res_norm_;
  double lambda_c_res_norm_;

  // solve info
  int n_iter_;
  double solve_time_;
};

}