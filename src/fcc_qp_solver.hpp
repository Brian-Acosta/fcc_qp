#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <tuple>

namespace py = pybind11;

namespace fcc_qp {

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::Ref;

using std::vector;
using std::unordered_map;

class FCCQPSolver {
 public:

  FCCQPSolver(int num_vars, int u_start, int nu, int lambda_c_start,
                 int nc, int num_equality_constraints);

  void set_rho(double rho) {
    assert(rho > 0);
    rho_ = rho;
  }

  /*!
   * Solves the QP associated with the problem data
   * @param A_eq linear equalityu constraints includeing dynamics,
   * holonomic, and contact constraints
   * @param Q hessian of the cost on the stacked decision variables
   * @param q linear term of the cost on the stacked decision variables
   */
  void Solve(const Ref<MatrixXd>& A_eq, const Ref<VectorXd>& b_eq,
             const Ref<MatrixXd>& Q, const Ref<VectorXd>& b,
             const vector<double>& friction_coeffs, const Ref<VectorXd>& u_lb,
             const Ref<VectorXd>& u_ub, bool warm_start);
 private:

  double rho_ = 10;
  double eps_ = 1e-5;
  int max_iter_ = 100;

  const int n_vars_;
  const int n_eq_;
  const int nu_;
  const int nc_;
  const int u_start_;
  const int lambda_c_start_;

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