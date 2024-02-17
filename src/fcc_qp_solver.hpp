#include <Eigen/Dense>
#include <tuple>

namespace fcc_qp {

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::Ref;

using std::vector;
using std::unordered_map;


struct FCCQPSolverDetails {
  int n_iter{};
  double eps_bounds{};
  double eps_friction_cone{};
  double solve_time{};
};

struct FCCQPSolution {
  FCCQPSolverDetails details{};
  VectorXd z;
};

class FCCQPSolver {
 public:

  FCCQPSolver(int num_vars, int num_equality_constraints, int nc,
              int lambda_c_start);

  void set_rho(double rho) {
    assert(rho > 0);
    rho_ = rho;
    P_rho_ = rho_ * MatrixXd::Identity(n_vars_, n_vars_);
  }

  void set_max_iter(int n) {
    assert(n > 0);
    max_iter_ = n;
  }

  void set_eps(double eps) {
    assert(eps > 0);
    eps_ = eps;
  }

  /*!
   * Solves the QP associated with the problem data
   * @param A_eq linear equalityu constraints includeing dynamics,
   * holonomic, and contact constraints
   * @param Q hessian of the cost on the stacked decision variables
   * @param q linear term of the cost on the stacked decision variables
   * @param friction_coeffs friction coefficients for the contact forces
   * @param lb lower bounds on the variables. must be -infinity for contact
   * force variables (friction cone constraint wil enforce positivity of
   * normal component)
   * @param ub upper bounds on the variables. Must be infinity for contact
   * forces.
   */
  void Solve(const Ref<const MatrixXd>& Q, const Ref<const VectorXd>& b,
             const Ref<const MatrixXd>& A_eq, const Ref<const VectorXd>& b_eq,
             const vector<double>& friction_coeffs,
             const Ref<const VectorXd>& lb, const Ref<const VectorXd>& ub);

  FCCQPSolution GetSolution() const;

  using Details = FCCQPSolverDetails;

 private:

  double rho_ = 10;
  double eps_ = 1e-10;
  int max_iter_ = 1000;

  const int n_vars_;
  const int n_eq_;
  const int nc_;
  const int lambda_c_start_;

  // Solver workspace variables
  MatrixXd P_rho_; // Hessian of augmented lagrangian term
  VectorXd q_rho_; // augmented lagrangian cost linear term

  MatrixXd M_kkt_;     // KKT matrix for admm update
  MatrixXd M_kkt_pre_; // KKT matrix for presolve equality constrained qp
  VectorXd b_kkt_;     // KKT right hand side

  // Variables
  VectorXd kkt_sol_;      // primal and equality duals from stage 1 primal solve
  VectorXd z_;            // stacked variables z = [dv, u, lambda_h, lambda_c]
  VectorXd z_bar_;        // ADMM copy of z
  VectorXd lambda_c_bar_; // ADMM copy of lambda_c
  VectorXd mu_z_;         // Dual for z = z_bar constraint
  VectorXd mu_lambda_c_;  // Dual for lambda_c = lambda_c_bar constraint

  // Decompositions
  Eigen::CompleteOrthogonalDecomposition<MatrixXd> M_kkt_factorization_;
  Eigen::CompleteOrthogonalDecomposition<MatrixXd> M_kkt_pre_factorization_;


  // residuals
  VectorXd z_res_;
  VectorXd lambda_c_res_;
  double z_res_norm_{};
  double lambda_c_res_norm_{};

  // solve info
  int n_iter_{};
  double solve_time_{};
};

}