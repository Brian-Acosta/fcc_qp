#pragma once
#include <Eigen/Dense>
#include <tuple>
#include <iostream>

namespace fcc_qp {

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::Ref;

using std::vector;

enum FCCQPSolveStatus {
  kSuccess = 0,
  kMaxIterations = 1
};

struct FCCQPDetails {
  int n_iter{};
  double admm_residual_bounds{};
  double admm_residual_friction_cone{};
  double solve_time{};
  double factorization_time{};
  double bounds_viol{};
  double friction_cone_viol{};
  FCCQPSolveStatus solve_status;
};

struct FCCQPOptions {
  int max_iter = 1000;
  double rho = 1e-6;
  double eps_fcone = 1e-3;
  double eps_bound = 1e-6;
};

struct FCCQPSolution {
  FCCQPDetails details{};
  VectorXd z;
};


/*!
  * FCCQP is a solver for the following convex LCQP, where Q is positive
  * semi-definite, and F is the set of friction cone constraints applied to a
  * subset of the decision variables.
  *
  * minimize     (1/2)xᵀQx + bᵀx
  * subject to   A_eq x = b_eq
  *              lb ≤ x ≤ ub
  *              x ∈ F
  *
  */
class FCCQP {
 public:

  /*!
   * Constructor for the QP solver
   * @param num_vars the total number of decision variables for the QP(s)
   * you will be solving
   * @param num_equality_constraints number of rows in A_eq
   * @param nc number of decision variables representing contact forces.
   * must be a multiple of 3.
   * @param lambda_c_start the index of the first contact force variable within
   * the vector of decision variables.
   *
   * Notice that the contact force variables are assumed to be in a contiguous
   * segment. In Eigen, this looks like
   *
   *   stacked_contact_forces = x.segment(lambda_c_start, nc)
   *
   */
  FCCQP(int num_vars, int num_equality_constraints, int nc, int lambda_c_start);

  void set_rho(double rho) {
    assert(rho > 0);
    options_.rho = rho;
  }

  void set_max_iter(int n) {
    assert(n > 0);
    options_.max_iter = n;
  }

  void set_options(FCCQPOptions opt) {
    options_ = opt;
  }

  void set_warm_start(bool warm_start) {
    warm_start_ = warm_start;
  }

 /*!
  *
  * Solves the QP for the given problem data
  *
  * @param A_eq linear equality constraints including dynamics,
  * holonomic, and contact constraints
  * @param Q hessian of the cost on the stacked decision variables
  * @param q linear term of the cost on the stacked decision variables
  * @param friction_coeffs friction coefficients for the contact forces
  * @param lb lower bounds on the variables. must be -infinity for contact
  * force variables (friction cone constraint wil enforce non-negativity of
  * the normal component)
  * @param ub upper bounds on the variables. Must be infinity for contact
  * forces.
  *
  * N.B. We have only tested for numerical stability of the solver when A_eq
  * is full row-rank, which is the case for our application. You may need to
  * switch the KKT factorization to a rank-revealing linear solver
  * (i.e. CompleteOrthogonalDecomposition) for low-rank A_eq.
  *
  */
  void Solve(const Ref<const MatrixXd>& Q, const Ref<const VectorXd>& b,
             const Ref<const MatrixXd>& A_eq, const Ref<const VectorXd>& b_eq,
             const vector<double>& friction_coeffs,
             const Ref<const VectorXd>& lb, const Ref<const VectorXd>& ub);

  FCCQPSolution GetSolution() const;

  int contact_vars_start() const {return lambda_c_start_;}

 private:

  void DoADMM(const Ref<const VectorXd>& b,
              const vector<double>& friction_coeffs,
              const Ref<const VectorXd>& lb, const Ref<const VectorXd>& ub);


  FCCQPOptions options_;

  const int n_vars_;
  const int nc_;
  const int lambda_c_start_;
  int n_eq_;

  bool warm_start_ = false;

  // Solver workspace variables
  MatrixXd P_rho_; // Hessian of augmented lagrangian term
  VectorXd q_rho_; // augmented lagrangian cost linear term

  MatrixXd M_kkt_;     // KKT matrix for admm update
  MatrixXd M_kkt_pre_; // KKT matrix for presolve equality constrained qp
  VectorXd b_kkt_;     // KKT right hand side

  // Variables
  VectorXd kkt_sol_;      // primal and equality duals from stage 1 primal solve
  VectorXd x_;            // stacked decision variables
  VectorXd x_bar_;        // ADMM copy of x
  VectorXd lambda_c_bar_; // ADMM copy of lambda_c
  VectorXd mu_x_;         // Dual for x = x_bar constraint
  VectorXd mu_lambda_c_;  // Dual for lambda_c = lambda_c_bar constraint

  // Decompositions
  Eigen::LDLT<MatrixXd> M_kkt_factorization_;
  Eigen::LDLT<MatrixXd> M_kkt_pre_factorization_;
  Eigen::CompleteOrthogonalDecomposition<MatrixXd> M_kkt_pre_factorization_backup_;

  // residuals
  VectorXd x_res_;
  VectorXd lambda_c_res_;
  double x_res_norm_{};
  double lambda_c_res_norm_{};
  double bounds_viol_{};
  double friction_cone_viol_{};

  // solve info
  int n_iter_{};
  double solve_time_{};
  double factorization_time_{};
};

}