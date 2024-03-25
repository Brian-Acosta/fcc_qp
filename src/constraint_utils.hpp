#pragma once
#include <vector>
#include <Eigen/Dense>

namespace fcc_qp {

static Eigen::Vector3d project_to_friction_cone(const Eigen::Vector3d &f, double mu) {
  double norm_fxy = f.head<2>().norm();

  // inside the friction cone, do nothing
  if (mu * f(2) >= norm_fxy) {
    return f;
  }

  // More than 90 degrees from the side of the cone, closest point is the origin
  if (f(2) < -mu * norm_fxy) {
    return Eigen::Vector3d::Zero();
  }

  // project up to the side of the friction cone
  double xy_ratio = mu * f(2) / norm_fxy;
  Eigen::Vector3d cone_ray(xy_ratio * f(0), xy_ratio * f(1), f(2));
  cone_ray.normalize();
  return cone_ray.dot(f) * cone_ray;

}

static Eigen::VectorXd project_to_friction_cone(
    const Eigen::VectorXd &f, const std::vector<double> &friction_coeffs) {
  Eigen::VectorXd out = Eigen::VectorXd::Zero(f.rows());
  for (int i = 0; i < f.rows() / 3; ++i) {
    out.segment<3>(3 * i) = project_to_friction_cone(
        f.segment<3>(3 * i), friction_coeffs.at(i));
  }
  return out;
}

static Eigen::MatrixX3d get_active_ray_constraint_matrix(const Eigen::Vector3d &f, double mu) {
  Eigen::Vector3d cone_ray = project_to_friction_cone(f, mu);
  if (cone_ray.norm() == 0) {
    return Eigen::MatrixX3d::Identity(3,3);
  }
  cone_ray.normalize();
  Eigen::MatrixX3d M = Eigen::MatrixX3d::Zero(2, 3);
  M.row(0)(0) = cone_ray(1);
  M.row(0)(1) = -cone_ray(0);
  M.row(1) = M.row(0).transpose().cross(cone_ray).transpose();

  return M;
}

static Eigen::VectorXd project_to_bounds(
    const Eigen::VectorXd &x, const Eigen::VectorXd &lb, const Eigen::VectorXd &ub) {
  Eigen::VectorXd out = x;
  for (int i = 0; i < x.rows(); ++i) {
    out(i) = std::max(std::min(x(i), ub(i)), lb(i));
  }
  return out;
}

static double calc_friction_cone_violation(
    const Eigen::VectorXd &f, const std::vector<double> &friction_coeffs) {
  double violation = 0;

  for (int i = 0; i < f.rows() / 3; ++i) {
    int start = 3 * i;
    double fz = f(start + 2);
    double mu = friction_coeffs.at(i);
    violation += std::max(0., f.segment<2>(start).norm() - mu * fz);
  }
  return violation;
}

static std::vector<int> guess_active_friction_cone_constraints(
    const Eigen::VectorXd &f, const std::vector<double> &friction_coeffs) {
  std::vector<int> indices;
  for (int i = 0; i < f.rows() / 3; ++i) {
    int start = 3 * i;
    double fz = f(start + 2);
    double mu = friction_coeffs.at(i);
    double violation = std::max(0., f.segment<2>(start).norm() - mu * fz);
    if (violation > 0) {
      indices.push_back(i);
    }
  }
  return indices;
}

static Eigen::MatrixXd get_active_set_friction_constraint(
    const Eigen::VectorXd &f, const std::vector<double> &friction_coeffs) {
  std::vector<int> active_indices = guess_active_friction_cone_constraints(
      f, friction_coeffs);

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(0, f.rows());
  for (int idx : active_indices) {
    int start_row = A.rows();
    Eigen::Vector3d lambda = f.segment<3>(idx * 3);
    Eigen::Matrix3Xd M = get_active_ray_constraint_matrix(
        lambda,
        friction_coeffs.at(idx)
    );
    A.conservativeResize(A.rows() + M.rows(), Eigen::NoChange);
    A.bottomRows(M.rows()).setZero();
    A.block(start_row, 3*idx, M.rows(), M.cols()) = M;
  }
  return A;
}

static bool validate_bounds(
    const Eigen::VectorXd& lb, const VectorXd& ub) {
  for (int i = 0; i < lb.rows(); ++i) {
    if (lb(i) > ub(i)) {
      return false;
    }
  }
  return true;
}

static std::pair<Eigen::MatrixXd, Eigen::VectorXd>
get_active_set_bounds_constraint(const Eigen::VectorXd& x,
                                 const Eigen::VectorXd& lb,
                                 const Eigen::VectorXd& ub) {
  std::vector<int> indices_viol{};
  std::vector<double> bounds_viol{};

  for (int i = 0; i < x.rows(); ++i) {
    if (x(i) < lb(i)) {
      indices_viol.push_back(i);
      bounds_viol.push_back(lb(i));
    } else if (x(i) > ub(i)) {
      indices_viol.push_back(i);
      bounds_viol.push_back(ub(i));
    }
  }
  MatrixXd A_viol = MatrixXd::Zero(indices_viol.size(), x.rows());
  VectorXd b_viol = VectorXd::Zero(indices_viol.size());
  for (int i = 0; i < indices_viol.size(); ++i) {
    A_viol(i, indices_viol.at(i)) = 1;
    b_viol(i) = bounds_viol.at(i);
  }

  return {A_viol, b_viol};
}

}