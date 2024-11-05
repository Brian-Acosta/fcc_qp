#include "constraint_utils.hpp"

namespace fcc_qp {

Eigen::Vector3d project_to_friction_cone(
    const Eigen::Vector3d &f, double mu) {
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

Eigen::VectorXd project_to_friction_cone(
    const Eigen::VectorXd &f, const std::vector<double> &friction_coeffs) {
  Eigen::VectorXd out = Eigen::VectorXd::Zero(f.rows());
  for (int i = 0; i < f.rows() / 3; ++i) {
    out.segment<3>(3 * i) = project_to_friction_cone(
        f.segment<3>(3 * i), friction_coeffs.at(i));
  }
  return out;
}

Eigen::VectorXd project_to_bounds(
    const Eigen::VectorXd &x,
    const Eigen::VectorXd &lb,
    const Eigen::VectorXd &ub) {
  Eigen::VectorXd out = x;
  for (int i = 0; i < x.rows(); ++i) {
    out(i) = std::max(std::min(x(i), ub(i)), lb(i));
  }
  return out;
}

double calc_friction_cone_violation(
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

double calc_bound_violation(
    const VectorXd &x, const VectorXd &lb, const VectorXd &ub) {
  return (x - project_to_bounds(x, lb, ub)).norm();
}

bool validate_bounds(
    const Eigen::VectorXd &lb, const VectorXd &ub) {
  for (int i = 0; i < lb.rows(); ++i) {
    if (lb(i) > ub(i)) {
      return false;
    }
  }
  return true;
}

}
