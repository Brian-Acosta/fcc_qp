#pragma once
#include <vector>
#include <Eigen/Dense>

namespace fcc_qp {

Eigen::Vector3d project_to_friction_cone(const Eigen::Vector3d &f, double mu);

Eigen::VectorXd project_to_friction_cone(
    const Eigen::VectorXd &f, const std::vector<double> &friction_coeffs);

Eigen::VectorXd project_to_bounds(
    const Eigen::VectorXd &x, const Eigen::VectorXd &lb,
    const Eigen::VectorXd &ub);

double calc_friction_cone_violation(
    const Eigen::VectorXd &f, const std::vector<double> &friction_coeffs);

double calc_bound_violation(
    const VectorXd& x, const VectorXd& lb, const VectorXd& ub);

bool validate_bounds(
    const Eigen::VectorXd& lb, const VectorXd& ub);

}