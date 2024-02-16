#include <iostream>
#include "fcc_qp_solver.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <Eigen/Dense>

#include <chrono>

#include <tuple>

namespace py = pybind11;

using namespace Eigen;
using namespace std;
using fcc_qp::FCCQPSolver;
using fcc_qp::FCCQPSolution;

PYBIND11_MODULE(fcc_qp_solver, m) {
    m.doc() = "Python bindings for C++/Eigen LCQP solver for WBC";

    py::class_<FCCQPSolution>(m, "FCCQPSolution")
        .def_readwrite("n_iter", &FCCQPSolution::n_iter)
        .def_readwrite("eps_bounds", &FCCQPSolution::eps_bounds)
        .def_readwrite("eps_friction_cone", &FCCQPSolution::eps_friction_cone)
        .def_readwrite("solve_time", &FCCQPSolution::solve_time)
        .def_readwrite("z", &FCCQPSolution::z);

    py::class_<FCCQPSolver>(m, "FCCQPSolver")
        .def(py::init<int,int,int,int>(),
            py::arg("num_vars"), py::arg("lambda_c_start"), py::arg("nc"),
            py::arg("num_equality_constraints"))
        .def("Solve",
             &FCCQPSolver::Solve,
             py::arg("Q"), py::arg("b"), py::arg("A_eq"), py::arg("b_eq"),
             py::arg("friction_coeffs"), py::arg("lb"), py::arg("ub"),
             py::arg("warm_start"))
        .def("GetSolution", &FCCQPSolver::GetSolution);

}
