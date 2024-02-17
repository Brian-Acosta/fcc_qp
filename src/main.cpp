#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "fcc_qp_solver.hpp"


namespace py = pybind11;

using namespace Eigen;
using namespace std;
using fcc_qp::FCCQPSolver;
using fcc_qp::FCCQPSolution;
using fcc_qp::FCCQPSolverDetails;

PYBIND11_MODULE(fcc_qp_solver, m) {
    m.doc() = "Python bindings for C++/Eigen LCQP solver for WBC";

    py::class_<FCCQPSolverDetails>(m, "FCCQPSolverDetails")
        .def_readwrite("n_iter", &FCCQPSolverDetails::n_iter)
        .def_readwrite("eps_bounds", &FCCQPSolverDetails::eps_bounds)
        .def_readwrite("eps_friction_cone",&FCCQPSolverDetails::eps_friction_cone)
        .def_readwrite("solve_time", &FCCQPSolverDetails::solve_time);

    py::class_<FCCQPSolution>(m, "FCCQPSolution")
        .def_readwrite("details", &FCCQPSolution::details)
        .def_readwrite("z", &FCCQPSolution::z);

    py::class_<FCCQPSolver>(m, "FCCQPSolver")
        .def(py::init<int,int,int,int>(),
            py::arg("num_vars"), py::arg("num_equality_constraints"),
            py::arg("nc"), py::arg("lambda_c_start"))
        .def("set_rho", &FCCQPSolver::set_rho)
        .def("set_max_iter", &FCCQPSolver::set_max_iter)
        .def("set_eps", &FCCQPSolver::set_eps)
        .def("Solve",
             &FCCQPSolver::Solve,
             py::arg("Q"), py::arg("b"), py::arg("A_eq"), py::arg("b_eq"),
             py::arg("friction_coeffs"), py::arg("lb"), py::arg("ub"))
        .def("GetSolution", &FCCQPSolver::GetSolution);

}
