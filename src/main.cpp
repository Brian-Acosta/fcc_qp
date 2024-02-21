#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "fcc_qp.hpp"


namespace py = pybind11;

using namespace Eigen;
using namespace std;

using fcc_qp::FCCQP;
using fcc_qp::FCCQPSolution;
using fcc_qp::FCCQPDetails;

PYBIND11_MODULE(fcc_qp_solver, m) {
    m.doc() = "Python bindings for C++/Eigen LCQP solver for WBC";

    py::class_<FCCQPDetails>(m, "FCCQPDetails")
        .def_readwrite("n_iter", &FCCQPDetails::n_iter)
        .def_readwrite("eps_bounds", &FCCQPDetails::eps_bounds)
        .def_readwrite("eps_friction_cone",&FCCQPDetails::eps_friction_cone)
        .def_readwrite("bounds_viol", &FCCQPDetails::bounds_viol)
        .def_readwrite("friction_cone_viol", &FCCQPDetails::friction_cone_viol)
        .def_readwrite("solve_time", &FCCQPDetails::solve_time)
        .def_readwrite("factorization_time", &FCCQPDetails::factorization_time);

    py::class_<FCCQPSolution>(m, "FCCQPSolution")
        .def_readwrite("details", &FCCQPSolution::details)
        .def_readwrite("z", &FCCQPSolution::z);

    py::class_<FCCQP>(m, "FCCQP")
        .def(py::init<int,int,int,int>(),
            py::arg("num_vars"), py::arg("num_equality_constraints"),
            py::arg("nc"), py::arg("lambda_c_start"))
        .def("set_rho", &FCCQP::set_rho)
        .def("set_max_iter", &FCCQP::set_max_iter)
        .def("set_eps", &FCCQP::set_eps)
        .def("Solve",
             &FCCQP::Solve,
             py::arg("Q"), py::arg("b"), py::arg("A_eq"), py::arg("b_eq"),
             py::arg("friction_coeffs"), py::arg("lb"), py::arg("ub"))
        .def("GetSolution", &FCCQP::GetSolution);

}
