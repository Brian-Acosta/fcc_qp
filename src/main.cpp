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

PYBIND11_MODULE(fcc_qp_solver, m) {
    m.doc() = "Python bindings for C++/Eigen LCQP solver for WBC";
    py::class_<FCCQPSolver>(m, "Solver")
        .def(py::init<int,int,int,int,int,int>());

}
