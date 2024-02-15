#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include "conehead_solver.hpp"

#include <chrono>
#include <iostream>

using namespace std;
using namespace Eigen;
namespace py = pybind11;

using std::abs;
using std::sqrt;
using std::max;
