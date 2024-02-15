# Solver for Lorentz Cone-Constrained Whole-Body-Control Quadratic Programs
## Overview
C++ solver (with python bindings) exclusively for solving instantaneous whole body control QPs of the form

```math
\displaystyle u = \arg \min_{z = [\dot{v}, u, \lambda_{h}, \lambda_{c}}
```