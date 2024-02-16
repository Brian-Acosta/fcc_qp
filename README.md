# Custom ADMM based solver for Friction Cone Constrained QPs
## Overview
Optimization based control of legged robots often leads to the formation of reactive controllers in the form of Quadratic Programs. 

```math
\displaystyle u = \arg \min_{z = [\dot{v}, u, \lambda_{h}, \lambda_{c}}
```
