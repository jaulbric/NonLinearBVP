# NonLinearBVP
A set of C++ routines for the numerical solution of nonlinear boundary values problems.

Requirements:
1. C++17
2. Boost
3. Eigen 3.4

Similar to MatLab's bvp4c and Scipy's solve_bvp. There is a 4th order method and a 6th order method, accessed as static methods of the bvp4c and bvp6c class. The routines use a Powell hybrid dogleg method to approximately solve monoimplicit Runge Kutta formula, controlling the residual (and in the case of bvp6c, the true error) of a continuous extension of the MIRK formula.

bvp4c indirectly solves a 3-point Lobatto IIIA formula, although the internal point is approximated using a cubic spline interpolator. bvp6c directly solves a 4-point Lobatto IIIA formula, using 3 internal points (one at the interval midpoint and the other two the Lobatto IIIA internal points).
