# NonLinearBVP
A set of C++ routines for the numerical solution of nonlinear boundary values problems.

Requirements:
1. C++17
2. Boost 1.79
3. Eigen 3.4

Similar to MatLab's bvp4c and Scipy's solve_bvp. There is a 4th order method and a 6th order method, accessed as static methods of the bvp4c and bvp6c class. The routines use a Powell hybrid dogleg method to approximately solve monoimplicit Runge Kutta formula, controlling the residual (and in the case of bvp6c, the true error) of a continuous extension of the MIRK formula.

bvp4c indirectly solves a 3-point Lobatto IIIA formula, although the internal point is approximated using a cubic spline interpolator. bvp6c directly solves a 5-point Lobatto IIIA formula, using 3 internal points (one at the interval midpoint and the other two the Lobatto IIIA internal points).

## Installation

There is no installation required, these are only header files. One simply needs to include `NonLinearBVP/collocation.hpp` in a driver file to use this library.

The linear algebra routines rely on Eigen, which contains a great number of runtime assertions. Once you are sure that your code is correct you should compile with the flag `-DNDEBUG` to remove these assertions, which will greatly reduces runtimes.

Setting optimization flags to `-O3` will also reduce runtimes substantially. Additional compliler options, such as `-march=native` and `-fopenmp` may also speed things up.

## Usage

The algorithms attempt to solve the differential system

$$
\frac{\mathrm{d} y}{\mathrm{d} x} = f \left( x, y, p \right) ,
$$

or in the case that a singular term is present

$$
\frac{\mathrm{d} y}{\mathrm{d} x} = \frac{1}{x - a} S y + f \left( x, y, p \right) ,
$$

on the interval $I = [a, b]$ with boundary conditions. The user must provide the algorithms with an initial mesh $x$, an initial guess of the solution on the mesh $y$, an initial guess for the unknown parameters $p$ (if present), the RHS derivative function $f$, the boundary conditions, and the singular matrix $S$ (if present). The calling syntax is
```c++
using nonlinearBVP::collocation::bvp6c;
auto sol = bvp6c<Scalar, RowsAtCompileTime, ParamsAtCompileTime>::solve(fun, bc, fun_jac, bc_jac, x, y, p, S, a_tol, r_tol, bc_tol, max_nodes);
```

| Template Parameter | Type | Description |
| ------------------ | ---- | ----------- |
|      `Scalar`      | floating point | A scalar floating point type such as `float`, `double`, or `long double`.[^1] |
| `RowsAtCompileTime` | integer | An integer (or convertible to `int`) value that describes the size of the system. Can be `Eigen::Dynamic` if not known at compile time. |
| `ParamsAtCompileTime` | integer | An integer (or convertible to `int`) value that describes the number of unknown parameters. Should be `0` if there are no unknown parameters. Can be `Eigen::Dynamic` if not known at compile time. |

In the following table we use `n` as the runtime number of rows of the differential system (is equal to `RowsAtCompileTime` if not `Dynamic`), `m` as the runtime number of columns (i.e. the number of mesh nodes), and `k` as the runtime number of unknown parameters (is equal to `ParamsAtCompileTime` if not `Dynamic`).

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| `fun` | callable | A callable object which evaluates the RHS of the differential equation, with calling syntax `fun(x, y, p, dydx)` or `fun(x, y, dydx)` if no unknown parameters are present.[^2] <ul><li> `x` : `const Array` with size `(m, 1)`, mesh nodes </li><li> `y` : `const Array` with size `(n, m)`, solution at mesh nodes</li><li> `p` : `const Array` with size `(k, 1)`, unknown parameters</li><li>`dydx` : `Array` with size `(n, m)`, evaluated RHS of differential equation</li></ul>|
| `bc`  | callable  | A callable object which evaluates the boundary conditions with calling syntax `bc(ya, yb, p, bc_res)` or `bc(ya, yb, bc_res)` if no unknown parameters are present.[^3] <ul><li>`ya` : `const Array` with size `(n, 1)`, solution at left boundary</li><li>`yb` : `const Array` with size `(n, 1)`, solution at right boundary</li><li>`p` : `const Array` with size `(k, 1)`, unknown parameters</li><li>`bc_res` : `Array` with size `(n + k, 1)`, evaluated boundary conditions</li></ul> |
| `fun_jac` | callable (optional) | A callable object which evaluates the Jacobian of the RHS of differential system with calling syntax `fun_jac(x, y, p, dfdy, dfdp)` or `fun_jac(x, y, dfdy)` if no unknown parameters are present. Will be estimate with forward finite differences if not supplied.[^4] <ul><li>`x` : `Array` with size `(m, 1)`, mesh nodes</li><li>`y` : `const Array` with size `(n, m)`, solution at mesh nodes</li><li>`p` : `const Array` with size `(k, 1)`, unknown parameters</li><li>`dfdy` : `Tensor` with size `(n, n, m)`, partial derivative of `fun` with respect to solution `y` with components `dfdy(i, j, l)` equal to $\frac{\partial f_{i l}}{\partial y_{j l}}$</li><li>`dfdp` : `Tensor` with size `(n, k, m)`, partial derivative of `fun` with respect to unknown parameters `p` with components `dfdp(i, j, l)` equal to $\frac{\partial f_{i l}}{\partial p_{j}}$</li></ul>|
| `bc_jac` | callable (optional) | A callable object which evaluates the Jacobian of the boundary conditions with calling syntax `bc_jac(ya, yb, p, dbc_dya, dbc_dyb, dbc_dp)` or `bc_jac(ya, yb, dbc_dya, dbc_yba)` if no uknown parameters are present. Will be estimated using forward finite differences if not supplied.[^5] <ul><li>`ya` : `const Array` with size `(n, 1)`, solution at left boundary</li><li>`yb` : `const Array` with size `(n, 1)`, solution at right boundary</li><li>`p` : `const Array` with size `(k, 1)`, unknown parameters</li><li>`dbc_dya` : `Array` with size `(n + k, n)`, partial derivative of boundary conditions with respect to `ya` with components `dbc_dya(i, j)` equal to $\frac{\partial bc_{i}}{\partial ya_{j}}$</li><li>`dbc_dyb` : `Array` with size `(n + k, n)`, derivative of boundary conditions with respect to `yb` with components `dbc_dyb(i, j)` equal to $\frac{\partial bc_{i}}{\partial yb_{j}}$</li><li>`dbc_dp` : `Array` with size `(n + k, k)`, partial derivative of `bc` with respect to `p` with components `dbc_dp(i, j)` equal to $\frac{\partial bc_{i}}{\partial p_{j}}$</li></ul>|
| `x` | `Array`  | Initial mesh nodes. The elements of `x` should be strictly increasing and should have a runtime size of `(m, 1)`. |
| `y` | `Array`  | Initial guess of solution at mesh nodes. Should have runtime size of `(n, m)` (where here `m` must match the initial mesh nodes). |
| `p` | `Array` (optional) | Initial guess of the unknown parameters. Should have runtime size of `(k, 1)`. If `ParamsAtCompileTime == 0` `p` should not be supplied (it will issue an assertion error).  |
| `S` | `Matrix` (optional) | A matrix that characterizes the singular term. Should have runtime size `(n, n)`. If not supplied it is assumed that there is no singular term. The singular term is handeled internally, and separately for `fun`, by enforcing the boundary condition $S y(0) = 0$. |
| `a_tol` | floating point | Absolute error tolerance. Must be same type as `Scalar` |
| `r_tol` | floating point | Relative error tolerance. Must be same type as `Scalar` |
| `bc_tol`| floating point | Boundary condition tolerance. Must be same type as `Scalar`. This just gives more control of over how precisely the boundary conditions are satisfied. A good choice is simply to let `bc_tol = r_tol`. |
| `max_nodes` | integer | The maximum allowable number of mesh nodes. The algorithm will quit if the number of mesh nodes exceeds this value. |

The calling syntax for `bvp4c` is identical to that of `bvp6c`. The return value `sol` is a `BVP_Result` object with the following members

| Member | Type | Description |
| ------ | ---- | ----------- |
| `sol` | `MIRK6_interpolator` or `MIRK4_interpolator`  | An interpolator object which provides a $\mathrm{C}1$ continuous extension of the solution. It can be used as `Array<Scalar, RowsAtCompileTime, 1> y_interp = sol(x);`|
| `x` | `Array` | The final mesh nodes. |
| `y`  | `Array` | The found solution at the mesh nodes. |
| `yp` | `Array` | Derivatives of the found solution at the mesh nodes. |
| `p` | `Array` | Found parameters. |
| `residuals` | `Array` | The residuals of the solution in each mesh interval. |
| `niter` | `int` | The number of iterations required for the solution to converge. |
| `status` | `int` | An integer which indicates the reason that the algorithm stopped. <ol><li>The algorithm successfully converged</li><li>The maximum number of mesh nodes was exceeded</li><li>A singular Jacobian was encountered</li><li>The algorithm was not able to satisfy the desired tolerances after 10 interations</li><li>The algorithm did not make progress on reducing the residual</li></ol> |
| `message` | `std::string` | A human readable message detailing the reason the algorithm stopped. |
| `success` | `bool` | `true` if the algorithm successfully converged, otherwise `false`. |

## Details

The nonlinear boundary value problem is linearized by using Lobatto IIIA quadrature routines on each mesh interval:

$$
\int^{x_{i + 1}}_{x_{i}} \frac{\mathrm{d} y}{\mathrm{d} x} \mathrm{d} x = y_{i + 1} - y_{i} \sim h_{i} \sum^{s}_{j = 1} a_{j} F\left(x_{i} + c_{j} h_{i}, y^{(j)}, p\right), \quad (h_{i} \to 0),
$$

where $F(x, y, p)$ is a function that returns the derivative $y'(x)$ at the point $x$. The solution values $y^{(1)} = y_{i}$ and $y^{(s)} = y_{i + 1}$ correspond to the solutions at the mesh nodes, while the $y^{(j)}$, with $1 < j < s$ are the solution at points internal to the mesh nodes (i.e. $c_{1} = 0$ and $c_{s} = 1$). The internal points are determined explicity from the solution values at the mesh nodes by building an interpolator with the appropriate order.

If $y_{i}$ is an $n$ dimensional vector at $m$ mesh node points, and there are $k$ unknown parameters, we then seek the solution of $n m + k$ unknown variables.

The algorithms then attempt to solve $n (m - 1)$ root equations

$$
y_{i + 1} - y_{i} \sim h_{i} \sum^{s}_{j = 1} a_{j} F \left( x_{i} + c_{j} h_{i}, y^{(j)}, p \right)
$$

along with $n + k$ boundary contitions

$$
g(y_{0}, y_{m}, p) = 0,
$$

implicitly for the solution values $y_{i}$ and unknown parameters $p$. The found solution is not in fact the solution at the mesh nodes, but a $\mathrm{C}1$ continuous extension $S(x)$, such that

$$
S'(x) = F(x, S(x), p) + r(x),
$$

where $r(x)$ is the *residual* of the continuous extension. The algorithms successfully converge if this residual is uniformly small across the mesh. In `bvp4c` we use the $L^{2}$ norm of the residual, scaled by the solution derivatives, to estimate the size of the residual on each interval. In `bvp6c` we estimate the $L^{\infty}$ norm of the scaled residual $h_{i} \vert\vert r(x) \vert\vert_{\infty}$ ($x_{i} < x < x_{i + 1}$).

The algorithms exit with a successfull convergence if

$$
r_{\mathrm{scaled}} \leq a_{\mathrm{tol}} + r_{\mathrm{tol}} \vert f \vert,
$$

in the case of `bvp4c`, or

$$
r_{\mathrm{scaled}} \leq a_{\mathrm{tol}} + r_{\mathrm{tol}} \vert y \vert,
$$

in the case of `bvp6c`, on each mesh interval.

Besides the fact that the two algorithms are of different orders (the global error in `bvp4c` is $O(h^{4})$ while in `bvp6c` it is $O(h^{6})$ ), the main difference between the two algorithms is that in `bvp4c`, although the error should be small if the scaled residual is also small, the global error is not directly controlled. In `bvp6c` we use a superconvergence result to bound the global error by the scaled residual, so the global error is directly controlled. Care should be taken when using `bvp4c` because the scaled residual can not be interpreted as a measure of the error.

After each iteration the mesh is redistributed. If the scaled residual exceeds the tolerance in any mesh interval a node is added at the midpoint (or two nodes are added, splitting the interval into thirds, if the scaled residual is very large). If the scaled residual is less than the tolerances than the algorithms attempt to remove a mesh node by estimating the new scaled residuals with the mesh node removed. The mesh node is removed if the estimate of the new scaled residuals are less than half the required tolerance. The removal of a mesh node is only done if we can merged three adjacent mesh intervals into two mesh intervals, essentially replacing the two internal mesh nodes with one and placing the new mesh node at an estimate of the optimum position.

## Example

As an example we will solve Bessel's equation on the interval $I = [0, 1]$. Bessel's equation with order $\nu = 0$ is

$$- u'' - \frac{1}{r} u' + u = p u.$$

The general solution is

$$u(r) = a J_{0}(\sqrt{p - 1} r) + b Y_{0}(\sqrt{p - 1} r).$$

We first need to transform this into a first order system with

$$y_{1}(r) = u(r), \quad y_{2}(r) = \frac{\mathrm{d} u(r)}{\mathrm{d} r}, \quad y(r) = \left(\begin{array}{c} y_{1}(r) \\ y_{2}(r) \end{array}\right).$$

The first order system is then

$$
\frac{\mathrm{d} y}{\mathrm{d} r} = \left(\begin{array}{c} y_{2} \\ - \frac{1}{r} y_{2} + \left(1 - p\right) y_{1} \end{array} \right)
$$

For boundary conditions we choose $y_{1}(1) = 0$ and $y_{2}(0) = 0$. These boundary conditions require $b = 0$ and $\sqrt{p - 1} = j_{0, k}$, where $j_{\nu, k}$ is the $k^{\text{th}}$ zero of the Bessel function of the first kind. The arbitrary constant $a$ cannot yet be determined, so we need to include another boundary condition, which we arbitrary choose to be $y_{1}(0) = 1$.

The derivatives, boundary conditions, and jacobians should be functors with an `operator()` method. For our problem the derivative functor can be written as
```c++
struct Bessel_fun {
  template <typename Derived1, typename Derived2, typename Derived3, typename Derived4>
  void operator()(const ArrayBase<Derived1>& r,
                  const ArrayBase<Derived2>& y,
                  const ArrayBase<Derived3>& p,
                  const ArrayBase<Derived4>& dydr) const {
    const_cast<ArrayBase<Derived4>&>(dydr).row(0) = y.row(1);
    const_cast<ArrayBase<Derived4>&>(dydr).row(1) = (1 - p.value()) * y.row(0);
  }
};
```
Note that we have templated the `operator()` method so that it accepts any `const ArrayBase<Derived>&` types. This avoids unnecessary temporaries and copies since our method will be overloaded by the compiler for Eigen expression templates as well as regular `Array` types. A template parameter is required for every input, since the derived type may not be the same for every input. Unfortunately this requires us to use `const` references, necessitating the use of `const_cast` inside the function body. This is mildly inconvenient, but the gains in performance are worth it.

Before proceeding one should also note that the singular term is completely ignored in the `Bessel_fun` structure. We will discuss this term in more detail in a moment.

Next we implement the boundary conditions, also as a functor:
```c++
struct Bessel_bc {
  template <typename Derived1, typename Derived2, typename Derived3, typename Derived4>
  void operator()(const ArrayBase<Derived1>& ya,
                  const ArrayBase<Derived2>& yb,
                  const ArrayBase<Derived3>& p,
                  const ArrayBase<Derived4>& bc_res) const {
    const_cast<ArrayBase<Derived4>&>(bc_res)(0) = ya(0) - 1;
    const_cast<ArrayBase<Derived4>&>(bc_res)(1) = yb(0);
    const_cast<ArrayBase<Derived4>&>(bc_res)(2) = ya(1);
  }
};
```
with the same notes about using Eigen expression templates. Similarly for the Jacobian (we use `Eigen::Index` in the `for` loop since it automatically match the Eigen defaul, usually `long int`)
```c++
struct Bessel_fun_jac {
  template <typename Derived1, typename Derived2, typename Derived3>
  void operator()(const ArrayBase<Derived1>& r,
                  const ArrayBase<Derived2>& y,
                  const ArrayBase<Derived3>& p,
                  Tensor<typename Derived2::Scalar, 3>& dfdy,
                  Tensor<typename Derived2::Scalar, 3>& dfdp) const {
    for (Index idx = 0; idx < r.size(); ++idx) {
      dfdy(0, 0, idx) = 0;
      dfdy(0, 1, idx) = 1;
      dfdy(1, 0, idx) = 1 - p.value();
      dfdy(1, 1, idx) = 0;
      dfdp(0, 0, idx) = 0;
      dfdp(1, 0, idx) = - y(0, idx);
    }
  }
};
```
The data for the Jacobians is contained in `Eigen::Tensor` objects. The `Tensor` module of Eigen is currently unsupported, but will probably be fully supported in the near future. One issue is that the `data()` member of `TensorBase<Derived>` is `protected`, so the algorithm has no way of directly accessing the pointer to the underlying data array. We could have created a new class that inherits from `TensorBase<Derived>` and then exposed the `data()` member through another public method, but internally the algorithms always input plain `Tensor` objects, so there wouldn't be any performance gain anyway. This isn't the most efficient way to set the values of the Jacobian either, one should probably set the tensor components using the `slice()` method, but we leave it to the user to investigate more efficient methods.

The boundary conditions Jacobian is also a functor
```c++
struct Bessel_bc_jac {
  template <typename Derived1, typename Derived2, typename Derived3, typename Derived4, typename Derived5, typename Derived6>
  void operator()(const ArrayBase<Derived1>& ya,
                  const ArrayBase<Derived2>& yb,
                  const ArrayBase<Derived3>& p,
                  const ArrayBase<Derived4>& dbc_dya,
                  const ArrayBase<Derived5>& dbc_dyb,
                  const ArrayBase<Derived6>& dbc_dp) {
    const_cast<ArrayBase<Derived4>&>(dbc_dya) = Array<typename Derived4::Scalar, 3, 2>({{1, 0}, {0, 0}, {0, 1}});
    const_cast<ArrayBase<Derived5>&>(dbc_dyb) = Array<typename Derived5::Scalar, 3, 2>({{0, 0}, {1, 0}, {0, 0}});
    const_cast<ArrayBase<Derived6>&>(dbc_dp) = Array<typename Derived6::Scalar, 3, 1>::Zero();
  }
};
```
Our choice of boundary conditions don't depend on the eigenvalue $p$ at all, so the partial derivate with respect to $p$ is just set to 0.

Now we come to the singular term. In the first order system we have

$$
\frac{1}{r} \left(\begin{array}{cc} 0 & 0 \\ 0 & - 1 \end{array} \right) \left( \begin{array}{c} y_{1}(r) \\ y_{2}(r) \end{array} \right) = - \frac{1}{r} \left(\begin{array}{c} 0 \\ y_{2}(r) \end{array} \right).
$$

Thus, the singular term is defined by the matrix

$$
S = \left(\begin{array}{cc} 0 & 0 \\ 0 & -1 \end{array} \right).
$$

In order for the solution to be regular at the origin (which we assume will always be the case) we must have $S y(0) = 0$, therefore

$$
\lim_{r \to 0} \frac{1}{r} S y(r) = S y'(0).
$$

Internally the algorithms treat the above as an additional boundary condition, and adjust the solution so that this boundary condition is always exactly satisfied. All the user needs to do is to input the matrix $S$:
```c++
Matrix<double, 2, 2> S;
S << 0, 0, 0, -1;
```

The first order system is now completely done. The next step is to construct an initial guess and let the algorithms find the solution. We choose an initial mesh with 10 points (the algorithms will adjust this mesh appropriately so as to reduce the residuals)
```c++
Array<double, 10, 1> r = Array<double, 10, 1>::LinSpaced(10, 0, 1);
```
We will make a rather poor guess for the solution, simply to illustrate the effectiveness of the nonlinear boundary value problem solvers:
```c++
Array<Scalar, 1, 1> p_guess;
p_guess(0, 0) = 5;

Array<Scalar, 2, 10> y_guess = Array<Scalar, 2, 10>::Zero();
y_guess(0, 0) = 1;
y_guess(1, 9) = -1;
```
Even though there is only a single unknown parameter it must still be input as an `Array` type, since this is what the solver expects. We are now ready to solve the problem. We set the tolerances to about as small as one should reasonably be able to ask for (if `r_tol` is less than `100 * std::numeric_limits<Scalar>::epsilon()` the solver will increase it to this minimum value). We also set a maximum number of nodes of 1000. This number can be as large as you want, but the linear algebra routines quickly become untenably slow if the number of nodes gets very large.
```c++
Bessel_fun fun;
Bessel_bc bc;
Bessel_fun_jac fun_jac;
Bessel_bc_jac bc_jac;

Scalar a_tol = std::numeric_limits<double>::epsilon();
Scalar r_tol = 100 * std::numeric_limits<double>::epsilon();
Scalar bc_tol = r_tol;

auto sol = nonlinearbvp::collocation::bvp6c<double, 2, 1>::solve(fun, bc, fun_jac, bc_jac, r, y_guess, p_guess, S, a_tol, r_tol, bc_tol, 1000);
```

Finally, we print the results. We use boost to compare to the exact values.
```c++
double p_exact = boost::math::cyl_bessel_j_zero(0.0, 1) * boost::math::cyl_bessel_j_zero(0.0, 1) + 1;

Array<double, 2, Eigen::Dynamic> y_exact(2, sol.x.size());
for (Index idx = 0; idx < sol.x.size(); ++idx) {
  y_exact(0, idx) = boost::math::cyl_bessel_j(0, std::sqrt(p_exact - 1) * sol.x(idx));
  y_exact(1, idx) = - std::sqrt(p_exact - 1) * boost::math::cyl_bessel_j(1, sqrt(p_exact - 1) * sol.x(idx));
}

// Print the eigenvalue
std::cout << "Found eigenvalue  : " << std::setprecision(std::numeric_limits<double>::max_digits10) << sol.p << std::endl;
std::cout << "Exact eigenvalue  : " << p_exact << std::endl;

// Print the maximum error
Array<double, 2, Eigen::Dynamic> err(2, sol.x.size());
err = y_exact - sol.y;
std::cout << "Max absolute error: " << err.maxCoeff() << std::endl;
```
This prints
```
Found eigenvalue  : 6.783185962946785
Exact eigenvalue  : 6.783185962946785
Max absolute error: 4.163336342344337e-16
```
On a 64-bit MacBook Air with four 1.30Ghz Intel Core i5-42500 CPUs it takes 7 milliseconds for the algorithms to converge to the above tolerances.

One final note is that the collocation algorithms can also be made to output their progress to varying degrees by defining the macro `COLLOCATION_VERBOSITY` to be in integer from 0 to 2 before including the NonLinearBVP header files. 0 (default) prints nothing, 1 prints the final results, and 2 will print progress updates during each iteration.

[^1]: User defined types can also be used, such as `boost::multiprecision::float128`. One needs to configure Eigen in order to be able to use such types, but for `boost::multiprecision` types this is done with a simple include: `#include <boost/multiprecision/eigen.hpp>`.

[^2]: The calling syntax can be changed to `dydx = fun(x, y, p)` (or `dydx = fun(x, y)` if no unknown parameters are present) by defining the macro `#define COLLOCATION_FUNCTION_RETURN` before including `collocation.hpp`.

[^3]: The calling syntax can be changed to `bc_res = bc(ya, yb, p)` (or `bc_res = bc(ya, yb)` if no unknown parameters are present) by defining the macro `#define COLLOCATION_BC_RETURN` before including `collocation.hpp`.

[^4]: The calling syntax can be changed to `std::tie(dfdy, dfdp) = fun_jac(x, y, p)` (or `dfdy = fun_jac(x, y)` if no unknown parameters are present) by defining the macro `#define COLLOCATION_FUNCTION_JACOBIAN_RETURN` before including `collocation.hpp`.

[^5]: The calling syntax can be changed to `std::tie(dbc_dya, dbc_dyb, dbc_dp) = bc_jac(ya, yb, p)` (or `std::tie(dbc_dya, dbc_dyb) = bc_jac(ya, yb, p)` if no unknown parameters are present) by defining the macro `#define COLLOCATION_BC_JACOBIAN_RETURN` before including `collocation.hpp`.
