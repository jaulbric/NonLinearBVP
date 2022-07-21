#include <Eigen/Dense>
#include <Eigen/CXX11/Tensor>
#include <boost/math/special_functions/bessel.hpp>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <chrono>

// #define COLLOCATION_VERBOSITY 2

#include "NonLinearBVP/collocation.hpp"

using Eigen::Array;
using Eigen::ArrayBase;
using Eigen::Matrix;
using Eigen::Tensor;
using Eigen::Index;
using std::sqrt;
using boost::math::cyl_bessel_j;
using boost::math::cyl_bessel_j_zero;

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

struct Bessel_fun {
  template <typename Derived1, typename Derived2, typename Derived3, typename Derived4>
  void operator()(const ArrayBase<Derived1>& r, const ArrayBase<Derived2>& y, const ArrayBase<Derived3>& p, const ArrayBase<Derived4>& dydr) const {
    const_cast<ArrayBase<Derived4>&>(dydr).row(0) = y.row(1);
    const_cast<ArrayBase<Derived4>&>(dydr).row(1) = (1 - p.value()) * y.row(0);
  }
};

struct Bessel_bc {
  template <typename Derived1, typename Derived2, typename Derived3, typename Derived4>
  void operator()(const ArrayBase<Derived1>& ya, const ArrayBase<Derived2>& yb, const ArrayBase<Derived3>& p, const ArrayBase<Derived4>& bc_res) const {
    const_cast<ArrayBase<Derived4>&>(bc_res)(0) = ya(0) - 1;
    const_cast<ArrayBase<Derived4>&>(bc_res)(1) = yb(0);
    const_cast<ArrayBase<Derived4>&>(bc_res)(2) = ya(1);
  }
};

struct Bessel_fun_jac {
  template <typename Derived1, typename Derived2, typename Derived3>
  void operator()(const ArrayBase<Derived1>& r, const ArrayBase<Derived2>& y, const ArrayBase<Derived3>& p, Tensor<typename Derived2::Scalar, 3>& dfdy, Tensor<typename Derived2::Scalar, 3>& dfdp) const {
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

struct Bessel_bc_jac {
  template <typename Derived1, typename Derived2, typename Derived3, typename Derived4, typename Derived5, typename Derived6>
  void operator()(const ArrayBase<Derived1>& ya, const ArrayBase<Derived2>& yb, const ArrayBase<Derived3>& p, const ArrayBase<Derived4>& dbc_dya, const ArrayBase<Derived5>& dbc_dyb, const ArrayBase<Derived6>& dbc_dp) {
    const_cast<ArrayBase<Derived4>&>(dbc_dya) = Array<typename Derived4::Scalar, 3, 2>({{1, 0}, {0, 0}, {0, 1}});
    const_cast<ArrayBase<Derived5>&>(dbc_dyb) = Array<typename Derived5::Scalar, 3, 2>({{0, 0}, {1, 0}, {0, 0}});
    const_cast<ArrayBase<Derived6>&>(dbc_dp) = Array<typename Derived6::Scalar, 3, 1>::Zero();
  }
};

int main() {
  using Scalar = double;

  Array<Scalar, 1, 1> p_guess;
  p_guess(0, 0) = 5;
  Scalar p_exact = cyl_bessel_j_zero(Scalar(0), 1) * cyl_bessel_j_zero(Scalar(0), 1) + 1;

  Array<Scalar, 10, 1> r = Array<Scalar, 10, 1>::LinSpaced(10, 0, 1); // Interval r = [0, 1]

  Array<Scalar, 2, 10> y_guess = Array<Scalar, 2, 10>::Zero();
  y_guess(0, 0) = 1;
  y_guess(1, 9) = -1;

  Bessel_fun fun;
  Bessel_bc bc;
  Bessel_fun_jac fun_jac;
  Bessel_bc_jac bc_jac;

  Matrix<Scalar, 2, 2> S;
  S << 0, 0, 0, -1;

  Scalar a_tol = std::numeric_limits<Scalar>::epsilon();
  Scalar r_tol = 100 * std::numeric_limits<Scalar>::epsilon();
  Scalar bc_tol = r_tol;

  auto start = high_resolution_clock::now();
  auto sol = nonlinearbvp::collocation::bvp6c<Scalar, 2, 1>::solve(fun, bc, fun_jac, bc_jac, r, y_guess, p_guess, S, a_tol, r_tol, bc_tol, 1000);
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);

  // Print the found eigenvalue
  std::cout << "Found eigenvalue: " << std::setprecision(std::numeric_limits<Scalar>::max_digits10) << sol.p << std::endl;
  std::cout << "Exact eigenvalue: " << p_exact << std::endl;

  // Print the maximum error
  Array<Scalar, 2, Eigen::Dynamic> y_exact(2, sol.x.size());
  for (Index idx = 0; idx < sol.x.size(); ++idx) {
    y_exact(0, idx) = cyl_bessel_j(0, sqrt(p_exact - 1) * sol.x(idx));
    y_exact(1, idx) = - sqrt(p_exact - 1) * cyl_bessel_j(1, sqrt(p_exact - 1) * sol.x(idx));
  }

  Array<Scalar, 2, Eigen::Dynamic> err(2, sol.x.size());
  err = (y_exact - sol.y).abs();
  std::cout << "Max absolute error: " << err.maxCoeff() << std::endl;

  std::cout << "Time to solve problem: " << duration.count() << " milliseconds." << std::endl;

  return 0;
}
