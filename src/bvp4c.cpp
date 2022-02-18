#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <iostream>
#include <iomanip>
#include <limits>

#define COLLOCATION_VERBOSITY 2

#include "collocation.hpp"
#include "TestProblems.hpp"
#include "constants/constants.hpp"

using Eigen::Array;
using Eigen::ArrayBase;
using Eigen::Matrix;
using Eigen::Index;
using std::sqrt;
using boost::math::constants::pi;
using boost::math::constants::half;
using boost::math::constants::quarter;
using boost::math::cyl_bessel_j;

int main() {
  using Real = long double;

  Real m = 1;
  Array<Real, 1, 1> HO_p_guess;
  Array<Real, 1, 1> HO_p_exact;
  Array<Real, 1, 1> Bessel_p_guess;
  Array<Real, 1, 1> Bessel_p_exact;
  HO_p_guess(0, 0) = 1.5;
  HO_p_exact(0, 0) = m * m + quarter<Real>();
  Bessel_p_guess(0, 0) = 5;
  Bessel_p_exact(0, 0) = static_cast<Real>(1998169552430516840)/static_cast<Real>(345513626093455697) + m * m;

  Array<Real, 10, 1> x = Array<Real, 10,1>::LinSpaced(10, 0, 2 * pi<Real>()); // Interval x = [0, 2 * pi]
  Array<Real, 10, 1> r = Array<Real, 10, 1>::LinSpaced(10, 0, 1); // Interval r = [0, 1]

  Array<Real, 2, 10> HO_y_guess = Array<Real, 2, 10>::Zero();
  Array<Real, 2, 10> Bessel_y_guess = Array<Real, 2, 10>::Zero();
  HO_y_guess(0, 4) = 1;
  HO_y_guess(0, 5) = 1;
  HO_y_guess(1, 0) = 0.5;
  HO_y_guess(1, 9) = -0.5;
  Bessel_y_guess(0, 0) = 1;
  Bessel_y_guess(1, 9) = -1;


  Array<Real, 2, 10> HO_y_exact;
  Array<Real, 2, 10> Bessel_y_exact;
  HO_y_exact.row(0) = (sqrt(HO_p_exact.value() - m * m) * x.transpose()).sin();
  HO_y_exact.row(1) = sqrt(HO_p_exact.value() - m * m) * (sqrt(HO_p_exact.value() - m * m) * x.transpose()).cos();
  for (Index idx = 0; idx < 10; ++idx) {
    Bessel_y_exact(0, idx) = cyl_bessel_j(0, sqrt(Bessel_p_exact.value() - m * m) * r(idx));
    Bessel_y_exact(1, idx) = - sqrt(Bessel_p_exact.value() - m * m) * cyl_bessel_j(1, sqrt(Bessel_p_exact.value() - m * m) * r(idx));
  }


  HO_fun ho_fun(m, HO_p_exact.value());
  HO_bc ho_bc(m, HO_p_exact.value());
  HO_fun_jac ho_fun_jac(m, HO_p_exact.value());
  HO_bc_jac ho_bc_jac(m, HO_p_exact.value());

  Bessel_fun bessel_fun(m, Bessel_p_exact.value());
  Bessel_bc bessel_bc(m, Bessel_p_exact.value());
  Bessel_fun_jac bessel_fun_jac(m, Bessel_p_exact.value());
  Bessel_bc_jac bessel_bc_jac(m, Bessel_p_exact.value());

  Matrix<Real, 2, 2> S;
  S << 0, 0, 0, -1;

  std::cout << "Harmonic Oscillator" << std::endl;
  auto HO_result1 = collocation::bvp4c<Real, 2, 1>::solve(ho_fun, ho_bc, ho_fun_jac, ho_bc_jac, x, HO_y_guess, HO_p_guess, 1.0e-9, 1.0e-9, 10000);
  // std::cout << "Found Solution:" << std::endl;
  // std::cout << HO_result1.sol(x) << std::endl;
  std::cout << "Found Eigenvalue: " << std::setprecision(std::numeric_limits<Real>::max_digits10) << HO_result1.p.value() << std::endl;
  std::cout << "Maximum relative error at mesh nodes: ";
  std::cout << std::setprecision(8) << ((HO_y_exact - HO_result1.sol(x)) / (1 + HO_result1.sol(x).abs())).matrix().colwise().stableNorm().maxCoeff() << std::endl << std::endl;

  std::cout << "Harmonic Oscillator estimate jacobians" << std::endl;
  auto HO_result2 = collocation::bvp4c<Real, 2, 1>::solve(ho_fun, ho_bc, x, HO_y_guess, HO_p_guess, 1.0e-9, 1.0e-9, 10000);
  // std::cout << "Found Solution:" << std::endl;
  // std::cout << std::setprecision(6) << HO_result2.sol(x) << std::endl;
  std::cout << "Found eigenvalue: " << std::setprecision(std::numeric_limits<Real>::max_digits10) << HO_result2.p.value() << std::endl;
  std::cout << "Maximum relative error at mesh nodes: ";
  std::cout << std::setprecision(8) << ((HO_y_exact - HO_result2.sol(x)) / (1 + HO_result2.sol(x).abs())).matrix().colwise().stableNorm().maxCoeff() << std::endl << std::endl;

  std::cout << "Harmonic Oscillator eigenvalue supplied" << std::endl;
  auto HO_result3 = collocation::bvp4c<Real, 2, 0>::solve(ho_fun, ho_bc, ho_fun_jac, ho_bc_jac, x, HO_y_guess, 1.0e-9, 1.0e-9, 10000);
  // std::cout << "Found Solution:" << std::endl;
  // std::cout << std::setprecision(6) << HO_result3.sol(x) << std::endl;
  std::cout << "Exact eigenvalue: " << std::setprecision(std::numeric_limits<Real>::max_digits10) << HO_p_exact.value() << std::endl;
  std::cout << "Maximum relative error at mesh nodes: ";
  std::cout << std::setprecision(8) << ((HO_y_exact - HO_result3.sol(x)) / (1 + HO_result3.sol(x).abs())).matrix().colwise().stableNorm().maxCoeff() << std::endl << std::endl;

  std::cout << "Harmonic Oscillator estimate jacobians eigenvalue supplied" << std::endl;
  auto HO_result4 = collocation::bvp4c<Real, 2, 0>::solve(ho_fun, ho_bc, x, HO_y_guess, 1.0e-9, 1.0e-9, 10000);
  // std::cout << "Found Solution:" << std::endl;
  // std::cout << std::setprecision(6) << HO_result4.sol(x) << std::endl;
  std::cout << "Exact eigenvalue: " << std::setprecision(std::numeric_limits<Real>::max_digits10) << HO_p_exact.value() << std::endl;
  std::cout << "Maximum relative error at mesh nodes: ";
  std::cout << std::setprecision(8) << ((HO_y_exact - HO_result4.sol(x)) / (1 + HO_result4.sol(x).abs())).matrix().colwise().stableNorm().maxCoeff() << std::endl << std::endl;

  std::cout << "Bessel Function" << std::endl;
  auto Bessel_result1 = collocation::bvp4c<Real, 2, 1>::solve(bessel_fun, bessel_bc, bessel_fun_jac, bessel_bc_jac, r, Bessel_y_guess, Bessel_p_guess, S, 1.0e-9, 1.0e-9, 10000);
  // std::cout << "Found Solution:" << std::endl;
  // std::cout << std::setprecision(6) << Bessel_result1.sol(r) << std::endl;
  std::cout << "Found eigenvalue: " << std::setprecision(std::numeric_limits<Real>::max_digits10) << Bessel_result1.p.value() << std::endl;
  std::cout << "Maximum relative error at mesh nodes: ";
  std::cout << std::setprecision(8) << ((Bessel_y_exact - Bessel_result1.sol(r)) / (1 + Bessel_result1.sol(r).abs())).matrix().colwise().stableNorm().maxCoeff() << std::endl << std::endl;

  std::cout << "Bessel Function estimate jacobians" << std::endl;
  auto Bessel_result2 = collocation::bvp4c<Real, 2, 1>::solve(bessel_fun, bessel_bc, r, Bessel_y_guess, Bessel_p_guess, S, 1.0e-9, 1.0e-9, 10000);
  // std::cout << "Found Solution:" << std::endl;
  // std::cout << std::setprecision(6) << Bessel_result2.sol(r) << std::endl;
  std::cout << "Found eigenvalue: " << std::setprecision(std::numeric_limits<Real>::max_digits10) << Bessel_result2.p.value() << std::endl;
  std::cout << "Maximum relative error at mesh nodes:";
  std::cout << std::setprecision(8) << ((Bessel_y_exact - Bessel_result2.sol(r)) / (1 + Bessel_result2.sol(r).abs())).matrix().colwise().stableNorm().maxCoeff() << std::endl << std::endl;

  std::cout << "Bessel Function eigenvalue supplied" << std::endl;
  auto Bessel_result3 = collocation::bvp4c<Real, 2, 0>::solve(bessel_fun, bessel_bc, bessel_fun_jac, bessel_bc_jac, r, Bessel_y_guess, S, 1.0e-9, 1.0e-9, 10000);
  // std::cout << "Found Solution:" << std::endl;
  // std::cout << std::setprecision(8) << Bessel_result3.sol(r) << std::endl;
  // std::cout << "Exact Solution:" << std::endl;
  // std::cout << std::setprecision(8) << Bessel_y_exact << std::endl;
  std::cout << "Exact eigenvalue: " << std::setprecision(std::numeric_limits<Real>::max_digits10) << Bessel_p_exact.value() << std::endl;
  std::cout << "Maximum relative error at mesh nodes: ";
  std::cout << std::setprecision(8) << ((Bessel_y_exact - Bessel_result3.sol(r)) / (1 + Bessel_result3.sol(r).abs())).matrix().colwise().stableNorm().maxCoeff() << std::endl << std::endl;

  std::cout << "Bessel Function estimate jacobains eigenvalue supplied" << std::endl;
  auto Bessel_result4 = collocation::bvp4c<Real, 2, 0>::solve(bessel_fun, bessel_bc, r, Bessel_y_guess, S, 1.0e-9, 1.0e-9, 10000);
  // std::cout << "Found Solution:" << std::endl;
  // std::cout << std::setprecision(6) << Bessel_result4.sol(r) << std::endl;
  std::cout << "Exact eigenvalue: " << std::setprecision(std::numeric_limits<Real>::max_digits10) << Bessel_p_exact.value() << std::endl;
  std::cout << "Maximum relative error at mesh nodes: ";
  std::cout << std::setprecision(8) << ((Bessel_y_exact - Bessel_result4.sol(r)) / (1 + Bessel_result4.sol(r).abs())).matrix().colwise().stableNorm().maxCoeff() << std::endl << std::endl;


  return 0;
}
