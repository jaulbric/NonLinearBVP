#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <iostream>

#include "NonLinearOptimization/PowellHybrid.hpp"
#include "collocation/methods/MIRK4.hpp"
#include "TestProblems.hpp"

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
  // Array<Real, 2, 10> HO_y_guess;
  Array<Real, 2, 10> Bessel_y_guess = Array<Real, 2, 10>::Zero();
  HO_y_guess(0, 4) = 1;
  HO_y_guess(0, 5) = 1;
  // HO_y_guess(0, 2) = 0.25;
  // HO_y_guess(0, 8) = 0.25;
  // HO_y_guess << 0, 0.4, 0.6, 0.8, 1.0, 1.0, 0.8, 0.6, 0.4, 0.0, 0.5, 0.5, 0.4, 0.3, 0.1, -0.1, -0.3, -0.4, -0.5, -0.5;
  HO_y_guess(1, 0) = 0.5;
  HO_y_guess(1, 9) = -0.5;
  Bessel_y_guess(0, 0) = 1;
  Bessel_y_guess(1, 9) = -1;


  Array<Real, 2, 10> HO_y_exact;
  Array<Real, 2, 10> Bessel_y_exact;
  HO_y_exact.row(0) = (sqrt(HO_p_exact.value() - m * m) * x.transpose()).sin();
  HO_y_exact.row(1) = sqrt(HO_p_exact.value() - m * m) * (sqrt(HO_p_exact.value() - m * m) * x.transpose()).cos();
  for (Index idx = 0; idx < 0; ++idx) {
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

  std::cout << "Creating instance of MIRK4 class..." << std::endl << std::endl;
  collocation::methods::MIRK4 method(ho_fun, ho_bc, ho_fun_jac, ho_bc_jac, x, HO_y_guess, HO_p_guess);

  std::cout << "Initial guess:" << std::endl;
  std::cout << method.y << std::endl;
  std::cout << "Residues:" << std::endl;
  std::cout << method.residues.transpose() << std::endl;

  std::cout << "Creating instance of PowellHybrid..." << std::endl;
  collocation::PowellHybrid solver(method);
  // std::cout << "Calling solveInit..." << std::endl;
  // auto status = solver.solveInit();
  // for (int i = 0; i < 2; ++i) {
  //   std::cout << "Calling solveOneStep..." << std::endl;
  //   status = solver.solveOneStep();
  // }

  std::cout << "Calling solve()..." << std::endl;
  auto status = solver.solve();
  std::cout << "status = " << status << std::endl;

  std::cout << "Found solution:" << std::endl;
  std::cout << method.y << std::endl;
  std::cout << "Residues:" << std::endl;
  std::cout << method.residues.transpose() << std::endl << std::endl;

  std::cout << "Exact Solution:" << std::endl;
  std::cout << HO_y_exact << std::endl << std::endl;

  std::cout << "Error:" << std::endl;
  std::cout << HO_y_exact - method.y << std::endl;

  return 0;
}
