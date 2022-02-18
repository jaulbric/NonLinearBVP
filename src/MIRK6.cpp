#include <Eigen/Dense>
#include <Eigen/CXX11/Tensor>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <utility>
#include <tuple>
#include <cmath>
#include <iostream>
#include <fstream>
#include <complex>
#include <iomanip>
#include <limits>
#include <chrono>

// #include "interpolators/vector_cubic_hermite.hpp"
// #include "collocation/tools.hpp"
// #include "collocation/collocation.hpp"

// #include "collocation/compute_jac_indices.hpp"
// #include "collocation/collocation_functions.hpp"
// #include "collocation/construct_global_jac.hpp"
// #include "collocation/estimate_rms_residuals.hpp"

// #include "collocation/methods/MIRK6.hpp"
#include "collocation/bvp6c.hpp"
#include "interpolators/MIRK6_interpolator.hpp"
#include "collocation/BVP_Result.hpp"

using Eigen::array;
using Eigen::Array;
using Eigen::ArrayBase;
using Eigen::Matrix;
using Eigen::Tensor;
using Eigen::TensorBase;
using Eigen::ReadOnlyAccessors;
using Eigen::TensorRef;
using Eigen::Index;
using Eigen::IndexPair;
using std::sqrt;
using boost::math::constants::pi;
using boost::math::constants::half;
using boost::math::constants::quarter;
using boost::math::cyl_bessel_j;

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

template <typename Real>
class HO_fun {
  public:
    HO_fun(Real m, Real p) : m_{m}, p_{p} {};
    ~HO_fun() = default;

    template <typename Derived1, typename Derived2, typename Derived3>
    Array<Real, 2, Derived2::ColsAtCompileTime>
    operator()(const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y, const ArrayBase<Derived3>& p) const {
      Array<Real, 2, Derived2::ColsAtCompileTime> dydx(2, y.cols());
      dydx.row(0) = y.row(1);
      dydx.row(1) = (m_ * m_ - p.value()) * y.row(0);
      return dydx;
    }

    template <typename Derived1, typename Derived2>
    Array<Real, 2, Derived2::ColsAtCompileTime>
    operator()(const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y) const {
      Array<Real, 2, Derived2::ColsAtCompileTime> dydx(2, y.cols());
      dydx.row(0) = y.row(1);
      dydx.row(1) = (m_ * m_ - p_) * y.row(0);
      return dydx;
    }
  private:
    Real m_;
    Real p_;
};

template <typename Real>
class HO_bc {
  public:
    HO_bc(Real m, Real p) : m_{m}, p_{p} {};
    ~HO_bc() = default;

    template <typename Derived1, typename Derived2, typename Derived3>
    Array<Real, 3, 1> operator()(const ArrayBase<Derived1>& ya, const ArrayBase<Derived2>& yb, const ArrayBase<Derived3>& p) const {
      return Array<Real, 3, 1>({ya(0), yb(0), ya(1) - sqrt(p.value() - m_ * m_)});
    }

    template <typename Derived1, typename Derived2>
    Array<Real, 2, 1> operator()(const ArrayBase<Derived1>& ya, const ArrayBase<Derived2>& yb) const {
      // return Array<Real, 2, 1>({ya(0), yb(0)});
      return Array<Real, 2, 1>({ya(0), yb(1) + half<Real>()});
    }
  private:
    Real m_;
    Real p_;
};

template <typename Real>
class HO_fun_jac {
  public:
    HO_fun_jac(Real m, Real p) : m_{m}, p_{p} {};
    ~HO_fun_jac() = default;

    template <typename Derived1, typename Derived2, typename Derived3>
    std::pair<Tensor<Real, 3>, Tensor<Real, 3>>
    operator()(const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y, const ArrayBase<Derived3>& p) const {
      Tensor<Real, 3> dfdy(2, 2, x.size());
      Tensor<Real, 3> dfdp(2, 1, x.size());
      for (Index idx = 0; idx < x.size(); ++idx) {
        dfdy(0, 0, idx) = 0;
        dfdy(0, 1, idx) = 1;
        dfdy(1, 0, idx) = m_ * m_ - p.value();
        dfdy(1, 1, idx) = 0;
        dfdp(0, 0, idx) = 0;
        dfdp(1, 0, idx) = -y(0, idx);
      }
      return std::make_pair(dfdy, dfdp);
    }

    template <typename Derived1, typename Derived2>
    Tensor<Real, 3>
    operator()(const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y) const {
      Tensor<Real, 3> dfdy(2, 2, x.size());
      for (Index idx = 0; idx < x.size(); ++idx) {
        dfdy(0, 0, idx) = 0;
        dfdy(0, 1, idx) = 1;
        dfdy(1, 0, idx) = m_ * m_ - p_;
        dfdy(1, 1, idx) = 0;
      }
      return dfdy;
    }
  private:
    Real m_;
    Real p_;
};

template <typename Real>
class HO_bc_jac {
  public:
    HO_bc_jac(Real m, Real p) : m_{m}, p_{p} {};
    ~HO_bc_jac() = default;

    template <typename Derived1, typename Derived2, typename Derived3>
    std::tuple<Array<Real, 3, 2>, Array<Real, 3, 2>, Array<Real, 3, 1>>
    operator()(const ArrayBase<Derived1>& ya, const ArrayBase<Derived2>& yb, const ArrayBase<Derived3>& p) const {
      Array<Real, 3, 2> dbc_dya = Array<Real, 3, 2>::Zero();
      Array<Real, 3, 2> dbc_dyb = Array<Real, 3, 2>::Zero();
      Array<Real, 3, 1> dbc_dp = Array<Real, 3, 1>::Zero();
      dbc_dya(0, 0) = 1;
      dbc_dya(2, 1) = 1;
      dbc_dyb(1, 0) = 1;
      dbc_dp(2, 0) = - half<Real>() / sqrt(p.value() - m_ * m_);
      return std::make_tuple(dbc_dya, dbc_dyb, dbc_dp);
    }

    template <typename Derived1, typename Derived2>
    std::tuple<Array<Real, 2, 2>, Array<Real, 2, 2>>
    operator()(const ArrayBase<Derived1>& ya, const ArrayBase<Derived2>& yb) const {
      Array<Real, 2, 2> dbc_dya = Array<Real, 2, 2>::Zero();
      Array<Real, 2, 2> dbc_dyb = Array<Real, 2, 2>::Zero();
      dbc_dya(0, 0) = 1;
      // dbc_dyb(1, 0) = 1;
      dbc_dyb(1, 1) = 1;
      return std::make_tuple(dbc_dya, dbc_dyb);
    }
  private:
    Real m_;
    Real p_;
};

template <typename Real>
class Bessel_fun {
  public:
    Bessel_fun(Real m, Real p) : m_{m}, p_{p} {};
    ~Bessel_fun() = default;

    template <typename Derived1, typename Derived2, typename Derived3>
    Array<Real, 2, Derived2::ColsAtCompileTime> operator()(const ArrayBase<Derived1>& r,
                                                           const ArrayBase<Derived2>& y,
                                                           const ArrayBase<Derived3>& p) const {
      Array<Real, 2, Derived2::ColsAtCompileTime> dydr(2, y.cols());
      dydr.row(0) = y.row(1);
      dydr.row(1) = (m_ * m_ - p.value()) * y.row(0);
      return dydr;
    }

    template <typename Derived1, typename Derived2>
    Array<Real, 2, Derived2::ColsAtCompileTime> operator()(const ArrayBase<Derived1>& r, const ArrayBase<Derived2>& y) const {
      Array<Real, 2, Derived2::ColsAtCompileTime> dydr(2, y.cols());
      dydr.row(0) = y.row(1);
      dydr.row(1) = (m_ * m_ - p_) * y.row(0);
      return dydr;
    }
  private:
    Real m_;
    Real p_;
};

template <typename Real>
class Bessel_bc {
  public:
    Bessel_bc(Real m, Real p) : m_{m}, p_{p} {};
    ~Bessel_bc() = default;

    template <typename Derived1, typename Derived2, typename Derived3>
    Array<Real, 3, 1> operator()(const ArrayBase<Derived1>& ya, const ArrayBase<Derived2>& yb, const ArrayBase<Derived3>& p) const {
      return Array<Real, 3, 1>({ya(0) - static_cast<Real>(1), yb(0), ya(1)});
    }

    template <typename Derived1, typename Derived2>
    Array<Real, 2, 1> operator()(const ArrayBase<Derived1>& ya, const ArrayBase<Derived2>& yb) const {
      return Array<Real, 2, 1>({ya(0) - static_cast<Real>(1), yb(0)});
    }
  private:
    Real m_;
    Real p_;
};

template <typename Real>
class Bessel_fun_jac {
  public:
    Bessel_fun_jac(Real m, Real p) : m_{m}, p_{p} {};
    ~Bessel_fun_jac() = default;

    template <typename Derived1, typename Derived2, typename Derived3>
    std::pair<Tensor<Real, 3>, Tensor<Real, 3>> operator()(const ArrayBase<Derived1>& r, const ArrayBase<Derived2>& y, const ArrayBase<Derived3>& p) const {
      Tensor<Real, 3> dfdy(2, 2, r.size());
      Tensor<Real, 3> dfdp(2, 1, r.size());
      for (Index idx = 0; idx < r.size(); ++idx) {
        dfdy(0, 0, idx) = 0;
        dfdy(0, 1, idx) = 1;
        dfdy(1, 0, idx) = m_ * m_ - p.value();
        dfdy(1, 1, idx) = 0;
        dfdp(0, 0, idx) = 0;
        dfdp(1, 0, idx) = - y(0, idx);
      }
      return std::make_pair(dfdy, dfdp);
    }

    template <typename Derived1, typename Derived2>
    Tensor<Real, 3> operator()(const ArrayBase<Derived1>& r, const ArrayBase<Derived2>& y) const {
      Tensor<Real, 3> dfdy(2, 2, r.size());
      for (Index idx = 0; idx < r.size(); ++idx) {
        dfdy(0, 0, idx) = 0;
        dfdy(0, 1, idx) = 1;
        dfdy(1, 0, idx) = m_ * m_ - p_;
        dfdy(1, 1, idx) = 0;
      }
      return dfdy;
    }
  private:
    Real m_;
    Real p_;
};

template <typename Real>
class Bessel_bc_jac {
  public:
    Bessel_bc_jac(Real m, Real p) : m_{m}, p_{p} {};
    ~Bessel_bc_jac() = default;

    template <typename Derived1, typename Derived2, typename Derived3>
    std::tuple<Array<Real, 3, 2>, Array<Real, 3, 2>, Array<Real, 3, 1>>
    operator()(const ArrayBase<Derived1>& ya, const ArrayBase<Derived2>& yb, const ArrayBase<Derived3>& p) {
      Array<Real, 3, 2> dbc_dya = Array<Real, 3, 2>::Zero();
      Array<Real, 3, 2> dbc_dyb = Array<Real, 3, 2>::Zero();
      Array<Real, 3, 1> dbc_dp = Array<Real, 3, 1>::Zero();
      dbc_dya(0, 0) = 1;
      dbc_dya(2, 1) = 1;
      dbc_dyb(1, 0) = 1;
      return std::make_tuple(dbc_dya, dbc_dyb, dbc_dp);
    }

    template <typename Derived1, typename Derived2>
    std::tuple<Array<Real, 2, 2>, Array<Real, 2, 2>>
    operator()(const ArrayBase<Derived1>& ya, const ArrayBase<Derived2>& yb) {
      Array<Real, 2, 2> dbc_dya = Array<Real, 2, 2>::Zero();
      Array<Real, 2, 2> dbc_dyb = Array<Real, 2, 2>::Zero();
      dbc_dya(0, 0) = 1;
      dbc_dyb(1, 0) = 1;
      return std::make_tuple(dbc_dya, dbc_dyb);
    }
  private:
    Real m_;
    Real p_;
};

template <typename Scalar, Index n, Index m, Index k, typename F, typename BC, typename FJ, typename BCJ>
void test_fun(const collocation::methods::MIRK4<Scalar, n, m, k, F, BC, FJ, BCJ>& mirk4) {
  std::cout << typeid(mirk4).name() << std::endl;
}

int main() {
  using Real = long double;

  // We solve four problems here. The first is the harmonic equation
  // - y'' + m**2 * y = p * y
  // y(0) = y(2 * pi) = 0
  // where p is the eigenvalue and m is some mass parameter.
  // The normalization is choosen so that the exact solution is y(x) = sin(sqrt(p - m**2) * x)
  // The exact eigenvalues are p = m**2 + (n / 2)**2, where n is any positive integer.
  // The second problem is Bessel's equation with order parameter nu = 0:
  // - y'' - (1/r) y' + m**2 y = p y
  Real m = 1;
  Array<Real, 1, 1> HO_p_guess;
  Array<Real, 1, 1> HO_p_exact;
  Array<Real, 1, 1> Bessel_p_guess;
  Array<Real, 1, 1> Bessel_p_exact;
  HO_p_guess(0, 0) = 1.5;
  HO_p_exact(0, 0) = m * m + quarter<Real>();
  Bessel_p_guess(0, 0) = 5;
  // Bessel_p_exact(0, 0) = static_cast<Real>(5.78318596294678452117599575845580703507144180642368558708712371445606430488554437388634035954449020438203125154) + m * m;
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

  std::cout << "Creating instance of MIRK4 class..." << std::endl;
  collocation::methods::MIRK4 method(ho_fun, ho_bc, ho_fun_jac, ho_bc_jac, x, HO_y_guess, HO_p_guess);

  std::cout << "Compile time dimensions:" << std::endl;
  std::cout << "RowsAtCompileTime: " << method.RowsAtCompileTime << ", ColsAtCompileTime: " << method.ColsAtCompileTime << ", ParamsAtCompileTime: " << method.ParamsAtCompileTime << std::endl;
  std::cout << "Runtime dimensions:" << std::endl;
  std::cout << "rows: " << method.rows() << ", cols: " << method.cols() << ", params: " << method.params() << std::endl << std::endl;

  std::cout << "method.x: " << std::endl;
  std::cout << method.x.transpose() << std::endl;
  std::cout << "method.y:" << std::endl;
  std::cout << method.y << std::endl;
  std::cout << "method.f:" << std::endl;
  std::cout << method.f << std::endl;
  std::cout << "method.residues:" << std::endl;
  std::cout << method.residues.transpose() << std::endl;
  std::cout << "method.normalized_residues:" << std::endl;
  std::cout << method.normalized_residues() << std::endl;
  method.estimate_rms_residuals();
  std::cout << "method.rms_residuals:" << std::endl;
  std::cout << method.rms_residuals.transpose() << std::endl;
  method.construct_global_jacobian();
  std::cout << "method.jacobian:" << std::endl;
  std::cout << method.jacobian << std::endl;

  std::cout << "method.ColsAtCompileTime: " << method.ColsAtCompileTime << std::endl << std::endl;

  // std::cout << method.x(seq(0, last-1)).transpose() << std::endl;
  // std::cout << method.y(all, 1) << std::endl;

  std::cout << "Force Dynamic mesh..." << std::endl;

  Array<Real, Dynamic, 1> x1 = x;
  Array<Real, Dynamic, 1> x2 = x.head<9>();
  Array<Real, 2, Dynamic> y1 = HO_y_guess;
  Array<Real, 2, Dynamic> y2 = y1(all, seq(0, 8));

  // collocation::methods::MIRK4<Real, 2, Dynamic, 1, decltype(ho_fun), decltype(ho_bc), decltype(ho_fun_jac), decltype(ho_bc_jac)> method1(ho_fun, ho_bc, ho_fun_jac, ho_bc_jac, x1, y1, HO_p_guess);

  collocation::methods::MIRK4<Real, 2, Dynamic, 1, decltype(ho_fun), decltype(ho_bc), decltype(ho_fun_jac), decltype(ho_bc_jac)> method1(ho_fun, ho_bc, ho_fun_jac, ho_bc_jac, x, HO_y_guess, HO_p_guess);

  // collocation::methods::MIRK4<Real, 2, Dynamic, 1> method1(ho_fun, ho_bc, ho_fun_jac, ho_bc_jac, x, HO_y_guess, HO_p_guess);

  std::cout << "method.ColsAtCompileTime: " << method1.ColsAtCompileTime << std::endl;

  method1.estimate_rms_residuals();
  method1.construct_global_jacobian();
  std::cout << method1.rows() << std::endl;
  std::cout << method1.cols() << std::endl;
  std::cout << method1.params() << std::endl;

  method1.modify_mesh(x2, y2);
  method1.estimate_rms_residuals();
  method1.construct_global_jacobian();
  std::cout << method1.rows() << std::endl;
  std::cout << method1.cols() << std::endl;
  std::cout << method1.params() << std::endl;

  // interpolators::MIRK4_interpolator sol(method1);
  typename collocation::methods::MIRK4<Real, 2, Dynamic, 1, decltype(ho_fun), decltype(ho_bc), decltype(ho_fun_jac), decltype(ho_bc_jac)>::InterpolatorType sol(method1);

  std::cout << sol(3.0) << std::endl;

  collocation::BVP_Result result(method1, 10, 0, "message", true);

  std::cout << result.sol(3.0) << std::endl;


  // std::cout << "Harmonic Oscillator" << std::endl;
  // auto HO_result1 = collocation::collocation_algorithm<Real, 2, 1>::solve(ho_fun, ho_bc, ho_fun_jac, ho_bc_jac, x, HO_y_guess, HO_p_guess, 1.0e-9, 1000, 2, 1.0e-9);
  // std::cout << "Found Solution:" << std::endl;
  // std::cout << std::setprecision(6) << HO_result1.sol(x) << std::endl;
  // std::cout << "Found eigenvalue: " << std::setprecision(std::numeric_limits<Real>::max_digits10) << HO_result1.p.value() << std::endl << std::endl;
  //
  // std::cout << "Harmonic Oscillator estimate jacobians" << std::endl;
  // auto HO_result2 = collocation::collocation_algorithm<Real, 2, 1>::solve(ho_fun, ho_bc, x, HO_y_guess, HO_p_guess, 1.0e-9, 1000, 2, 1.0e-9);
  // std::cout << "Found Solution:" << std::endl;
  // std::cout << std::setprecision(6) << HO_result2.sol(x) << std::endl;
  // std::cout << "Found eigenvalue: " << std::setprecision(std::numeric_limits<Real>::max_digits10) << HO_result2.p.value() << std::endl << std::endl;
  //
  // std::cout << "Harmonic Oscillator eigenvalue supplied" << std::endl;
  // auto HO_result3 = collocation::collocation_algorithm<Real, 2, 0>::solve(ho_fun, ho_bc, ho_fun_jac, ho_bc_jac, x, HO_y_guess, 1.0e-9, 1000, 2, 1.0e-9);
  // std::cout << "Found Solution:" << std::endl;
  // std::cout << std::setprecision(6) << HO_result3.sol(x) << std::endl;
  // std::cout << "Exact eigenvalue: " << std::setprecision(std::numeric_limits<Real>::max_digits10) << HO_p_exact.value() << std::endl << std::endl;
  //
  // std::cout << "Harmonic Oscillator estimate jacobians eigenvalue supplied" << std::endl;
  // auto HO_result4 = collocation::collocation_algorithm<Real, 2, 0>::solve(ho_fun, ho_bc, x, HO_y_guess, 1.0e-9, 1000, 2, 1.0e-9);
  // std::cout << "Found Solution:" << std::endl;
  // std::cout << std::setprecision(6) << HO_result4.sol(x) << std::endl;
  // std::cout << "Exact eigenvalue: " << std::setprecision(std::numeric_limits<Real>::max_digits10) << HO_p_exact.value() << std::endl << std::endl;
  //
  // std::cout << "Bessel Function" << std::endl;
  // auto Bessel_result1 = collocation::collocation_algorithm<Real, 2, 1>::solve(bessel_fun, bessel_bc, bessel_fun_jac, bessel_bc_jac, r, Bessel_y_guess, Bessel_p_guess, S, 1.0e-9, 1000, 2, 1.0e-9);
  // std::cout << "Found Solution:" << std::endl;
  // std::cout << std::setprecision(6) << Bessel_result1.sol(r) << std::endl;
  // std::cout << "Found eigenvalue: " << std::setprecision(std::numeric_limits<Real>::max_digits10) << Bessel_result1.p.value() << std::endl << std::endl;
  //
  // std::cout << "Bessel Function estimate jacobians" << std::endl;
  // auto Bessel_result2 = collocation::collocation_algorithm<Real, 2, 1>::solve(bessel_fun, bessel_bc, r, Bessel_y_guess, Bessel_p_guess, S, 1.0e-9, 1000, 2, 1.0e-9);
  // std::cout << "Found Solution:" << std::endl;
  // std::cout << std::setprecision(12) << Bessel_result2.sol(r) << std::endl;
  // std::cout << "Found eigenvalue: " << std::setprecision(std::numeric_limits<Real>::max_digits10) << Bessel_result2.p.value() << std::endl << std::endl;
  //
  // std::cout << "Bessel Function eigenvalue supplied" << std::endl;
  // auto Bessel_result3 = collocation::collocation_algorithm<Real, 2, 0>::solve(bessel_fun, bessel_bc, bessel_fun_jac, bessel_bc_jac, r, Bessel_y_guess, S, 1.0e-9, 1000, 2, 1.0e-9);
  // std::cout << "Found Solution:" << std::endl;
  // std::cout << std::setprecision(12) << Bessel_result3.sol(r) << std::endl;
  // std::cout << "Exact eigenvalue: " << std::setprecision(std::numeric_limits<Real>::max_digits10) << Bessel_p_exact.value() << std::endl << std::endl;
  //
  // std::cout << "Bessel Function estimate jacobains eigenvalue supplied" << std::endl;
  // auto Bessel_result4 = collocation::collocation_algorithm<Real, 2, 0>::solve(bessel_fun, bessel_bc, r, Bessel_y_guess, S, 1.0e-9, 1000, 2, 1.0e-9);
  // std::cout << "Found Solution:" << std::endl;
  // std::cout << std::setprecision(6) << Bessel_result4.sol(r) << std::endl;
  // std::cout << "Exact eigenvalue: " << std::setprecision(std::numeric_limits<Real>::max_digits10) << Bessel_p_exact.value() << std::endl << std::endl;

  return 0;
}
