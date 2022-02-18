#include <Eigen/Dense>
#include <Eigen/CXX11/Tensor>
#include <boost/math/constants/constants.hpp>
#include <tuple>
#include <utility>
#include <cmath>

using Eigen::Array;
using Eigen::ArrayBase;
using Eigen::Tensor;
using Eigen::Index;

using std::sqrt;
using boost::math::constants::half;

template <typename Real>
class HO_fun {
  public:
    HO_fun(Real m, Real p) : m_{m}, p_{p} {};
    ~HO_fun() = default;

    template <typename Derived1, typename Derived2, typename Derived3>
    Array<typename Derived2::Scalar, 2, Derived2::ColsAtCompileTime>
    operator()(const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y, const ArrayBase<Derived3>& p) const {
      Array<typename Derived2::Scalar, 2, Derived2::ColsAtCompileTime> dydx(2, y.cols());
      dydx.row(0) = y.row(1);
      dydx.row(1) = (m_ * m_ - p.value()) * y.row(0);
      return dydx;
    }

    template <typename Derived1, typename Derived2>
    Array<typename Derived2::Scalar, 2, Derived2::ColsAtCompileTime>
    operator()(const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y) const {
      Array<typename Derived2::Scalar, 2, Derived2::ColsAtCompileTime> dydx(2, y.cols());
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
