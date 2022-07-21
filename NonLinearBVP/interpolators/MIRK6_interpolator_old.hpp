#ifndef MIRK6_INTERPOLATOR_HPP
#define MIRK6_INTERPOLATOR_HPP

#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <limits>
#include <Eigen/Dense>

#include "NonLinearBVP/detail/collocation/methods/MIRK6.hpp"
#include "NonLinearBVP/detail/constants/constants.hpp"
#include "NonLinearBVP/constants/MIRK6_constants.hpp"

namespace collocation {namespace methods {
template <typename Scalar, Index RowsAtCompileTime, Index ColsAtCompileTime, Index ParamsAtCompileTime, typename F, typename BC, typename FJ, typename BCJ>
class MIRK6;
}}

namespace interpolators {

using Eigen::last;
using Eigen::all;
using Eigen::Dynamic;
using Eigen::ArrayX;
using Eigen::Array;
using Eigen::ArrayBase;
using Eigen::Index;

using boost::math::constants::half;
using boost::math::constants::MIRK6_alpha;

template <Index ColsAtCompileTime>
struct MIRK6_Interpolator_Traits {
  static constexpr Index IntervalsAtCompileTime = ColsAtCompileTime - 1;
};

template <>
struct MIRK6_Interpolator_Traits<Dynamic> {
  static constexpr Index IntervalsAtCompileTime = Dynamic;
};


template <typename _Scalar, Index _RowsAtCompileTime, Index _ColsAtCompileTime>
class MIRK6_interpolator : MIRK6_Interpolator_Traits<_ColsAtCompileTime> {
public:
  typedef _Scalar Scalar;
  static constexpr Index RowsAtCompileTime = _RowsAtCompileTime;
  static constexpr Index ColsAtCompileTime = _ColsAtCompileTime;
  static constexpr Index IntervalsAtCompileTime = MIRK6_Interpolator_Traits<_ColsAtCompileTime>::IntervalsAtCompileTime;

  template <typename Derived1, typename Derived2, typename Derived3, typename Derived4, typename Derived5, typename Derived6>
  MIRK6_interpolator(const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y, const ArrayBase<Derived3>& f, const ArrayBase<Derived4>& f1, const ArrayBase<Derived5>& f2, const ArrayBase<Derived6>& f3) : m_x{x}, m_y{y}, m_f{f}, m_f1{f1}, m_f2{f2}, m_f3{f3}, m_rows{y.rows()}, m_cols{y.cols()} {
    assert(m_x.size() == m_y.cols());
    assert(m_x.size() == m_f.cols());
    assert(m_y.rows() == m_f.rows());
    assert(m_x.size() > 1);
    assert(((m_x.tail(m_cols - 1) - m_x.head(m_cols - 1)) >= 0).all());
  }

  template <Index ParamsAtCompileTime, typename F, typename BC, typename FJ, typename BCJ>
  MIRK6_interpolator(const collocation::methods::MIRK6<_Scalar, _RowsAtCompileTime, _ColsAtCompileTime, ParamsAtCompileTime, F, BC, FJ, BCJ>& mirk6) : m_x{mirk6.m_x}, m_y{mirk6.m_y}, m_f{mirk6.m_f}, m_f1{mirk6.m_f_internal1}, m_f2{mirk6.m_f_internal2}, m_f3{mirk6.m_f_internal3}, m_rows{mirk6.m_rows}, m_cols{mirk6.m_cols} {}

  Array<Scalar, RowsAtCompileTime, 1> operator()(Scalar x) const {
    if  (x < m_x(0) || x > m_x(last)) {
      std::ostringstream oss;
      oss.precision(std::numeric_limits<Scalar>::digits10+3);
      oss << "Requested abscissa x = " << x << ", which is outside of allowed range ["
          << m_x(0) << ", " << m_x(last) << "]";
      throw std::domain_error(oss.str());
    }
    // We need t := (x-x_k)/(x_{k+1}-x_k) \in [0,1) for this to work.
    // Sadly this neccessitates this loathesome check, otherwise we get t = 1 at x = xf.
    if (x == m_x(last)) {
      return m_y(all, last);
    }

    auto it = std::upper_bound(m_x.begin(), m_x.end(), x);
    auto i = std::distance(m_x.begin(), it) -1;
    Scalar xi = *(it-1);
    Scalar xip1 = *it;
    auto yi = m_y.col(i);
    auto yip1 = m_y.col(i+1);
    auto fi = m_f.col(i);
    auto fip1 = m_f.col(i+1);
    auto f1 = m_f1.col(i);
    auto f2 = m_f2.col(i);
    auto f3 = m_f3.col(i);
    Scalar hi = (xip1 - xi);
    Scalar w = (x - xi) / hi;

    return m_A(w) * yip1 + m_A(1-w) * yi + hi * (m_B(w) * fip1 - m_B(1 - w) * fi + m_C(w) * f3 - m_C(1-w) * f1 + m_D(w) * f2);
  }

  template <typename Derived>
  Array<Scalar, RowsAtCompileTime, Derived::SizeAtCompileTime> operator()(const ArrayBase<Derived>& x) const {
    assert(x.rows() == 1 || x.cols() == 1);
    Array<Scalar, RowsAtCompileTime, Derived::SizeAtCompileTime> ret(m_rows, x.size());
    for (Index idx = 0; idx < x.size(); ++idx) {
      ret.col(idx) = this->operator()(x(idx));
    }
    return ret;
  }

  Array<Scalar, RowsAtCompileTime, 1> prime(Scalar x) const {
    if  (x < m_x(0) || x > m_x(last)) {
      std::ostringstream oss;
      oss.precision(std::numeric_limits<Scalar>::digits10+3);
      oss << "Requested abscissa x = " << x << ", which is outside of allowed range ["
          << m_x(0) << ", " << m_x(last) << "]";
      throw std::domain_error(oss.str());
    }
    if (x == m_x(last)) {
      return m_f(all, last);
    }
    auto it = std::upper_bound(m_x.begin(), m_x.end(), x);
    auto i = std::distance(m_x.begin(), it) -1;
    Scalar xi = *(it-1);
    Scalar xip1 = *it;
    auto yi = m_y.col(i);
    auto yip1 = m_y.col(i+1);
    auto fi = m_f.col(i);
    auto fip1 = m_f.col(i+1);
    auto f1 = m_f1.col(i);
    auto f2 = m_f2.col(i);
    auto f3 = m_f3.col(i);
    Scalar hi = (xip1 - xi);
    Scalar w = (x - xi) / hi;

    return m_Aprime(w) * (yip1 - yi) / hi - m_Bprime(w) * fip1 - m_Bprime(1 - w) * fi - m_Cprime(w) * f3 - m_Cprime(1-w) * f3 - m_Dprime(w) * f2;
  }

  template <typename Derived>
  Array<Scalar, RowsAtCompileTime, Derived::SizeAtCompileTime> prime(const ArrayBase<Derived>& x) {
    assert(x.rows() == 1 || x.cols() == 1);
    Array<Scalar, RowsAtCompileTime, Derived::SizeAtCompileTime> ret(m_rows, x.size());
    for (Index idx = 0; idx < x.size(); ++idx) {
      ret.col(idx) = prime(x(idx));
    }
    return ret;
  }

  friend std::ostream& operator<<(std::ostream & os, const MIRK6_interpolator & sol) {
    os << "x = " << sol.m_x << std::endl;
    os << "y = " << sol.m_y << std::endl;
    os << "y' = " << sol.m_f << std::endl;
    return os;
  }

  Index size() const {return m_y.size();}

  int64_t bytes() const {return 3*m_y.size()*sizeof(Scalar) + 3*sizeof(m_y);}

  std::pair<Scalar, Scalar> domain() const {return {m_x(0), m_x(last)};}

  Index rows() const {return m_rows;}

  Index cols() const {return m_cols;}

private:
  Array<Scalar, ColsAtCompileTime, 1> m_x;
  Array<Scalar, RowsAtCompileTime, ColsAtCompileTime> m_y;
  Array<Scalar, RowsAtCompileTime, ColsAtCompileTime> m_f;
  Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> m_f1;
  Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> m_f2;
  Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> m_f3;

  Index m_rows;
  Index m_cols;

  Scalar m_A(const Scalar w) const {Scalar temp = w * w; return w * temp * (10 - 15 * w + 6 * temp);}
  Scalar m_B(const Scalar w) const {Scalar temp = w * w; return half<Scalar>() * temp * (1 - w) * (1 - 2 * w + 3 * temp);}
  Scalar m_C(const Scalar w) const {Scalar temp = 1 - w; return 25 * half<Scalar>() * w * w * temp * temp * (w - half<Scalar>() + MIRK6_alpha<Scalar>());}
  Scalar m_D(const Scalar w) const {Scalar temp = 1 - w; return 8 * w * w * temp * temp * (1 - 2 * w);}
  Scalar m_Aprime(const Scalar w) const {Scalar temp = 1 - w; return 30 * w * w * temp * temp;}
  Scalar m_Bprime(const Scalar w) const {return half<Scalar>() * w * (2 + w * (5 * w * (4 - 3 * w) - 9));}
  Scalar m_Cprime(const Scalar w) const {return 25 * half<Scalar>() * w * (w - 1) * (2 * MIRK6_alpha<Scalar>() * (2 * w - 1) + 5 * w * (w - 1) + 1);}
  Scalar m_Dprime(const Scalar w) const {return 16 * w * (1 - w) * (5 * w * (w - 1) + 1);}
};

template <typename Derived1, typename Derived2, typename Derived3>
MIRK6_interpolator(const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y, const ArrayBase<Derived3> & f) -> MIRK6_interpolator<typename Derived2::Scalar, Derived2::RowsAtCompileTime, Derived2::ColsAtCompileTime>;

template <typename Scalar, Index RowsAtCompileTime, Index ColsAtCompileTime, Index ParamsAtCompileTime, typename F, typename BC, typename FJ, typename BCJ>
MIRK6_interpolator(const collocation::methods::MIRK6<Scalar, RowsAtCompileTime, ColsAtCompileTime, ParamsAtCompileTime, F, BC, FJ, BCJ>& mirk4) -> MIRK6_interpolator<Scalar, RowsAtCompileTime, ColsAtCompileTime>;

} // interpolators
#endif // MIRK6_INTERPOLATOR_HPP
