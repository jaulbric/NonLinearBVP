#ifndef MIRK4_INTERPOLATOR_HPP
#define MIRK4_INTERPOLATOR_HPP

#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <limits>
#include <Eigen/Dense>

#include "collocation/methods/MIRK4.hpp"

namespace collocation {namespace methods {
template <typename Scalar, Index RowsAtCompileTime, Index ColsAtCompileTime, Index ParamsAtCompileTime, typename F, typename BC, typename FJ, typename BCJ>
class MIRK4;
}}

namespace interpolators {

using Eigen::last;
using Eigen::all;
using Eigen::Dynamic;
using Eigen::ArrayX;
using Eigen::Array;
using Eigen::ArrayBase;
// using Eigen::Index;

// template <typename Derived1, typename Derived2, typename Derived3 = Derived2>
template <typename _Scalar, Index _RowsAtCompileTime, Index _ColsAtCompileTime>
class MIRK4_interpolator {
public:
  // using Scalar = typename Derived2::Scalar;
  // using Index = typename Derived2::Index;
  // static constexpr Index RowsAtCompileTime = Derived2::RowsAtCompileTime;
  // static constexpr Index ColsAtCompileTime = Derived2::ColsAtCompileTime;
  typedef _Scalar Scalar;
  static constexpr Index RowsAtCompileTime = _RowsAtCompileTime;
  static constexpr Index ColsAtCompileTime = _ColsAtCompileTime;

  template <typename Derived1, typename Derived2, typename Derived3>
  MIRK4_interpolator(const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y, const ArrayBase<Derived3> & f) : m_x{x}, m_y{y}, m_f{f}, m_rows{y.rows()}, m_cols{y.cols()} {
    assert(m_x.size() == m_y.cols());
    assert(m_x.size() == m_f.cols());
    assert(m_y.rows() == m_f.rows());
    assert(m_x.size() > 1);
    assert(((m_x.tail(m_cols - 1) - m_x.head(m_cols - 1)) >= 0).all());
  }

  template <Index ParamsAtCompileTime, typename F, typename BC, typename FJ, typename BCJ>
  MIRK4_interpolator(const collocation::methods::MIRK4<_Scalar, _RowsAtCompileTime, _ColsAtCompileTime, ParamsAtCompileTime, F, BC, FJ, BCJ>& mirk4) : m_x{mirk4.m_x}, m_y{mirk4.m_y}, m_f{mirk4.m_f}, m_rows{mirk4.m_rows}, m_cols{mirk4.m_cols} {}

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
    auto yi = m_y(all, i);
    auto yip1 = m_y(all, i+1);
    auto fi = m_f(all, i);
    auto fip1 = m_f(all, i+1);
    Scalar hi = (xip1 - xi);
    Scalar w = (x - xi) / hi;

    // return (1 - w) * (1 - w) * (yi * (1 + 2 * w) + fi * (x - xi)) + w * w * (yip1 * (3 - 2 * w) + hi * fip1 * (w - 1));
    return m_A(w) * yip1 + m_A(1-w) * yi + hi * (m_B(w) * fip1 - m_B(1 - w) * fi);
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
    auto yi = m_y(all, i);
    auto yip1 = m_y(all, i+1);
    auto fi = m_f(all, i);
    auto fip1 = m_f(all, i+1);
    Scalar hi = (xip1 - xi);
    Scalar w = (x - xi) / hi;

    // return 6 * (1 - w) * w * (yip1 - yi) / hi - w * (2 - 3 * w) * fip1 + (1 - w) * (1 - 3 * w) * fi;
    return m_Aprime(w) * (yip1 - yi) / hi + m_Bprime(w) * fip1 + m_Bprime(1 - w) * fi;
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

  friend std::ostream& operator<<(std::ostream & os, const MIRK4_interpolator & sol) {
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

  Index m_rows;
  Index m_cols;

  Scalar m_A(Scalar w) const {return w * w * (3 - 2 * w);}
  Scalar m_Aprime(Scalar w) const {return 6 * (1 - w) * w;}
  Scalar m_B(Scalar w) const {return w * w * (w - 1);}
  Scalar m_Bprime(Scalar w) const {return w * (3 * w - 2);}
};

template <typename Derived1, typename Derived2, typename Derived3>
MIRK4_interpolator(const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y, const ArrayBase<Derived3> & f) -> MIRK4_interpolator<typename Derived2::Scalar, Derived2::RowsAtCompileTime, Derived2::ColsAtCompileTime>;

template <typename Scalar, Index RowsAtCompileTime, Index ColsAtCompileTime, Index ParamsAtCompileTime, typename F, typename BC, typename FJ, typename BCJ>
MIRK4_interpolator(const collocation::methods::MIRK4<Scalar, RowsAtCompileTime, ColsAtCompileTime, ParamsAtCompileTime, F, BC, FJ, BCJ>& mirk4) -> MIRK4_interpolator<Scalar, RowsAtCompileTime, ColsAtCompileTime>;

} // interpolators
#endif // VECTOR_CUBIC_HERMITE
