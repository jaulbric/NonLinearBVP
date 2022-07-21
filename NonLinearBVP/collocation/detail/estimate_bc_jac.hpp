#ifndef COLLOCATION_ESTIMATE_BC_JAC_HPP
#define COLLOCATION_ESTIMATE_BC_JAC_HPP

namespace nonlinearbvp { namespace collocation { namespace detail {

using Eigen::Array;
using Eigen::ArrayBase;
using Eigen::Dynamic;
using Eigen::Index;

template <Index _RowsAtCompileTime, Index _ParamsAtCompileTime>
struct estimate_bc_jac_traits {
  static constexpr Index RowsAtCompileTime = _RowsAtCompileTime;
  static constexpr Index ParamsAtCompileTime = _ParamsAtCompileTime;
  static constexpr Index BCsAtCompileTime = _RowsAtCompileTime + _ParamsAtCompileTime;
};

template <Index _RowsAtCompileTime>
struct estimate_bc_jac_traits<_RowsAtCompileTime, Dynamic> {
  static constexpr Index RowsAtCompileTime = _RowsAtCompileTime;
  static constexpr Index ParamsAtCompileTime = Dynamic;
  static constexpr Index BCsAtCompileTime = Dynamic;
};

template <Index _ParamsAtCompileTime>
struct estimate_bc_jac_traits<Dynamic, _ParamsAtCompileTime> {
  static constexpr Index RowsAtCompileTime = Dynamic;
  static constexpr Index ParamsAtCompileTime = _ParamsAtCompileTime;
  static constexpr Index BCsAtCompileTime = Dynamic;
};

template <>
struct estimate_bc_jac_traits<Dynamic, Dynamic> {
  static constexpr Index RowsAtCompileTime = Dynamic;
  static constexpr Index ParamsAtCompileTime = Dynamic;
  static constexpr Index BCsAtCompileTime = Dynamic;
};

template <typename BC, typename Scalar, Index _RowsAtCompileTime, Index _ParamsAtCompileTime>
class estimate_bc_jac : public estimate_bc_jac_traits<_RowsAtCompileTime, _ParamsAtCompileTime> {
public:
  using Traits = estimate_bc_jac_traits<_RowsAtCompileTime, _ParamsAtCompileTime>;
  using Traits::RowsAtCompileTime;
  using Traits::ParamsAtCompileTime;
  using Traits::BCsAtCompileTime;

  estimate_bc_jac(BC& bc) : m_bc{bc} {};
  ~estimate_bc_jac() = default;

#ifdef COLLOCATION_BC_JACOBIAN_RETURN
  template <typename Derived1, typename Derived2, typename Derived3>
  std::tuple<Array<Scalar, BCsAtCompileTime, RowsAtCompileTime>, Array<Scalar, BCsAtCompileTime, RowsAtCompileTime>, Array<Scalar, BCsAtCompileTime, ParamsAtCompileTime>> operator()(const ArrayBase<Derived1>& ya, const ArrayBase<Derived2>& yb, const ArrayBase<Derived3>& p) const {
    Index rows = ya.rows();
    Index params = p.size();
    assert(yb.rows() == rows);

    Array<Scalar, BCsAtCompileTime, 1> bc0 = m_bc(ya, yb, p);

    Array<Scalar, BCsAtCompileTime, RowsAtCompileTime> dbc_dya(rows + params, rows);
    Array<Scalar, BCsAtCompileTime, RowsAtCompileTime> dbc_dyb(rows + params, rows);
    Array<Scalar, BCsAtCompileTime, ParamsAtCompileTime> dbc_dp(rows + params, params);

    Array<Scalar, RowsAtCompileTime, 1> y_new(rows);
    Array<Scalar, ParamsAtCompileTime, 1> p_new(params);

    Array<Scalar, BCsAtCompileTime, 1> bc_new(rows + params);
    Array<Scalar, RowsAtCompileTime, 1> h(rows);
    Array<Scalar, ParamsAtCompileTime, 1> hp(params);

    h = sqrt_EPS * (1 + ya.abs());
    for (Index j = 0; j < rows; ++j) {
      y_new = ya;
      y_new(j) += h(j);
      bc_new = m_bc(y_new, yb, p);
      Scalar hj = y_new(j) - ya(j);
      for (Index i = 0; i < rows + params; ++i) {
       dbc_dya(i, j) = (bc_new(i) - bc0(i)) / hj;
      }
    }

    h = sqrt_EPS * (1 + yb.abs());
    for (Index j = 0; j < rows; ++j) {
      y_new = yb;
      y_new(j) += h(j);
      Scalar hj = y_new(j) - yb(j);
      bc_new = m_bc(ya, y_new, p);
      for (int i = 0; i < rows + params; i++) {
        dbc_dyb(i, j) = (bc_new(i) - bc0(i)) / hj;
      }
    }

    hp = sqrt_EPS * (1 + p.abs());
    for (Index j = 0; j < params; ++j) {
      p_new = p;
      p_new(j) += hp(j);
      Scalar hj = p_new(j) - p(j);
      bc_new = m_bc(ya, yb, p_new);
      for (Index i = 0; i < rows + params; ++i) {
        dbc_dp(i, j) = (bc_new(i) - bc0(i)) / hj;
      }
    }

    return std::make_tuple(dbc_dya, dbc_dyb, dbc_dp);
  }
#else
  template <typename Derived1, typename Derived2, typename Derived3, typename Derived4, typename Derived5, typename Derived6>
  void operator()(const ArrayBase<Derived1>& ya, const ArrayBase<Derived2>& yb, const ArrayBase<Derived3>& p, const ArrayBase<Derived4>& dbc_dya, const ArrayBase<Derived5>& dbc_dyb, const ArrayBase<Derived6>& dbc_dp) const {
    Index rows = ya.rows();
    Index params = p.size();
    assert(yb.rows() == rows);

    Array<Scalar, BCsAtCompileTime, 1> bc0(rows + params);
    m_bc(ya, yb, p, bc0);

    Array<Scalar, RowsAtCompileTime, 1> y_new(rows);
    Array<Scalar, ParamsAtCompileTime, 1> p_new(params);

    Array<Scalar, BCsAtCompileTime, 1> bc_new(rows + params);
    Array<Scalar, RowsAtCompileTime, 1> h(rows);
    Array<Scalar, ParamsAtCompileTime, 1> hp(params);

    h = sqrt_EPS * (1 + ya.abs());
    for (Index j = 0; j < rows; ++j) {
      y_new = ya;
      y_new(j) += h(j);
      m_bc(y_new, yb, p, bc_new);
      Scalar hj = y_new(j) - ya(j);
      for (Index i = 0; i < rows + params; ++i) {
        const_cast<ArrayBase<Derived4>&>(dbc_dya)(i, j) = (bc_new(i) - bc0(i)) / hj;
      }
    }

    h = sqrt_EPS * (1 + yb.abs());
    for (Index j = 0; j < rows; ++j) {
      y_new = yb;
      y_new(j) += h(j);
      Scalar hj = y_new(j) - yb(j);
      m_bc(ya, y_new, p, bc_new);
      for (int i = 0; i < rows + params; i++) {
        const_cast<ArrayBase<Derived5>&>(dbc_dyb)(i, j) = (bc_new(i) - bc0(i)) / hj;
      }
    }

    hp = sqrt_EPS * (1 + p.abs());
    for (Index j = 0; j < params; ++j) {
      p_new = p;
      p_new(j) += hp(j);
      Scalar hj = p_new(j) - p(j);
      m_bc(ya, yb, p_new, bc_new);
      for (Index i = 0; i < rows + params; ++i) {
        const_cast<ArrayBase<Derived6>&>(dbc_dp)(i, j) = (bc_new(i) - bc0(i)) / hj;
      }
    }
  }
#endif

private:
  BC& m_bc;
  // static constexpr Real sqrt_EPS = sqrt(boost::math::tools::epsilon<Real>());
  // static constexpr Real sqrt_EPS = sqrt(std::numeric_limits<Real>::epsilon());
  const Scalar sqrt_EPS = sqrt(std::numeric_limits<Scalar>::epsilon());
};

// Same thing but with no unknown parameters
template <typename BC, typename Scalar, Index _RowsAtCompileTime>
class estimate_bc_jac<BC, Scalar, _RowsAtCompileTime, 0> : public estimate_bc_jac_traits<_RowsAtCompileTime, 0> {
public:
  using Traits = estimate_bc_jac_traits<_RowsAtCompileTime, 0>;
  using Traits::RowsAtCompileTime;
  using Traits::ParamsAtCompileTime;
  using Traits::BCsAtCompileTime;
  estimate_bc_jac(BC& bc) : m_bc{bc} {};
  ~estimate_bc_jac() = default;

#ifdef COLLOCATION_BC_JACOBIAN_RETURN
  template <typename Derived1, typename Derived2>
  std::tuple<Array<Scalar, RowsAtCompileTime, RowsAtCompileTime>, Array<Scalar, RowsAtCompileTime, RowsAtCompileTime>> operator()(const ArrayBase<Derived1>& ya, const ArrayBase<Derived2>& yb) const {
    Index rows = ya.rows();
    assert(yb.rows() == rows);

    Array<Scalar, BCsAtCompileTime, 1> bc0 = m_bc(ya, yb);

    Array<Scalar, BCsAtCompileTime, RowsAtCompileTime> dbc_dya(rows, rows);
    Array<Scalar, BCsAtCompileTime, RowsAtCompileTime> dbc_dyb(rows, rows);

    Array<Scalar, RowsAtCompileTime, 1> y_new(rows);
    Array<Scalar, BCsAtCompileTime, 1> bc_new(rows);
    Array<Scalar, RowsAtCompileTime, 1> h(rows);

    h = sqrt_EPS * (1 + ya.abs());
    for (Index j = 0; j < rows; ++j) {
      y_new = ya;
      y_new(j) += h(j);
      bc_new = m_bc(y_new, yb);
      Scalar hj = y_new(j) - ya(j);
      dbc_dya.col(j) = (bc_new - bc0) / hj;
    }

    h = sqrt_EPS * (1 + yb.abs());
    for (Index j = 0; j < rows; ++j) {
      y_new = yb;
      y_new(j) += h(j);
      Scalar hj = y_new(j) - yb(j);
      bc_new = m_bc(ya, y_new);
      dbc_dyb.col(j) = (bc_new - bc0) / hj;
    }

    return std::make_tuple(dbc_dya, dbc_dyb);
  }
#else
  template <typename Derived1, typename Derived2, typename Derived3, typename Derived4>
  void operator()(const ArrayBase<Derived1>& ya, const ArrayBase<Derived2>& yb, const ArrayBase<Derived3>& dbc_dya, const ArrayBase<Derived4>& dbc_dyb) const {
    Index rows = ya.rows();
    assert(yb.rows() == rows);

    Array<Scalar, BCsAtCompileTime, 1> bc0;
    m_bc(ya, yb, bc0);

    Array<Scalar, RowsAtCompileTime, 1> y_new(rows);
    Array<Scalar, BCsAtCompileTime, 1> bc_new(rows);
    Array<Scalar, RowsAtCompileTime, 1> h(rows);

    h = sqrt_EPS * (1 + ya.abs());
    for (Index j = 0; j < rows; ++j) {
      y_new = ya;
      y_new(j) += h(j);
      m_bc(y_new, yb, bc_new);
      Scalar hj = y_new(j) - ya(j);
      const_cast<ArrayBase<Derived3>&>(dbc_dya).col(j) = (bc_new - bc0) / hj;
    }

    h = sqrt_EPS * (1 + yb.abs());
    for (Index j = 0; j < rows; ++j) {
      y_new = yb;
      y_new(j) += h(j);
      Scalar hj = y_new(j) - yb(j);
      bc_new = m_bc(ya, y_new);
      const_cast<ArrayBase<Derived4>&>(dbc_dyb).col(j) = (bc_new - bc0) / hj;
    }
  }
#endif

private:
  BC& m_bc;
  // static constexpr Real sqrt_EPS = sqrt(boost::math::tools::epsilon<Real>());
  // static constexpr Real sqrt_EPS = sqrt(std::numeric_limits<Real>::epsilon());
  const Scalar sqrt_EPS = sqrt(std::numeric_limits<Scalar>::epsilon());
};

} // namespace detail
} // namespace collocation
} // namespace nonlinearbvp

#endif // COLLOCATION_DETAIL_ESTIMATE_BC_JAC
