#ifndef COLLOCATION_ESTIMATE_FUN_JAC_HPP
#define COLLOCATION_ESTIMATE_FUN_JAC_HPP

namespace nonlinearbvp { namespace collocation { namespace detail {

template <typename F, typename Scalar, Index RowsAtCompileTime, Index ParamsAtCompileTime, bool Singular>
class estimate_fun_jac {};

// Estimates the jacobian of a function using forward finite differences
template <class F, typename _Scalar, Index _RowsAtCompileTime, Index _ParamsAtCompileTime>
class estimate_fun_jac<F, _Scalar, _RowsAtCompileTime, _ParamsAtCompileTime, true> {
public:
  typedef F Functor;
  typedef _Scalar Scalar;
  static constexpr Index RowsAtCompileTime = _RowsAtCompileTime;
  static constexpr Index ParamsAtCompileTime = _ParamsAtCompileTime;

  template <typename Derived1, typename Derived2>
  estimate_fun_jac(F& fun, const Scalar& a, const MatrixBase<Derived1>& S, const MatrixBase<Derived2>& D) : m_fun{fun}, m_a{a}, m_S{S}, m_D{D} {};

  ~estimate_fun_jac() = default;

  template <typename Derived1, typename Derived2, typename Derived3>
  void operator()(const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y, const ArrayBase<Derived3>& p, const Tensor<Scalar, 3>& df_dy, const Tensor<Scalar, 3>& df_dp) const {
    Index rows = y.rows();
    Index cols = y.cols();
    Index params = p.size();
    assert(x.size() == cols);

    Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> dy = sqrt_EPS * (1 + y.abs());
    Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> y_new(rows, cols);
# ifndef COLLOCATION_FUNCTION_RETURN
    Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> f_new(rows, cols);
# endif

# ifdef COLLOCATION_FUNCTION_RETURN
    Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> f0 = m_fun(x, y, p);
# else
    Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> f0(rows, cols);
    m_fun(x, y, p, f0);
# endif

    for (Index j = 0; j < rows; ++j) {
      y_new = y;
      y_new.row(j) += dy.row(j);
# ifdef COLLOCATION_FUNCTION_RETURN
      Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> f_new = m_fun(x, y_new, p);
# else
      m_fun(x, y_new, p, f_new);
# endif
      for (Index i = 0; i < rows; ++i) {
        for (Index k = 0; k < cols; ++k) {
          const_cast<Tensor<Scalar, 3>&>(df_dy)(i, j, k) = (f_new(i, k) - f0(i, k)) / (y_new(j, k) - y(j, k));
        }
      }
    }

    if (x(0) == m_a) {
      Map<Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>> m(const_cast<Tensor<Scalar, 3>&>(df_dy).data(), rows, rows);
      m = m_D * m;
    }
    else {
      Map<Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>> m(const_cast<Tensor<Scalar, 3>&>(df_dy).data(), rows, rows);
      m.noalias() += m_S / (x(0) - m_a);
    }
    for (Index idx = 1; idx < cols; ++idx) {
      Map<Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>> m(const_cast<Tensor<Scalar, 3>&>(df_dy).data() + rows * rows * idx, rows, rows);
      m.noalias() += m_S / (x(idx) - m_a);
    }

    Array<Scalar, ParamsAtCompileTime, 1> dp = sqrt_EPS * (1 + p.abs());
    Array<Scalar, ParamsAtCompileTime, 1> p_new;

    for (Index j = 0; j < params; ++j) {
      p_new = p;
      p_new(j) += dp(j);
# ifdef COLLOCATION_FUNCTION_RETURN
      Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> f_new = m_fun(x, y, p_new);
# else
      m_fun(x, y, p_new, f_new);
# endif
      for (Index i = 0; i < rows; ++i) {
        for (Index k = 0; k < cols; ++k) {
          const_cast<Tensor<Scalar, 3>&>(df_dp)(i, j, k) = (f_new(i, k) - f0(i, k)) / (p_new(j) - p(j));
        }
      }
    }
  }

  template <typename Derived1, typename Derived2, typename Derived3>
  std::pair<Tensor<Scalar, 3>, Tensor<Scalar, 3>> operator()(const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y, const ArrayBase<Derived3>& p) const {
    Tensor<Scalar, 3> df_dy(y.rows(), y.rows(), y.cols());
    Tensor<Scalar, 3> df_dp(y.rows(), p.size(), y.cols());
    this->operator()(x, y, p, df_dy, df_dp);
    return std::make_pair(df_dy, df_dp);
  }

private:
  const F& m_fun;
  const Scalar m_a;
  const Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime> m_S;
  const Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime> m_D;
  // static constexpr Scalar sqrt_EPS = sqrt(boost::math::tools::epsilon<Scalar>());
  // static constexpr Scalar sqrt_EPS = sqrt(std::numeric_limits<Scalar>::epsilon());
  const Scalar sqrt_EPS = sqrt(std::numeric_limits<Scalar>::epsilon());
};

template <class F, typename _Scalar, Index _RowsAtCompileTime, Index _ParamsAtCompileTime>
class estimate_fun_jac<F, _Scalar, _RowsAtCompileTime, _ParamsAtCompileTime, false> {
public:
  typedef F Functor;
  typedef _Scalar Scalar;
  static constexpr Index RowsAtCompileTime = _RowsAtCompileTime;
  static constexpr Index ParamsAtCompileTime = _ParamsAtCompileTime;

  estimate_fun_jac(F& fun) : m_fun{fun} {};
  ~estimate_fun_jac() = default;

  template <typename Derived1, typename Derived2, typename Derived3>
  void operator()(const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y, const ArrayBase<Derived3>& p, const Tensor<Scalar, 3>& df_dy, const Tensor<Scalar, 3>& df_dp) const {
    Index rows = y.rows();
    Index cols = y.cols();
    Index params = p.size();
    assert(x.size() == cols);

    Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> dy = sqrt_EPS * (1 + y.abs());
    Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> y_new(rows, cols);
# ifndef COLLOCATION_FUNCTION_RETURN
    Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> f_new(rows, cols);
# endif

# ifdef COLLOCATION_FUNCTION_RETURN
    Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> f0 = m_fun(x, y, p);
# else
    Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> f0(rows, cols);
    m_fun(x, y, p, f0);
# endif

    for (Index j = 0; j < rows; ++j) {
      y_new = y;
      y_new.row(j) += dy.row(j);
# ifdef COLLOCATION_FUNCTION_RETURN
      Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> f_new = m_fun(x, y_new, p);
# else
      m_fun(x, y_new, p, f_new);
# endif
      for (Index i = 0; i < rows; ++i) {
        for (Index k = 0; k < cols; ++k) {
          const_cast<Tensor<Scalar, 3>&>(df_dy)(i, j, k) = (f_new(i, k) - f0(i, k)) / (y_new(j, k) - y(j, k));
        }
      }
    }

    Array<Scalar, ParamsAtCompileTime, 1> hp = sqrt_EPS * (1 + p.abs());
    Array<Scalar, ParamsAtCompileTime, 1> p_new;

    for (Index j = 0; j < params; ++j) {
      p_new = p;
      p_new(j) += hp(j);
# ifdef COLLOCATION_FUNCTION_RETURN
      Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> f_new = m_fun(x, y, p_new);
# else
      m_fun(x, y, p_new, f_new);
# endif
      for (Index i = 0; i < rows; ++i) {
        for (Index l = 0; l < cols; ++l) {
          const_cast<Tensor<Scalar, 3>&>(df_dp)(i, j, l) = (f_new(i, l) - f0(i, l)) / (p_new(j) - p(j));
        }
      }
    }
      // return std::make_pair(df_dy, df_dp);
  }

  template <typename Derived1, typename Derived2, typename Derived3>
  std::pair<Tensor<Scalar, 3>, Tensor<Scalar, 3>> operator()(const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y, const ArrayBase<Derived3>& p) const {
    Tensor<Scalar, 3> df_dy(y.rows(), y.rows(), y.cols());
    Tensor<Scalar, 3> df_dp(y.rows(), p.size(), y.cols());
    this->operator()(x, y, p, df_dy, df_dp);
    return std::make_pair(df_dy, df_dp);
  }

private:
  F& m_fun;
  // static constexpr Scalar sqrt_EPS = sqrt(boost::math::tools::epsilon<Scalar>());
  // static constexpr Scalar sqrt_EPS = sqrt(std::numeric_limits<Scalar>::epsilon());
  const Scalar sqrt_EPS = sqrt(std::numeric_limits<Scalar>::epsilon());
};

template <class F, typename _Scalar, Index _RowsAtCompileTime>
class estimate_fun_jac<F, _Scalar, _RowsAtCompileTime, 0, true> {
public:
  typedef F Functor;
  typedef _Scalar Scalar;
  static constexpr Index RowsAtCompileTime = _RowsAtCompileTime;
  static constexpr Index ParamsAtCompileTime = 0;

  template <typename Derived1, typename Derived2>
  estimate_fun_jac(F& fun, const Scalar& a, const MatrixBase<Derived1>& S, const MatrixBase<Derived2>& D) : m_fun{fun}, m_a{a}, m_S{S}, m_D{D} {};

  ~estimate_fun_jac() = default;

  template <typename Derived1, typename Derived2>
  void operator()(const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y, const Tensor<Scalar, 3>& df_dy) const {
    Index rows = y.rows();
    Index cols = y.cols();
    assert(x.size() == cols);

    Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> dy = sqrt_EPS * (1 + y.abs());
    Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> y_new(rows, cols);
# ifndef COLLOCATION_FUNCTION_RETURN
    Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> f_new(rows, cols);
# endif

# ifdef COLLOCATION_FUNCTION_RETURN
    Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> f0 = m_fun(x, y);
# else
    Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> f0(rows, cols);
    m_fun(x, y, f0);
# endif

    for (Index j = 0; j < rows; ++j) {
      y_new = y;
      y_new.row(j) += dy.row(j);
# ifdef COLLOCATION_FUNCTION_RETURN
      Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> f_new = m_fun(x, y_new);
# else
      m_fun(x, y_new, f_new);
# endif
      for (Index i = 0; i < rows; ++i) {
        for (Index k = 0; k < cols; ++k) {
          const_cast<Tensor<Scalar, 3>&>(df_dy)(i, j, k) = (f_new(i, k) - f0(i, k)) / (y_new(j, k) - y(j, k));
        }
      }
    }

    if (x(0) == m_a) {
      Map<Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>> m(const_cast<Tensor<Scalar, 3>&>(df_dy).data(), rows, rows);
      m = m_D * m;
    }
    else {
      Map<Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>> m(const_cast<Tensor<Scalar, 3>&>(df_dy).data(), rows, rows);
      m.noalias() += m_S / (x(0) - m_a);
    }
    for (Index idx = 1; idx < cols; ++idx) {
      Map<Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>> m(const_cast<Tensor<Scalar, 3>&>(df_dy).data() + rows * rows * idx, rows, rows);
      m.noalias() += m_S / (x(idx) - m_a);
    }
  }

  template <typename Derived1, typename Derived2>
  Tensor<Scalar, 3> operator()(const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y) const {
    Tensor<Scalar, 3> df_dy(y.rows(), y.rows(), y.cols());
    this->operator()(x, y, df_dy);
    return df_dy;
  }

private:
  F& m_fun;
  const Scalar m_a;
  const Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime> m_S;
  const Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime> m_D;
  // static constexpr Scalar sqrt_EPS = sqrt(boost::math::tools::epsilon<Scalar>());
  // static constexpr Scalar sqrt_EPS = sqrt(std::numeric_limits<Scalar>::epsilon());
  const Scalar sqrt_EPS = sqrt(std::numeric_limits<Scalar>::epsilon());
};

template <class F, typename _Scalar, Index _RowsAtCompileTime>
class estimate_fun_jac<F, _Scalar, _RowsAtCompileTime, 0, false> {
public:
  typedef F Functor;
  typedef _Scalar Scalar;
  static constexpr Index RowsAtCompileTime = _RowsAtCompileTime;
  static constexpr Index ParamsAtCompileTime = 0;

  estimate_fun_jac(F& fun) : m_fun{fun} {};

  ~estimate_fun_jac() = default;

  template <typename Derived1, typename Derived2>
  void operator()(const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y, const Tensor<Scalar, 3>& df_dy) const {
    Index rows = y.rows();
    Index cols = y.cols();
    assert(x.size() == cols);

    Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> dy = sqrt_EPS * (1 + y.abs());
    Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> y_new(rows, cols);
# ifndef COLLOCATION_FUNCTION_RETURN
    Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> f_new(rows, cols);
# endif

# ifdef COLLOCATION_FUNCTION_RETURN
    Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> f0 = m_fun(x, y);
# else
    Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> f0(rows, cols);
    m_fun(x, y, f0);
# endif

    for (Index j = 0; j < rows; ++j) {
      y_new = y;
      y_new.row(j) += dy.row(j);
# ifdef COLLOCATION_FUNCTION_RETURN
      Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> f_new = m_fun(x, y_new);
# else
      m_fun(x, y_new, f_new);
# endif
      for (Index i = 0; i < rows; ++i) {
        for (Index k = 0; k < cols; ++k) {
          const_cast<Tensor<Scalar, 3>&>(df_dy)(i, j, k) = (f_new(i, k) - f0(i, k)) / (y_new(j, k) - y(j, k));
        }
      }
    }
  }

  template <typename Derived1, typename Derived2>
  Tensor<Scalar, 3> operator()(const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y) const {
    Tensor<Scalar, 3> df_dy(y.rows(), y.rows(), y.cols());
    this->operator()(x, y, df_dy);
    return df_dy;
  }

private:
  F& m_fun;
  // static constexpr Scalar sqrt_EPS = sqrt(boost::math::tools::epsilon<Scalar>());
  // static constexpr Scalar sqrt_EPS = sqrt(std::numeric_limits<Scalar>::epsilon());
  const Scalar sqrt_EPS = sqrt(std::numeric_limits<Scalar>::epsilon());
};

} // namespace detail
} // namespace collocation
} // namespace nonlinearbvp

#endif // COLLOCATION_DETAIL_ESTIMATE_FUN_JAC_HPP
