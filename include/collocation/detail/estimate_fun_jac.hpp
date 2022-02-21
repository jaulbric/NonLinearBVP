#ifndef COLLOCATION_ESTIMATE_FUN_JAC_HPP
#define COLLOCATION_ESTIMATE_FUN_JAC_HPP

namespace collocation { namespace detail {

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

    template <typename Derived1, typename Derived2, typename Derived3, typename Derived4>
    std::pair<Tensor<Scalar, 3>, Tensor<Scalar, 3>> operator()(const ArrayBase<Derived1>& x,
                                                               const ArrayBase<Derived2>& y,
                                                               const ArrayBase<Derived3>& p,
                                                               const ArrayBase<Derived4>& f0) {
      Index rows = y.rows();
      Index cols = y.cols();
      Index params = p.size();
      assert(x.size() == cols);

      Tensor<Scalar, 3> df_dy(rows, rows, cols);
      Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> dy = sqrt_EPS * (1 + y.abs());
      Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> y_new(rows, cols);
      Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> f_new(rows, cols);

      for (Index j = 0; j < rows; ++j) {
        y_new = y;
        y_new.row(j) += dy.row(j);
        f_new = m_fun(x, y_new, p);
        for (Index i = 0; i < rows; ++i) {
          for (Index k = 0; k < cols; ++k) {
            df_dy(i, j, k) = (f_new(i, k) - f0(i, k)) / (y_new(j, k) - y(j, k));
          }
        }
      }

      if (x(0) == m_a) {
        Map<Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>> m(df_dy.data(), rows, rows);
        m = m_D * m;
      }
      else {
        Map<Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>> m(df_dy.data(), rows, rows);
        m += m_S / (x(0) - m_a);
      }
      for (Index idx = 1; idx < cols; ++idx) {
        Map<Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>> m(df_dy.data() + rows * rows * idx);
        m += m_S / (x(idx) - m_a);
      }

      Tensor<Scalar, 3> df_dp(rows, params, cols);
      Array<Scalar, ParamsAtCompileTime, 1> dp = sqrt_EPS * (1 + p.abs());
      Array<Scalar, ParamsAtCompileTime, 1> p_new;

      for (Index j = 0; j < params; ++j) {
        p_new = p;
        p_new(j) += dp(j);
        f_new = m_fun(x, y, p_new);
        for (Index i = 0; i < rows; ++i) {
          for (Index k = 0; k < cols; ++k) {
            df_dp(i, j, k) = (f_new(i, k) - f0(i, k)) / (p_new(j) - p(j));
          }
        }
      }
      return std::make_pair(df_dy, df_dp);
  };

  template <typename Derived1, typename Derived2, typename Derived3>
  inline std::pair<Tensor<Scalar, 3>, Tensor<Scalar, 3>> operator()(const ArrayBase<Derived1>& x,
                                                                    const ArrayBase<Derived2>& y,
                                                                    const ArrayBase<Derived3>& p) {
    // auto f0 = fun_(x, y, p);
    return (*this)(x, y, p, m_fun(x, y, p));
  };

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

    template <typename Derived1, typename Derived2, typename Derived3, typename Derived4>
    std::pair<Tensor<Scalar, 3>, Tensor<Scalar, 3>> operator()(const ArrayBase<Derived1>& x,
                                                               const ArrayBase<Derived2>& y,
                                                               const ArrayBase<Derived3>& p,
                                                               const ArrayBase<Derived4>& f0) {
      Index rows = y.rows();
      Index cols = y.cols();
      Index params = p.size();
      assert(x.size() == cols);

      Tensor<Scalar, 3> df_dy(rows, rows, cols);
      Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> dy = sqrt_EPS * (1 + y.abs());
      Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> y_new(rows, cols);
      Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> f_new(rows, cols);

      for (Index j = 0; j < rows; ++j) {
        y_new = y;
        y_new.row(j) += dy.row(j);
        f_new = m_fun(x, y_new, p);
        for (Index i = 0; i < rows; ++i) {
          for (Index k = 0; k < cols; ++k) {
            df_dy(i, j, k) = (f_new(i, k) - f0(i, k)) / (y_new(j, k) - y(j, k));
          }
        }
      }

      Tensor<Scalar, 3> df_dp(rows, params, cols);
      Array<Scalar, ParamsAtCompileTime, 1> hp = sqrt_EPS * (1 + p.abs());
      Array<Scalar, ParamsAtCompileTime, 1> p_new;

      for (Index j = 0; j < params; ++j) {
        p_new = p;
        p_new(j) += hp(j);
        f_new = m_fun(x, y, p_new);
        for (Index i = 0; i < rows; ++i) {
          for (Index l = 0; l < cols; ++l) {
            df_dp(i, j, l) = (f_new(i, l) - f0(i, l)) / (p_new(j) - p(j));
          }
        }
      }
      return std::make_pair(df_dy, df_dp);
  };

  template <typename Derived1, typename Derived2, typename Derived3>
  inline std::pair<Tensor<Scalar, 3>, Tensor<Scalar, 3>> operator()(const ArrayBase<Derived1>& x,
                                                                    const ArrayBase<Derived2>& y,
                                                                    const ArrayBase<Derived3>& p) {
    // auto f0 = fun_(x, y, p);
    return (*this)(x, y, p, m_fun(x, y, p));
  };

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

    template <typename Derived1, typename Derived2, typename Derived3>
    Tensor<Scalar, 3> operator()(const ArrayBase<Derived1>& x,
                                 const ArrayBase<Derived2>& y,
                                 const ArrayBase<Derived3>& f0) {
      Index rows = y.rows();
      Index cols = y.cols();
      assert(x.size() == cols);

      Tensor<Scalar, 3> df_dy(rows, rows, cols);
      Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> dy = sqrt_EPS * (1 + y.abs());
      Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> y_new(rows, cols);
      Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> f_new(rows, cols);

      for (Index j = 0; j < rows; ++j) {
        y_new = y;
        y_new.row(j) += dy.row(j);
        f_new = m_fun(x, y_new);
        for (Index i = 0; i < rows; ++i) {
          for (Index k = 0; k < cols; ++k) {
            df_dy(i, j, k) = (f_new(i, k) - f0(i, k)) / (y_new(j, k) - y(j, k));
          }
        }
      }

      if (x(0) == m_a) {
        Map<Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>> m(df_dy.data(), rows, rows);
        m = m_D * m;
      }
      else {
        Map<Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>> m(df_dy.data(), rows, rows);
        m += m_S / (x(0) - m_a);
      }
      for (Index idx = 1; idx < cols; ++idx) {
        Map<Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>> m(df_dy.data() + rows * rows * idx);
        m += m_S / (x(idx) - m_a);
      }

      return df_dy;
  };

  template <typename Derived1, typename Derived2>
  inline Tensor<Scalar, 3> operator()(const ArrayBase<Derived1>& x,
                                      const ArrayBase<Derived2>& y) {
    return (*this)(x, y, m_fun(x, y));
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

    template <typename Derived1, typename Derived2, typename Derived3>
    Tensor<Scalar, 3> operator()(const ArrayBase<Derived1>& x,
                                 const ArrayBase<Derived2>& y,
                                 const ArrayBase<Derived3>& f0) {
      Index rows = y.rows();
      Index cols = y.cols();
      assert(x.size() == cols);

      Tensor<Scalar, 3> df_dy(rows, rows, cols);
      Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> dy = sqrt_EPS * (1 + y.abs());
      Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> y_new(rows, cols);
      Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> f_new(rows, cols);

      for (Index j = 0; j < rows; ++j) {
        y_new = y;
        y_new.row(j) += dy.row(j);
        f_new = m_fun(x, y_new);
        for (Index i = 0; i < rows; ++i) {
          for (Index k = 0; k < cols; ++k) {
            df_dy(i, j, k) = (f_new(i, k) - f0(i, k)) / (y_new(j, k) - y(j, k));
          }
        }
      }

      return df_dy;
  };

  template <typename Derived1, typename Derived2>
  inline Tensor<Scalar, 3> operator()(const ArrayBase<Derived1>& x,
                                      const ArrayBase<Derived2>& y) {
    return (*this)(x, y, m_fun(x, y));
  }

  private:
    F& m_fun;
    // static constexpr Scalar sqrt_EPS = sqrt(boost::math::tools::epsilon<Scalar>());
    // static constexpr Scalar sqrt_EPS = sqrt(std::numeric_limits<Scalar>::epsilon());
    const Scalar sqrt_EPS = sqrt(std::numeric_limits<Scalar>::epsilon());
};

} // namespace detail
} // namespace collocation

#endif // COLLOCATION_DETAIL_ESTIMATE_FUN_JAC_HPP
