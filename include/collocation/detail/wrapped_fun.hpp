#ifndef COLLOCATION_WRAPPED_FUN_HPP
#define COLLOCATION_WRAPPED_FUN_HPP

#include <Eigen/Dense>
#include <Eigen/CXX11/Tensor>
#include <utility>

using Eigen::Array;
using Eigen::Matrix;
using Eigen::ArrayBase;
using Eigen::all;
using Eigen::seq;
using Eigen::last;
using Eigen::Tensor;
using Eigen::TensorMap;
using Eigen::Index;
using Eigen::IndexPair;
using Eigen::array;

namespace collocation { namespace detail {

template <typename F, typename Scalar, Index RowsAtCompileTime, Index ParamsAtCompileTime>
class BVP_wrapped_fun {
  public:
    template <typename Derived1, typename Derived2>
    BVP_wrapped_fun(F& fun, const Scalar &a, const MatrixBase<Derived1>& S, const MatrixBase<Derived2>& D) : m_fun{fun}, m_a{a}, m_S{S}, m_D{D} {
      static_assert(Derived1::RowsAtCompileTime == Derived1::ColsAtCompileTime, "S is not a square matrix.");
      static_assert(Derived2::RowsAtCompileTime == Derived2::ColsAtCompileTime, "D is not a square matrix.");
      static_assert(Derived1::SizeAtCompileTime == Derived2::SizeAtCompileTime, "S and D do not have the same size.");
      static_assert(Derived1::RowsAtCompileTime == RowsAtCompileTime, "template parameter RowsAtCompileTime != S.rows()");
    };

    ~BVP_wrapped_fun() = default;

    template <typename Derived1, typename Derived2, typename Derived3>
    Array<typename Derived2::Scalar, Derived2::RowsAtCompileTime, Derived2::ColsAtCompileTime> operator()(const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y, const ArrayBase<Derived3>& p) const {
      Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> f = m_fun(x, y, p);
      if (x(0) == m_a) {
        f.col(0).matrix() = m_D * f.col(0).matrix();
      }
      else {
        f.col(0).matrix() += m_S * y.col(0).matrix() / (x(0) - m_a);
      }
      for (Index idx = 1; idx < x.size(); ++idx) {
        Map<Matrix<Scalar, RowsAtCompileTime, 1>> m(f.data() + y.rows() * idx, y.rows(), 1);
        m += m_S * y.col(idx).matrix() / (x(idx) - m_a);
      }
      return f;
    };

  private:
    F& m_fun;
    const Scalar m_a;
    const Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime> m_S;
    const Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime> m_D;
};

template <typename F, typename Scalar, Index RowsAtCompileTime>
class BVP_wrapped_fun<F, Scalar, RowsAtCompileTime, 0> {
  public:
    template <typename Derived1, typename Derived2>
    BVP_wrapped_fun(F& fun, const Scalar &a, const MatrixBase<Derived1>& S, const MatrixBase<Derived2>& D) : m_fun{fun}, m_a{a}, m_S{S}, m_D{D} {
      static_assert(Derived1::RowsAtCompileTime == Derived1::ColsAtCompileTime, "S is not a square matrix.");
      static_assert(Derived2::RowsAtCompileTime == Derived2::ColsAtCompileTime, "D is not a square matrix.");
      static_assert(Derived1::SizeAtCompileTime == Derived2::SizeAtCompileTime, "S and D do not have the same size.");
      static_assert(Derived1::RowsAtCompileTime == RowsAtCompileTime, "template parameter RowsAtCompileTime != S.rows()");
    };

    ~BVP_wrapped_fun() = default;

    template <typename Derived1, typename Derived2>
    Array<typename Derived2::Scalar, Derived2::RowsAtCompileTime, Derived2::ColsAtCompileTime> operator()(const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y) const {
      Array<Scalar, RowsAtCompileTime, Derived2::ColsAtCompileTime> f = m_fun(x, y);
      // f = m_fun(x, y);
      if (x(0) == m_a) {
        f.col(0).matrix() = m_D * f.col(0).matrix();
      }
      else {
        f.col(0).matrix() += m_S * y.col(0).matrix() / (x(0) - m_a);
      }
      for (Index idx = 1; idx < x.size(); ++idx) {
        Map<Matrix<Scalar, RowsAtCompileTime, 1>> m(f.data() + y.rows() * idx, y.rows(), 1);
        m += m_S * y.col(idx).matrix() / (x(idx) - m_a);
      }
      return f;
    };

  private:
    F& m_fun;
    const Scalar m_a;
    const Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime> m_S;
    const Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime> m_D;
};

template <typename FJ, typename Scalar, Index RowsAtCompileTime, Index ParamsAtCompileTime>
class BVP_wrapped_fun_jac {
  public:
    template <typename Derived1, typename Derived2>
    BVP_wrapped_fun_jac(FJ& fun_jac, const Scalar& a, const MatrixBase<Derived1>& S, const MatrixBase<Derived2>& D) : m_fun_jac{fun_jac}, m_a{a}, m_S{S}, m_D{D} {
      static_assert(Derived1::RowsAtCompileTime == Derived1::ColsAtCompileTime, "S is not a square matrix.");
      static_assert(Derived2::RowsAtCompileTime == Derived2::ColsAtCompileTime, "D is not a square matrix.");
      static_assert(Derived1::SizeAtCompileTime == Derived2::SizeAtCompileTime, "S and D do not have the same size.");
      static_assert(Derived1::RowsAtCompileTime == RowsAtCompileTime, "template parameter RowsAtCompileTime != S.rows()");
    };
    ~BVP_wrapped_fun_jac() = default;

    template <typename Derived1, typename Derived2, typename Derived3>
    std::pair<Tensor<Scalar, 3>, Tensor<Scalar, 3>> operator()(const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y, const ArrayBase<Derived3>& p) {
      Index rows = y.rows();
      Index cols = y.cols();
      Index params = p.size();
      assert(x.size() == cols);
      Tensor<Scalar, 3> df_dy(rows, rows, cols);
      Tensor<Scalar, 3> df_dp(rows, params, cols);
      std::tie(df_dy, df_dp) = m_fun_jac(x, y, p);
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
      return std::make_pair(df_dy, df_dp);
    }
  private:
    FJ& m_fun_jac;
    const Scalar m_a;
    const Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime> m_S;
    const Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime> m_D;
};

template <typename FJ, typename Scalar, Index RowsAtCompileTime>
class BVP_wrapped_fun_jac<FJ, Scalar, RowsAtCompileTime, 0> {
  public:
    template <typename Derived1, typename Derived2>
    BVP_wrapped_fun_jac(FJ& fun_jac, const Scalar& a, const MatrixBase<Derived1>& S, const MatrixBase<Derived2>& D) : m_fun_jac{fun_jac}, m_a{a}, m_S{S}, m_D{D} {
      // D_ = TensorMap<TensorFixedSize<Scalar, Sizes<n, n>>>(D.data(), n, n);
    };
    ~BVP_wrapped_fun_jac() = default;

    template <typename Derived1, typename Derived2>
    Tensor<Scalar, 3> operator()(const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y) {
      Index rows = y.rows();
      Index cols = y.cols();
      assert(x.size() == cols);
      Tensor<Scalar, 3> df_dy(rows, rows, cols);
      df_dy = m_fun_jac(x, y);

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
    }
  private:
    FJ& m_fun_jac;
    const Scalar m_a;
    const Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime> m_S;
    const Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime> m_D;
};

} // namespace detail
} // namespace collocation

#endif // COLLOCATION_WRAPPED_FUN_HPP
