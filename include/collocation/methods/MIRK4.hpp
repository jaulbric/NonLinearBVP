#ifndef MIRK4_HPP
#define MIRK4_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/CXX11/Tensor>
#include <cmath>
#include <utility>

#include "MIRK.hpp"
#include "constants/constants.hpp"
#include "constants/MIRK4_constants.hpp"
#include "interpolators/MIRK4_interpolator.hpp"

using Eigen::Array;
using Eigen::ArrayBase;
using Eigen::Matrix;
using Eigen::MatrixBase;
using Eigen::DenseBase;
using Eigen::Dynamic;
using Eigen::Index;
using Eigen::all;
using Eigen::last;
using Eigen::lastN;
using Eigen::seq;
using Eigen::Tensor;
using Eigen::Triplet;

using std::sqrt;

namespace collocation { namespace methods {

using boost::math::constants::half;
using boost::math::constants::three_half;
using boost::math::constants::third;
using boost::math::constants::two_thirds;
using boost::math::constants::three_quarters;
using boost::math::constants::sixth;
using boost::math::constants::eighth;
using boost::math::constants::twelfth;
#define QALPHA(T) boost::math::constants::MIRK4_GLQ_alpha<T>()
#define QA1(T) boost::math::constants::MIRK4_GLQ_A1<T>()
#define QA3(T) boost::math::constants::MIRK4_GLQ_A3<T>()
#define QB1(T) boost::math::constants::MIRK4_GLQ_B1<T>()
#define QB3(T) boost::math::constants::MIRK4_GLQ_B3<T>()
#define QAP1(T) boost::math::constants::MIRK4_GLQ_Ap1<T>()
#define QBP1(T) boost::math::constants::MIRK4_GLQ_Bp1<T>()
#define QBP3(T) boost::math::constants::MIRK4_GLQ_Bp3<T>()
#define QW1(T) boost::math::constants::MIRK4_GLQ_w1<T>()
#define QW2(T) boost::math::constants::MIRK4_GLQ_w2<T>()
#define A13(T) boost::math::constants::MIRK4_A13<T>()
#define A23(T) boost::math::constants::MIRK4_A23<T>()
#define B13(T) boost::math::constants::MIRK4_B13<T>()
#define B23(T) boost::math::constants::MIRK4_B23<T>()


template <typename _Scalar, Index _RowsAtCompileTime, Index _ColsAtCompileTime, Index _ParamsAtCompileTime, typename F, typename BC, typename FJ, typename BCJ>
class MIRK4 : public MIRK<_Scalar, _RowsAtCompileTime, _ColsAtCompileTime, _ParamsAtCompileTime, F, BC, FJ, BCJ> {
public:
  using Base = MIRK<_Scalar, _RowsAtCompileTime, _ColsAtCompileTime, _ParamsAtCompileTime, F, BC, FJ, BCJ>;
  using Scalar = typename Base::Scalar;
  using Base::RowsAtCompileTime;
  using Base::ColsAtCompileTime;
  using Base::SizeAtCompileTime;
  using Base::ParamsAtCompileTime;
  using Base::IntervalsAtCompileTime;
  using Base::ResiduesAtCompileTime;
  using Base::BCsAtCompileTime;
  //
  // using Functor = typename Base::Functor;
  // using BCFunctor = typename Base::BCFunctor;
  // using FunctorJacobian = typename Base::FunctorJacobian;
  // using BCFunctorJacobian = typename Base::BCFunctorJacobian;

private:
  using Base::m_fun;
  using Base::m_bc;
  using Base::m_fun_jac;
  using Base::m_bc_jac;
  using Base::m_rows;
  using Base::m_cols;
  using Base::m_params;
  using Base::m_x;
  using Base::m_h;
  using Base::m_y;
  using Base::m_f;
  using Base::m_residues;
  using Base::m_residuals;
  using Base::m_i_jac;
  using Base::m_j_jac;
  using Base::m_jacobian;

  // MIRK4 internal points
  Array<Scalar, IntervalsAtCompileTime, 1> m_x_internal;
  Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> m_y_internal;
  Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> m_f_internal;

public:
  template <typename Derived1, typename Derived2, typename Derived3>
  MIRK4(F& fun, BC& bc, FJ& fun_jac, BCJ& bc_jac, const ArrayBase<Derived1>& x, const ArrayBase<Derived2>&y, const ArrayBase<Derived3>& p) : Base{fun, bc, fun_jac, bc_jac, x, y, p},
  m_x_internal{m_cols - 1},
  m_y_internal{m_rows, m_cols - 1},
  m_f_internal{m_rows, m_cols - 1} {
    static_assert(ParamsAtCompileTime != 0);
    if constexpr (IntervalsAtCompileTime == Dynamic) {
      m_x_internal = m_x.head(m_cols - 1) + half<Scalar>() * m_h;
    }
    else {
      m_x_internal = m_x.template head<IntervalsAtCompileTime>() + half<Scalar>() * m_h;
    }
    calculate();
  }

  template <typename Derived1, typename Derived2>
  MIRK4(F& fun, BC& bc, FJ& fun_jac, BCJ& bc_jac, const ArrayBase<Derived1>& x, const ArrayBase<Derived2>&y) : Base{fun, bc, fun_jac, bc_jac, x, y},
  m_x_internal{m_cols - 1},
  m_y_internal{m_rows, m_cols - 1},
  m_f_internal{m_rows, m_cols - 1} {
    static_assert(ParamsAtCompileTime == 0);
    if constexpr (IntervalsAtCompileTime == Dynamic) {
      m_x_internal = m_x.head(m_cols - 1) + half<Scalar>() * m_h;
    }
    else {
      m_x_internal = m_x.template head<IntervalsAtCompileTime>() + half<Scalar>() * m_h;
    }
    calculate();
  }

  ~MIRK4() = default;

  template <typename Derived>
  Matrix<Scalar, ResiduesAtCompileTime, 1> operator()(const MatrixBase<Derived>& Y) {
    assert(Y.size() == m_rows * m_cols + m_params);
    Map<const Array<Scalar, RowsAtCompileTime, ColsAtCompileTime>> y(Y.derived().data(), m_rows, m_cols);
    Matrix<Scalar, ResiduesAtCompileTime, 1> residues(m_rows * m_cols + m_params);
    if constexpr (ParamsAtCompileTime == 0) {
      const Array<Scalar, RowsAtCompileTime, ColsAtCompileTime> f = m_fun(m_x, y);
      const Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> y_internal = calculate_internal(y, f, m_h);
      const Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> f_internal = m_fun(m_x_internal, y_internal);
      residues = calculate_residues(y, f, y_internal, f_internal, m_h);
    }
    else {
      Map<const Array<Scalar, ParamsAtCompileTime, 1>> p(Y.derived().data() + m_rows * m_cols, m_params, 1);
      const Array<Scalar, RowsAtCompileTime, ColsAtCompileTime> f = m_fun(m_x, y, p);
      const Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> y_internal = calculate_internal(y, f, m_h);
      const Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> f_internal = m_fun(m_x_internal, y_internal, p);
      residues = calculate_residues(y, f, y_internal, f_internal, m_h, p);
    }
    return residues;
  }

  using Base::collocation_residues;
  using Base::bc_residues;

  void calculate() {
    m_y_internal = calculate_internal(m_y, m_f, m_h);
    if constexpr (ParamsAtCompileTime == 0) {
      m_f_internal = m_fun(m_x_internal, m_y_internal);
      m_residues = calculate_residues(m_y, m_f, m_y_internal, m_f_internal, m_h);
    }
    else {
      m_f_internal = m_fun(m_x_internal, m_y_internal, this->m_p);
      m_residues = calculate_residues(m_y, m_f, m_y_internal, m_f_internal, m_h, this->m_p);
    }
    new (&collocation_residues) Map<const Array<Scalar, RowsAtCompileTime, Dynamic>>(m_residues.data(), m_rows, m_cols - 1);
    new (&bc_residues) Map<const Array<Scalar, BCsAtCompileTime, 1>>(m_residues.data() + m_rows * (m_cols - 1), m_rows + m_params);
  }

  // Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> normalized_residues() const {
  //   Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> res(m_rows, m_cols - 1);
  //   for (Index idx = 0; idx < m_cols - 1; ++idx) {
  //     Map<const Array<Scalar, RowsAtCompileTime, 1>> fip05(m_f_internal.data() + m_rows * idx, m_rows, 1);
  //     Map<const Array<Scalar, RowsAtCompileTime, 1>> col_res(m_residues.data() + m_rows * idx, m_rows, 1);
  //     Map<Array<Scalar, RowsAtCompileTime, 1>> rip05(res.data() + m_rows * idx, m_rows, 1);
  //     rip05 = three_half<Scalar>() * col_res * (m_h(idx) * (1 + fip05.abs())).inverse();
  //   }
  //   return res;
  // }

  Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> normalized_residues() const {
    Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> result(m_rows, m_cols - 1);
    for (Index idx = 0; idx < m_cols - 1; ++idx) {
      // Map<Array<Scalar, RowsAtCompileTime, 1>> r(result.data() + m_rows * idx, m_rows, 1);
      if constexpr (RowsAtCompileTime == Dynamic) {
        result.col(idx) = three_half<Scalar>() * m_residues.segment(m_rows * idx, m_rows).array().abs() / (1 + m_f_internal.col(idx).abs()) / m_h(idx);
      }
      else {
        result.col(idx) = three_half<Scalar>() * m_residues.template segment<RowsAtCompileTime>(m_rows * idx).array().abs() / (1 + m_f_internal.col(idx).abs()) / m_h(idx);
      }
    }
    return result;
  }

  auto y_half(Index j) const {
    return m_y_internal.col(j);
  }

  auto y_third(Index j) const {
    return A13(Scalar) * m_y.col(j + 1) + A23(Scalar) * m_y.col(j) + m_h(j) * (B13(Scalar) * m_f.col(j + 1) - B23(Scalar) * m_f.col(j));
  }

  auto y_two_thirds(Index j) const {
    return A23(Scalar) * m_y.col(j + 1) + A13(Scalar) * m_y.col(j) + m_h(j) * (B23(Scalar) * m_f.col(j + 1) - B13(Scalar) * m_f.col(j));
  }

  void calculate_residuals();

  void construct_global_jacobian();

  Array<Scalar, RowsAtCompileTime, 1> S(Index i, Scalar w) const {
    return A(w) * m_y.col(i + 1) + A(1 - w) * m_y.col(i) + m_h(i) * (B(w) * m_f.col(i + 1) - B(1 - w) * m_f.col(i));
  }

  template <typename Derived1, typename Derived2>
  void modify_mesh(const ArrayBase<Derived1>& x_new, const ArrayBase<Derived2>& y_new) {
    Base::modify_mesh(x_new, y_new);
    m_x_internal = m_x.head(m_cols - 1) + half<Scalar>() * m_h;
    m_y_internal.resize(m_rows, m_cols - 1);
    m_f_internal.resize(m_rows, m_cols - 1);
    calculate();
  }

  typedef interpolators::MIRK4_interpolator<Scalar, RowsAtCompileTime, ColsAtCompileTime> InterpolatorType;
  friend InterpolatorType;

  static constexpr Scalar residual_order = 3; // the error term in the rms residual is h^residual_order
  static constexpr int internal_points = 1;

private:
  template <typename Derived1, typename Derived2, typename Derived3>
  Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> calculate_internal(const ArrayBase<Derived1>& y,
                                                                              const ArrayBase<Derived2>& f,
                                                                              const ArrayBase<Derived3>& h) {
    Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> y_internal(m_rows, m_cols - 1);
    for (Index idx = 0; idx < m_cols - 1; ++idx) {
      y_internal.col(idx) = half<Scalar>() * (y.col(idx + 1) + y.col(idx)) - h(idx) * eighth<Scalar>() * (f.col(idx + 1) - f.col(idx));
    }
    return y_internal;
  }

  template <typename Derived1, typename Derived2, typename Derived3, typename Derived4, typename Derived5>
  Matrix<Scalar, ResiduesAtCompileTime, 1> calculate_residues(const ArrayBase<Derived1>& y,
                                                              const ArrayBase<Derived2>& f,
                                                              const ArrayBase<Derived3>& y_internal,
                                                              const ArrayBase<Derived4>& f_internal,
                                                              const ArrayBase<Derived5>& h) {
    static_assert(ParamsAtCompileTime == 0);
    Matrix<Scalar, ResiduesAtCompileTime, 1> residues(m_rows * m_cols + m_params);
    for (Index idx = 0; idx < m_cols - 1; ++idx) {
      if constexpr (RowsAtCompileTime == Dynamic) {
        residues.segment(m_rows * idx, m_rows).array() = y.col(idx + 1) - y.col(idx) - h(idx) * (sixth<Scalar>() * (f.col(idx + 1) + f.col(idx)) + two_thirds<Scalar>() * f_internal.col(idx));
      }
      else {
        residues.template segment<RowsAtCompileTime>(m_rows * idx).array() = y.col(idx + 1) - y.col(idx) - h(idx) * (sixth<Scalar>() * (f.col(idx + 1) + f.col(idx)) + two_thirds<Scalar>() * f_internal.col(idx));
      }
    }
    if constexpr (BCsAtCompileTime == Dynamic) {
      // residues.tail(m_rows).array() = m_bc(y.col(0), y.col(last));
      residues.tail(m_rows).array() = m_bc(y.col(0), y.col(m_cols - 1));
    }
    else {
      // residues.template tail<BCsAtCompileTime>().array() = m_bc(y.col(0), y.col(last));
      residues.template tail<BCsAtCompileTime>().array() = m_bc(y.col(0), y.col(m_cols - 1));
    }
    return residues;
  }

  template <typename Derived1, typename Derived2, typename Derived3, typename Derived4, typename Derived5, typename Derived6>
  Matrix<Scalar, ResiduesAtCompileTime, 1> calculate_residues(const ArrayBase<Derived1>& y,
                                                              const ArrayBase<Derived2>& f,
                                                              const ArrayBase<Derived3>& y_internal,
                                                              const ArrayBase<Derived4>& f_internal,
                                                              const ArrayBase<Derived5>& h,
                                                              const ArrayBase<Derived6>& p) {
    static_assert(ParamsAtCompileTime != 0);
    Matrix<Scalar, ResiduesAtCompileTime, 1> residues(m_rows * m_cols + m_params);
    for (Index idx = 0; idx < m_cols - 1; ++idx) {
      if constexpr (RowsAtCompileTime == Dynamic) {
        residues.segment(m_rows * idx, m_rows).array() = y.col(idx + 1) - y.col(idx) - h(idx) * (sixth<Scalar>() * (f.col(idx + 1) + f.col(idx)) + two_thirds<Scalar>() * f_internal.col(idx));
      }
      else {
        residues.template segment<RowsAtCompileTime>(m_rows * idx).array() = y.col(idx + 1) - y.col(idx) - h(idx) * (sixth<Scalar>() * (f.col(idx + 1) + f.col(idx)) + two_thirds<Scalar>() * f_internal.col(idx));
      }
    }
    if constexpr (BCsAtCompileTime == Dynamic) {
      // residues.tail(m_rows + m_params).array() = m_bc(y.col(0), y.col(last), p);
      residues.tail(m_rows + m_params).array() = m_bc(y.col(0), y.col(m_cols - 1), p);
    }
    else {
      // residues.template tail<BCsAtCompileTime>().array() = m_bc(y.col(0), y.col(last), p);
      residues.template tail<BCsAtCompileTime>().array() = m_bc(y.col(0), y.col(m_cols - 1), p);
    }
    return residues;
  }

  // Interpolation
  Scalar A(Scalar w) const {
    return 3 * w * w - 2 * w * w * w;
  }
  Scalar B(Scalar w) const {
    return - w * w + w * w * w;
  }
};

template <typename F, typename BC, typename FJ, typename BCJ, typename Derived1, typename Derived2, typename Derived3>
MIRK4(F& fun, BC& bc, FJ& fun_jac, BCJ& bc_jac, const ArrayBase<Derived1>& x, const ArrayBase<Derived2>&y, const ArrayBase<Derived3>& p) -> MIRK4<typename Derived2::Scalar, Derived2::RowsAtCompileTime, Derived2::ColsAtCompileTime, Derived3::SizeAtCompileTime, F, BC, FJ, BCJ>;

template <typename F, typename BC, typename FJ, typename BCJ, typename Derived1, typename Derived2>
MIRK4(F& fun, BC& bc, FJ& fun_jac, BCJ& bc_jac, const ArrayBase<Derived1>& x, const ArrayBase<Derived2>&y) -> MIRK4<typename Derived2::Scalar, Derived2::RowsAtCompileTime, Derived2::ColsAtCompileTime, 0, F, BC, FJ, BCJ>;

template <typename Scalar, Index RowsAtCompileTime, Index ColsAtCompileTime, Index ParamsAtCompileTime, typename F, typename BC, typename FJ, typename BCJ>
void MIRK4<Scalar, RowsAtCompileTime, ColsAtCompileTime, ParamsAtCompileTime, F, BC, FJ, BCJ>::calculate_residuals() {
  Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> y1(m_rows, m_cols - 1);
  Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> y3(m_rows, m_cols - 1);
  for (Index idx = 0; idx < m_cols - 1; ++idx) {
    y1.col(idx) = QA1(Scalar) * m_y.col(idx + 1) + QA3(Scalar) * m_y.col(idx) + m_h(idx) * (QB1(Scalar) * m_f.col(idx + 1) - QB3(Scalar) * m_f.col(idx));
    y3.col(idx) = QA3(Scalar) * m_y.col(idx + 1) + QA1(Scalar) * m_y.col(idx) + m_h(idx) * (QB3(Scalar) * m_f.col(idx + 1) - QB1(Scalar) * m_f.col(idx));
  }

  auto x1 = m_x_internal - QALPHA(Scalar) * m_h;
  auto x3 = m_x_internal + QALPHA(Scalar) * m_h;
  Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> f1(m_rows, m_cols - 1);
  Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> f3(m_rows, m_cols - 1);
  if constexpr (ParamsAtCompileTime == 0) {
    f1 = m_fun(x1, y1);
    f3 = m_fun(x3, y3);
  }
  else {
    f1 = m_fun(x1, y1, this->m_p);
    f3 = m_fun(x3, y3, this->m_p);
  }

  if constexpr (RowsAtCompileTime == Dynamic) {
    for (Index idx = 0; idx < m_cols - 1; ++idx) {
      Scalar h_inv = 1/m_h(idx);
      auto y1_prime = QAP1(Scalar) * (m_y.col(idx + 1) - m_y.col(idx)) * h_inv + QBP1(Scalar) * m_f.col(idx + 1) + QBP3(Scalar) * m_f.col(idx);
      auto y3_prime = QAP1(Scalar) * (m_y.col(idx + 1) - m_y.col(idx)) * h_inv + QBP3(Scalar) * m_f.col(idx + 1) + QBP1(Scalar) * m_f.col(idx);
      auto r1 = (y1_prime - f1.col(idx)) * (1 + f1.col(idx).abs()).inverse();
      auto r2 = three_half<Scalar>() * m_residues.segment(m_rows * idx, m_rows).array() * (m_h(idx) * (1 + m_f_internal.col(idx).abs())).inverse();
      auto r3 = (y3_prime - f3.col(idx)) * (1 + f3.col(idx).abs()).inverse();
      m_residuals(idx) = sqrt((QW1(Scalar) * (r1.matrix().squaredNorm() + r3.matrix().squaredNorm()) + QW2(Scalar) * r2.matrix().squaredNorm()));
    }
  }
  else {
    for (Index idx = 0; idx < m_cols - 1; ++idx) {
      Scalar h_inv = 1/m_h(idx);
      auto y1_prime = QAP1(Scalar) * (m_y.col(idx + 1) - m_y.col(idx)) * h_inv + QBP1(Scalar) * m_f.col(idx + 1) + QBP3(Scalar) * m_f.col(idx);
      auto y3_prime = QAP1(Scalar) * (m_y.col(idx + 1) - m_y.col(idx)) * h_inv + QBP3(Scalar) * m_f.col(idx + 1) + QBP1(Scalar) * m_f.col(idx);
      auto r1 = (y1_prime - f1.col(idx)) * (1 + f1.col(idx).abs()).inverse();
      auto r2 = three_half<Scalar>() * m_residues.template segment<RowsAtCompileTime>(m_rows * idx).array() * (m_h(idx) * (1 + m_f_internal.col(idx).abs())).inverse();
      auto r3 = (y3_prime - f3.col(idx)) * (1 + f3.col(idx).abs()).inverse();
      m_residuals(idx) = sqrt((QW1(Scalar) * (r1.matrix().squaredNorm() + r3.matrix().squaredNorm()) + QW2(Scalar) * r2.matrix().squaredNorm()));
    }
  }
}

template <typename Scalar, Index RowsAtCompileTime, Index ColsAtCompileTime, Index ParamsAtCompileTime, typename F, typename BC, typename FJ, typename BCJ>
void MIRK4<Scalar, RowsAtCompileTime, ColsAtCompileTime, ParamsAtCompileTime, F, BC, FJ, BCJ>::construct_global_jacobian() {
  Tensor<Scalar, 3> df_dy(m_rows, m_rows, m_cols);
  Tensor<Scalar, 3> df_dp(m_rows, m_params, m_cols);
  Tensor<Scalar, 3> df_dy_internal(m_rows, m_rows, m_cols - 1);
  Tensor<Scalar, 3> df_dp_internal(m_rows, m_params, m_cols - 1);
  Array<Scalar, BCsAtCompileTime, RowsAtCompileTime> dbc_dya(m_rows + m_params, m_rows), dbc_dyb(m_rows + m_params, m_rows);
  Array<Scalar, BCsAtCompileTime, ParamsAtCompileTime> dbc_dp(m_rows + m_params, m_params);

  if constexpr (ParamsAtCompileTime == 0) {
    df_dy = m_fun_jac(m_x, m_y);
    df_dy_internal = m_fun_jac(m_x_internal, m_y_internal);
    std::tie(dbc_dya, dbc_dyb) = m_bc_jac(m_y.col(0), m_y.col(m_cols - 1));
  }
  else {
    std::tie(df_dy, df_dp) = m_fun_jac(m_x, m_y, this->m_p);
    std::tie(df_dy_internal, df_dp_internal) = m_fun_jac(m_x_internal, m_y_internal, this->m_p);
    std::tie(dbc_dya, dbc_dyb, dbc_dp) = m_bc_jac(m_y.col(0), m_y.col(m_cols - 1), this->m_p);
  }

  Array<Scalar, Dynamic, 1> values((2 * m_rows + m_params) * (m_cols * m_rows + m_params));
  for (Index idx = 0; idx < m_cols - 1; ++idx) {
    Map<const Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>> Ji(df_dy.data() + idx * m_rows * m_rows, m_rows, m_rows);
    Map<const Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>> Jip1(df_dy.data() + (idx + 1) * m_rows * m_rows, m_rows, m_rows);
    Map<const Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>> Jip05(df_dy_internal.data() + idx * m_rows * m_rows, m_rows, m_rows);
    Map<const Matrix<Scalar, RowsAtCompileTime, ParamsAtCompileTime>> Ki(df_dp.data() + idx * m_rows * m_params, m_rows, m_params);
    Map<const Matrix<Scalar, RowsAtCompileTime, ParamsAtCompileTime>> Kip1(df_dp.data() + (idx + 1) * m_rows * m_params, m_rows, m_params);
    Map<const Matrix<Scalar, RowsAtCompileTime, ParamsAtCompileTime>> Kip05(df_dp_internal.data() + idx * m_rows * m_params, m_rows, m_params);
    Map<Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>> m1(values.data() + idx * m_rows * m_rows, m_rows, m_rows);
    Map<Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>> m2(values.data() + idx * m_rows * m_rows + (m_cols - 1) * m_rows * m_rows, m_rows, m_rows);
    Map<Matrix<Scalar, RowsAtCompileTime, ParamsAtCompileTime>> m3(values.data() + idx * m_rows * m_params + 2 * (m_cols - 1) * m_rows * m_rows, m_rows, m_params);
    m1.noalias() = - Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>::Identity(m_rows, m_rows) - sixth<Scalar>() * m_h(idx) * (Ji + 2 * Jip05) - m_h(idx) * m_h(idx) * twelfth<Scalar>() * Jip05 * Ji;
    m2.noalias() = Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>::Identity(m_rows, m_rows) - sixth<Scalar>() * m_h(idx) * (Jip1 + 2 * Jip05) + twelfth<Scalar>() * m_h(idx) * m_h(idx) * Jip05 * Jip1;
    m3.noalias() = - sixth<Scalar>() * m_h(idx) * (Ki + Kip1 + 4 * (Kip05 + eighth<Scalar>() * m_h(idx) * Jip05 * (Kip1 - Kip05)));
  }
  if constexpr ((RowsAtCompileTime != Dynamic) && (ParamsAtCompileTime != Dynamic)) {
    values.template segment<RowsAtCompileTime * (RowsAtCompileTime + ParamsAtCompileTime)>((2 * m_rows + m_params) * (m_cols - 1) * m_rows) = dbc_dya.reshaped();
    values.template segment<RowsAtCompileTime * (RowsAtCompileTime + ParamsAtCompileTime)>((2 * m_cols -1) * m_rows * m_rows + m_params * m_cols * m_rows) = dbc_dyb.reshaped();
    values.template tail<ParamsAtCompileTime * (RowsAtCompileTime + ParamsAtCompileTime)>() = dbc_dp.reshaped();
  }
  else {
    values.segment((2 * m_rows + m_params) * (m_cols - 1) * m_rows, m_rows * (m_rows + m_params)) = dbc_dya.reshaped();
    values.segment((2 * m_cols -1) * m_rows * m_rows + m_params * m_cols * m_rows, m_rows * (m_rows + m_params)) = dbc_dyb.reshaped();
    values.tail(m_params * (m_rows + m_params)) = dbc_dp.reshaped();
  }

  // Fill a vector of triplets. This will be used to construct a sparse matrix.
  std::vector<Triplet<Scalar>> coefficients;
  coefficients.reserve((2 * m_rows + m_params) * (m_cols * m_rows + m_params));

  // ODE coefficients
  for (Index idx = 0; idx < (2 * m_rows + m_params) * (m_cols * m_rows + m_params); ++idx) {
    coefficients.emplace_back(m_i_jac(idx), m_j_jac(idx), values(idx));
  }

  m_jacobian.setFromTriplets(coefficients.begin(), coefficients.end());
  m_jacobian.makeCompressed();
}

} // namespace methods
} // namespace collocation

#undef QALPHA
#undef QA1
#undef QA3
#undef QB1
#undef QB3
#undef QAP1
#undef QBP1
#undef QBP3
#undef QW1
#undef QW2
#undef A13
#undef A23
#undef B13
#undef B23

#endif // MIRK4_HPP
