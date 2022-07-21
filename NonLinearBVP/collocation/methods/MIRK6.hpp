#ifndef MIRK6_HPP
#define MIRK6_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/CXX11/Tensor>
#include <cmath>
#include <utility>
#include <algorithm>
#include <initializer_list>

#include "NonLinearBVP/collocation/methods/MIRK.hpp"
#include "NonLinearBVP/constants/constants.hpp"
#include "NonLinearBVP/constants/MIRK6_constants.hpp"
#include "NonLinearBVP/interpolators/MIRK6_interpolator.hpp"

namespace nonlinearbvp { namespace collocation { namespace methods {

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

using boost::math::constants::half;
using boost::math::constants::three_half;
using boost::math::constants::third;
using boost::math::constants::two_thirds;
using boost::math::constants::quarter;
using boost::math::constants::three_quarters;
using boost::math::constants::sixth;
using boost::math::constants::eighth;
using boost::math::constants::tenth;
using boost::math::constants::twelfth;
using boost::math::constants::MIRK6_alpha;
using boost::math::constants::MIRK6_beta1;
using boost::math::constants::MIRK6_beta2;
using boost::math::constants::MIRK6_beta3;
using boost::math::constants::MIRK6_A31;
using boost::math::constants::MIRK6_A33;
using boost::math::constants::MIRK6_B31;
using boost::math::constants::MIRK6_B33;
using boost::math::constants::MIRK6_A51;
using boost::math::constants::MIRK6_A53;
using boost::math::constants::MIRK6_B51;
using boost::math::constants::MIRK6_B52;
using boost::math::constants::MIRK6_B53;
using boost::math::constants::MIRK6_C52;
using boost::math::constants::MIRK6_C53;
using boost::math::constants::MIRK6_D51;
using boost::math::constants::MIRK6_A5p1;
using boost::math::constants::MIRK6_A5p2;
using boost::math::constants::MIRK6_B5p1;
using boost::math::constants::MIRK6_B5p2;
using boost::math::constants::MIRK6_C5p2;
using boost::math::constants::MIRK6_C5p3;
using boost::math::constants::MIRK6_D5p1;
using boost::math::constants::MIRK6R_factor1;
using boost::math::constants::MIRK6R_factor2;
using boost::math::constants::MIRK6R_w1;
using boost::math::constants::MIRK6R_w2;
using boost::math::constants::MIRK6R_w3;
using boost::math::constants::MIRK6R_w4;
using boost::math::constants::MIRK6R_A51;
using boost::math::constants::MIRK6R_A52;
using boost::math::constants::MIRK6R_A53;
using boost::math::constants::MIRK6R_A54;
using boost::math::constants::MIRK6R_B51;
using boost::math::constants::MIRK6R_B52;
using boost::math::constants::MIRK6R_B53;
using boost::math::constants::MIRK6R_B54;
using boost::math::constants::MIRK6R_C51;
using boost::math::constants::MIRK6R_C52;
using boost::math::constants::MIRK6R_C53;
using boost::math::constants::MIRK6R_C54;
using boost::math::constants::MIRK6R_D51;
using boost::math::constants::MIRK6R_D52;
using boost::math::constants::MIRK6R_A5p1;
using boost::math::constants::MIRK6R_A5p2;
using boost::math::constants::MIRK6R_B5p1;
using boost::math::constants::MIRK6R_B5p2;
using boost::math::constants::MIRK6R_B5p3;
using boost::math::constants::MIRK6R_B5p4;
using boost::math::constants::MIRK6R_C5p1;
using boost::math::constants::MIRK6R_C5p2;
using boost::math::constants::MIRK6R_C5p3;
using boost::math::constants::MIRK6R_C5p4;
using boost::math::constants::MIRK6R_D5p1;
using boost::math::constants::MIRK6R_D5p2;
using boost::math::constants::MIRK6M_A51;
using boost::math::constants::MIRK6M_A53;
using boost::math::constants::MIRK6M_B51;
using boost::math::constants::MIRK6M_B53;
using boost::math::constants::MIRK6M_C51;
using boost::math::constants::MIRK6M_C53;
using boost::math::constants::MIRK6M_D51;
using boost::math::constants::MIRK6J_a1;
using boost::math::constants::MIRK6J_a2;
using boost::math::constants::MIRK6J_a3;
using boost::math::constants::MIRK6J_a4;
using boost::math::constants::MIRK6J_a5;
using boost::math::constants::MIRK6J_a6;
using boost::math::constants::MIRK6J_a7;
using boost::math::constants::MIRK6J_a8;
using boost::math::constants::MIRK6J_a9;
using boost::math::constants::MIRK6J_a10;
using boost::math::constants::MIRK6J_a11;
using boost::math::constants::MIRK6J_a12;
using boost::math::constants::MIRK6J_a13;
using boost::math::constants::MIRK6J_a14;
using boost::math::constants::MIRK6J_a15;
using boost::math::constants::MIRK6J_a17;
using boost::math::constants::MIRK6J_a18;
using boost::math::constants::MIRK6J_a19;
using boost::math::constants::MIRK6J_a20;

template <typename _Scalar, Index _RowsAtCompileTime, Index _ColsAtCompileTime, Index _ParamsAtCompileTime, typename F, typename BC, typename FJ, typename BCJ>
class MIRK6 : public MIRK<_Scalar, _RowsAtCompileTime, _ColsAtCompileTime, _ParamsAtCompileTime, F, BC, FJ, BCJ> {
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

  // MIRK6 internal points
  Array<Scalar, IntervalsAtCompileTime, 1> m_x_internal1;
  Array<Scalar, IntervalsAtCompileTime, 1> m_x_internal2;
  Array<Scalar, IntervalsAtCompileTime, 1> m_x_internal3;
  Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> m_y_internal1;
  Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> m_y_internal2;
  Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> m_y_internal3;
  Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> m_f_internal1;
  Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> m_f_internal2;
  Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> m_f_internal3;

public:
  template <typename Derived1, typename Derived2, typename Derived3>
  MIRK6(F& fun, BC& bc, FJ& fun_jac, BCJ& bc_jac, const ArrayBase<Derived1>& x, const ArrayBase<Derived2>&y, const ArrayBase<Derived3>& p) : Base{fun, bc, fun_jac, bc_jac, x, y, p},
  m_x_internal1{m_cols - 1},
  m_x_internal2{m_cols - 1},
  m_x_internal3{m_cols - 1},
  m_y_internal1{m_rows, m_cols - 1},
  m_y_internal2{m_rows, m_cols - 1},
  m_y_internal3{m_rows, m_cols - 1},
  m_f_internal1{m_rows, m_cols - 1},
  m_f_internal2{m_rows, m_cols - 1},
  m_f_internal3{m_rows, m_cols - 1} {
    static_assert(ParamsAtCompileTime != 0);
    if constexpr (IntervalsAtCompileTime == Dynamic) {
      m_x_internal2 = m_x.head(m_cols - 1) + half<Scalar>() * m_h;
      m_x_internal1 = m_x_internal2 - MIRK6_alpha<Scalar>() * m_h;
      m_x_internal3 = m_x_internal2 + MIRK6_alpha<Scalar>() * m_h;
    }
    else {
      m_x_internal2 = m_x.template head<IntervalsAtCompileTime>() + half<Scalar>() * m_h;
      m_x_internal1 = m_x_internal2 - MIRK6_alpha<Scalar>() * m_h;
      m_x_internal3 = m_x_internal2 + MIRK6_alpha<Scalar>() * m_h;
    }
    calculate();
  }

  template <typename Derived1, typename Derived2>
  MIRK6(F& fun, BC& bc, FJ& fun_jac, BCJ& bc_jac, const ArrayBase<Derived1>& x, const ArrayBase<Derived2>&y) : Base{fun, bc, fun_jac, bc_jac, x, y},
  m_x_internal1{m_cols - 1},
  m_x_internal2{m_cols - 1},
  m_x_internal3{m_cols - 1},
  m_y_internal1{m_rows, m_cols - 1},
  m_y_internal2{m_rows, m_cols - 1},
  m_y_internal3{m_rows, m_cols - 1},
  m_f_internal1{m_rows, m_cols - 1},
  m_f_internal2{m_rows, m_cols - 1},
  m_f_internal3{m_rows, m_cols - 1} {
    static_assert(ParamsAtCompileTime == 0);
    if constexpr (IntervalsAtCompileTime == Dynamic) {
      m_x_internal2 = m_x.head(m_cols - 1) + half<Scalar>() * m_h;
      m_x_internal1 = m_x_internal2 - MIRK6_alpha<Scalar>() * m_h;
      m_x_internal3 = m_x_internal2 + MIRK6_alpha<Scalar>() * m_h;
    }
    else {
      m_x_internal2 = m_x.template head<IntervalsAtCompileTime>() + half<Scalar>() * m_h;
      m_x_internal1 = m_x_internal2 - MIRK6_alpha<Scalar>() * m_h;
      m_x_internal3 = m_x_internal2 + MIRK6_alpha<Scalar>() * m_h;
    }
    calculate();
  }

  ~MIRK6() = default;

  /* Calculates the residues given a solution Y. Does not modify class members */
  template <typename Derived1, typename Derived2>
  void operator()(const MatrixBase<Derived1>& Y, const MatrixBase<Derived2>& residues) const {
    assert(Y.size() == m_rows * m_cols + m_params);
    Map<const Array<typename Derived1::Scalar, RowsAtCompileTime, ColsAtCompileTime>> y(Y.derived().data(), m_rows, m_cols);
    // Matrix<typename Derived::Scalar, ResiduesAtCompileTime, 1> residues(m_rows * m_cols + m_params);

    Array<typename Derived1::Scalar, RowsAtCompileTime, IntervalsAtCompileTime> y1(m_rows, m_cols - 1);
    Array<typename Derived1::Scalar, RowsAtCompileTime, IntervalsAtCompileTime> y2(m_rows, m_cols - 1);
    Array<typename Derived1::Scalar, RowsAtCompileTime, IntervalsAtCompileTime> y3(m_rows, m_cols - 1);

    if constexpr (ParamsAtCompileTime == 0) {
# ifdef COLLOCATION_FUNCTION_RETURN
      Array<typename Derived1::Scalar, RowsAtCompileTime, ColsAtCompileTime> f = m_fun(m_x, y);
# else
      Array<typename Derived1::Scalar, RowsAtCompileTime, ColsAtCompileTime> f(m_rows, m_cols);
      m_fun(m_x, y, f);
# endif
      /* calculate the internal points */
      for (Index idx = 0; idx < m_cols - 1; ++idx) {
        y1.col(idx) = MIRK6_A31<Scalar>() * y.col(idx + 1) + MIRK6_A33<Scalar>() * y.col(idx) - m_h(idx) * (MIRK6_B31<Scalar>() * f.col(idx + 1) - MIRK6_B33<Scalar>() * f.col(idx));
        y2.col(idx) = half<Scalar>() * (y.col(idx + 1) + y.col(idx)) - m_h(idx) * eighth<Scalar>() * (f.col(idx + 1) - f.col(idx));
        y3.col(idx) = MIRK6_A33<Scalar>() * y.col(idx + 1) + MIRK6_A31<Scalar>() * y.col(idx) - m_h(idx) * (MIRK6_B33<Scalar>() * f.col(idx + 1) - MIRK6_B31<Scalar>() * f.col(idx));
      }
# ifdef COLLOCATION_FUNCTION_RETURN
      Array<typename Derived1::Scalar, RowsAtCompileTime, IntervalsAtCompileTime> f1 = m_fun(m_x_internal1, y1);
      Array<typename Derived1::Scalar, RowsAtCompileTime, IntervalsAtCompileTime> f2 = m_fun(m_x_internal2, y2);
      Array<typename Derived1::Scalar, RowsAtCompileTime, IntervalsAtCompileTime> f3 = m_fun(m_x_internal3, y3);
# else
      Array<typename Derived1::Scalar, RowsAtCompileTime, IntervalsAtCompileTime> f1(m_rows, m_cols - 1);
      Array<typename Derived1::Scalar, RowsAtCompileTime, IntervalsAtCompileTime> f2(m_rows, m_cols - 1);
      Array<typename Derived1::Scalar, RowsAtCompileTime, IntervalsAtCompileTime> f3(m_rows, m_cols - 1);
      m_fun(m_x_internal1, y1, f1);
      m_fun(m_x_internal2, y2, f2);
      m_fun(m_x_internal3, y3, f3);
# endif

      /* bootstrap the internal points */
      for (Index idx = 0; idx < m_cols - 1; ++idx) {
        y1.col(idx) = MIRK6_A51<Scalar>() * y.col(idx + 1) + MIRK6_A53<Scalar>() * y.col(idx) - m_h(idx) * (MIRK6_B51<Scalar>() * f.col(idx + 1) - MIRK6_B53<Scalar>() * f.col(idx) - MIRK6_C53<Scalar>() * f1.col(idx) + MIRK6_D51<Scalar>() * f2.col(idx));
        y2.col(idx) = half<Scalar>() * (y.col(idx + 1) + y.col(idx)) - m_h(idx) * (MIRK6_B52<Scalar>() * (f.col(idx + 1) - f.col(idx)) + MIRK6_C52<Scalar>() * (f3.col(idx) - f1.col(idx)));
        y3.col(idx) = MIRK6_A53<Scalar>() * y.col(idx + 1) + MIRK6_A51<Scalar>() * y.col(idx) - m_h(idx) * (MIRK6_B53<Scalar>() * f.col(idx + 1) - MIRK6_B51<Scalar>() * f.col(idx) + MIRK6_C53<Scalar>() * f3.col(idx) - MIRK6_D51<Scalar>() * f2.col(idx));
      }
# ifdef COLLOCATION_FUNCTION_RETURN
      f1 = m_fun(m_x_internal1, y1);
      f2 = m_fun(m_x_internal2, y2);
      f3 = m_fun(m_x_internal3, y3);
# else
      m_fun(m_x_internal1, y1, f1);
      m_fun(m_x_internal2, y2, f2);
      m_fun(m_x_internal3, y3, f3);
# endif

      /* bootstrap the internal points again */
      for (Index idx = 0; idx < m_cols - 1; ++idx) {
        y1.col(idx) = MIRK6_A51<Scalar>() * y.col(idx + 1) + MIRK6_A53<Scalar>() * y.col(idx) - m_h(idx) * (MIRK6_B51<Scalar>() * f.col(idx + 1) - MIRK6_B53<Scalar>() * f.col(idx) - MIRK6_C53<Scalar>() * f1.col(idx) + MIRK6_D51<Scalar>() * f2.col(idx));
        y2.col(idx) = half<Scalar>() * (y.col(idx + 1) + y.col(idx)) - m_h(idx) * (MIRK6_B52<Scalar>() * (f.col(idx + 1) - f.col(idx)) + MIRK6_C52<Scalar>() * (f3.col(idx) - f1.col(idx)));
        y3.col(idx) = MIRK6_A53<Scalar>() * y.col(idx + 1) + MIRK6_A51<Scalar>() * y.col(idx) - m_h(idx) * (MIRK6_B53<Scalar>() * f.col(idx + 1) - MIRK6_B51<Scalar>() * f.col(idx) + MIRK6_C53<Scalar>() * f3.col(idx) - MIRK6_D51<Scalar>() * f2.col(idx));
      }
# ifdef COLLOCATION_FUNCTION_RETURN
      f1 = m_fun(m_x_internal1, y1);
      f2 = m_fun(m_x_internal2, y2);
      f3 = m_fun(m_x_internal3, y3);
# else
      m_fun(m_x_internal1, y1, f1);
      m_fun(m_x_internal2, y2, f2);
      m_fun(m_x_internal3, y3, f3);
# endif

      /* calculate residues */
      for (Index idx = 0; idx < m_cols - 1; ++idx) {
        if constexpr (RowsAtCompileTime == Dynamic) {
          const_cast<MatrixBase<Derived2>&>(residues).segment(m_rows * idx, m_rows).array() = y.col(idx + 1) - y.col(idx) - m_h(idx) * (MIRK6_beta1<Scalar>() * (f.col(idx + 1) + f.col(idx)) + MIRK6_beta2<Scalar>() * (f1.col(idx) + f3.col(idx)) + MIRK6_beta2<Scalar>() * f2.col(idx));
        }
        else {
          const_cast<MatrixBase<Derived2>&>(residues).template segment<RowsAtCompileTime>(m_rows * idx).array() = y.col(idx + 1) - y.col(idx) - m_h(idx) * (MIRK6_beta1<Scalar>() * (f.col(idx + 1) + f.col(idx)) + MIRK6_beta2<Scalar>() * (f1.col(idx) + f3.col(idx)) + MIRK6_beta3<Scalar>() * f2.col(idx));
        }
      }
      if constexpr (BCsAtCompileTime == Dynamic) {
# ifdef COLLOCATION_BC_RETURN
        const_cast<MatrixBase<Derived2>&>(residues).tail(m_rows).array() = m_bc(y.col(0), y.col(m_cols - 1));
# else
        m_bc(y.col(0), y.col(m_cols - 1), const_cast<MatrixBase<Derived2>&>(residues).tail(m_rows).array());
# endif
      }
      else {
# ifdef COLLOCATION_BC_RETURN
        const_cast<MatrixBase<Derived2>&>(residues).template tail<BCsAtCompileTime>().array() = m_bc(y.col(0), y.col(m_cols - 1));
# else
        m_bc(y.col(0), y.col(m_cols - 1), const_cast<MatrixBase<Derived2>&>(residues).template tail<BCsAtCompileTime>().array());
# endif
      }
    }
    else {
      Map<const Array<typename Derived1::Scalar, ParamsAtCompileTime, 1>> p(Y.derived().data() + m_rows * m_cols, m_params, 1);
# ifdef COLLOCATION_FUNCTION_RETURN
      Array<typename Derived1::Scalar, RowsAtCompileTime, ColsAtCompileTime> f = m_fun(m_x, y, p);
# else
      Array<typename Derived1::Scalar, RowsAtCompileTime, ColsAtCompileTime> f(m_rows, m_cols);
      m_fun(m_x, y, p, f);
# endif
      /* calculate the internal points */
      for (Index idx = 0; idx < m_cols - 1; ++idx) {
        y1.col(idx) = MIRK6_A31<Scalar>() * y.col(idx + 1) + MIRK6_A33<Scalar>() * y.col(idx) - m_h(idx) * (MIRK6_B31<Scalar>() * f.col(idx + 1) - MIRK6_B33<Scalar>() * f.col(idx));
        y2.col(idx) = half<Scalar>() * (y.col(idx + 1) + y.col(idx)) - m_h(idx) * eighth<Scalar>() * (f.col(idx + 1) - f.col(idx));
        y3.col(idx) = MIRK6_A33<Scalar>() * y.col(idx + 1) + MIRK6_A31<Scalar>() * y.col(idx) - m_h(idx) * (MIRK6_B33<Scalar>() * f.col(idx + 1) - MIRK6_B31<Scalar>() * f.col(idx));
      }
# ifdef COLLOCATION_FUNCTION_RETURN
      Array<typename Derived1::Scalar, RowsAtCompileTime, IntervalsAtCompileTime> f1 = m_fun(m_x_internal1, y1, p);
      Array<typename Derived1::Scalar, RowsAtCompileTime, IntervalsAtCompileTime> f2 = m_fun(m_x_internal2, y2, p);
      Array<typename Derived1::Scalar, RowsAtCompileTime, IntervalsAtCompileTime> f3 = m_fun(m_x_internal3, y3, p);
# else
      Array<typename Derived1::Scalar, RowsAtCompileTime, IntervalsAtCompileTime> f1(m_rows, m_cols - 1);
      Array<typename Derived1::Scalar, RowsAtCompileTime, IntervalsAtCompileTime> f2(m_rows, m_cols - 1);
      Array<typename Derived1::Scalar, RowsAtCompileTime, IntervalsAtCompileTime> f3(m_rows, m_cols - 1);
      m_fun(m_x_internal1, y1, p, f1);
      m_fun(m_x_internal2, y2, p, f2);
      m_fun(m_x_internal3, y3, p, f3);
# endif

      /* bootstrap the internal points */
      for (Index idx = 0; idx < m_cols - 1; ++idx) {
        y1.col(idx) = MIRK6_A51<Scalar>() * y.col(idx + 1) + MIRK6_A53<Scalar>() * y.col(idx) - m_h(idx) * (MIRK6_B51<Scalar>() * f.col(idx + 1) - MIRK6_B53<Scalar>() * f.col(idx) - MIRK6_C53<Scalar>() * f1.col(idx) + MIRK6_D51<Scalar>() * f2.col(idx));
        y2.col(idx) = half<Scalar>() * (y.col(idx + 1) + y.col(idx)) - m_h(idx) * (MIRK6_B52<Scalar>() * (f.col(idx + 1) - f.col(idx)) + MIRK6_C52<Scalar>() * (f3.col(idx) - f1.col(idx)));
        y3.col(idx) = MIRK6_A53<Scalar>() * y.col(idx + 1) + MIRK6_A51<Scalar>() * y.col(idx) - m_h(idx) * (MIRK6_B53<Scalar>() * f.col(idx + 1) - MIRK6_B51<Scalar>() * f.col(idx) + MIRK6_C53<Scalar>() * f3.col(idx) - MIRK6_D51<Scalar>() * f2.col(idx));
      }
# ifdef COLLOCATION_FUNCTION_RETURN
      f1 = m_fun(m_x_internal1, y1, p);
      f2 = m_fun(m_x_internal2, y2, p);
      f3 = m_fun(m_x_internal3, y3, p);
# else
      m_fun(m_x_internal1, y1, p, f1);
      m_fun(m_x_internal2, y2, p, f2);
      m_fun(m_x_internal3, y3, p, f3);
# endif

      /* bootstrap the internal points again */
      for (Index idx = 0; idx < m_cols - 1; ++idx) {
        y1.col(idx) = MIRK6_A51<Scalar>() * y.col(idx + 1) + MIRK6_A53<Scalar>() * y.col(idx) - m_h(idx) * (MIRK6_B51<Scalar>() * f.col(idx + 1) - MIRK6_B53<Scalar>() * f.col(idx) - MIRK6_C53<Scalar>() * f1.col(idx) + MIRK6_D51<Scalar>() * f2.col(idx));
        y2.col(idx) = half<Scalar>() * (y.col(idx + 1) + y.col(idx)) - m_h(idx) * (MIRK6_B52<Scalar>() * (f.col(idx + 1) - f.col(idx)) + MIRK6_C52<Scalar>() * (f3.col(idx) - f1.col(idx)));
        y3.col(idx) = MIRK6_A53<Scalar>() * y.col(idx + 1) + MIRK6_A51<Scalar>() * y.col(idx) - m_h(idx) * (MIRK6_B53<Scalar>() * f.col(idx + 1) - MIRK6_B51<Scalar>() * f.col(idx) + MIRK6_C53<Scalar>() * f3.col(idx) - MIRK6_D51<Scalar>() * f2.col(idx));
      }
# ifdef COLLOCATION_FUNCTION_RETURN
      f1 = m_fun(m_x_internal1, y1, p);
      f2 = m_fun(m_x_internal2, y2, p);
      f3 = m_fun(m_x_internal3, y3, p);
# else
      m_fun(m_x_internal1, y1, p, f1);
      m_fun(m_x_internal2, y2, p, f2);
      m_fun(m_x_internal3, y3, p, f3);
# endif

      /* calculate residues */
      for (Index idx = 0; idx < m_cols - 1; ++idx) {
        if constexpr (RowsAtCompileTime == Dynamic) {
          const_cast<MatrixBase<Derived2>&>(residues).segment(m_rows * idx, m_rows).array() = y.col(idx + 1) - y.col(idx) - m_h(idx) * (MIRK6_beta1<Scalar>() * (f.col(idx + 1) + f.col(idx)) + MIRK6_beta2<Scalar>() * (f1.col(idx) + f3.col(idx)) + MIRK6_beta3<Scalar>() * f2.col(idx));
        }
        else {
          const_cast<MatrixBase<Derived2>&>(residues).template segment<RowsAtCompileTime>(m_rows * idx).array() = y.col(idx + 1) - y.col(idx) - m_h(idx) * (MIRK6_beta1<Scalar>() * (f.col(idx + 1) + f.col(idx)) + MIRK6_beta2<Scalar>() * (f1.col(idx) + f3.col(idx)) + MIRK6_beta3<Scalar>() * f2.col(idx));
        }
      }
      if constexpr (BCsAtCompileTime == Dynamic) {
# ifdef COLLOCATION_BC_RETURN
        const_cast<MatrixBase<Derived2>&>(residues).tail(m_rows + m_params).array() = m_bc(y.col(0), y.col(m_cols - 1), p);
# else
        m_bc(y.col(0), y.col(m_cols - 1), p, const_cast<MatrixBase<Derived2>&>(residues).tail(m_rows + m_params).array());
# endif
      }
      else {
# ifdef COLLOCATION_BC_RETURN
        const_cast<MatrixBase<Derived2>&>(residues).template tail<BCsAtCompileTime>().array() = m_bc(y.col(0), y.col(m_cols - 1), p);
# else
        m_bc(y.col(0), y.col(m_cols - 1), p, const_cast<MatrixBase<Derived2>&>(residues).template tail<BCsAtCompileTime>().array());
# endif
      }
    }
  }

  template <typename Derived>
  Matrix<typename Derived::Scalar, ResiduesAtCompileTime, 1> operator()(const MatrixBase<Derived>& Y) const {
    Matrix<typename Derived::Scalar, ResiduesAtCompileTime, 1> residues(m_rows * m_cols + m_params);
    this->operator()(Y, residues);
    return residues;
  }

  using Base::residual_norms;
  using Base::collocation_residues;
  using Base::bc_residues;

  void calculate() {
    calculate_internal();
    calculate_residues();
    new (&collocation_residues) Map<const Array<Scalar, RowsAtCompileTime, Dynamic>>(m_residues.data(), m_rows, m_cols - 1);
    new (&bc_residues) Map<const Array<Scalar, BCsAtCompileTime, 1>>(m_residues.data() + m_rows * (m_cols - 1), m_rows + m_params);
  }

  // Array<Scalar, IntervalsAtCompileTime, 1> normalized_residues() const {
  //   Array<Scalar, IntervalsAtCompileTime, 1> result(m_cols - 1);
  //   for (Index idx = 0; idx < m_cols - 1; ++idx) {
  //     Scalar hinv = 1 / m_h(idx);
  //     auto y1prime = MIRK6_A5p1<Scalar>() * (m_y.col(idx + 1) - m_y.col(idx)) * hinv - MIRK6_B5p1<Scalar>() * (m_f.col(idx + 1) + m_f.col(idx)) - sixth<Scalar>() * m_f_internal3.col(idx) - MIRK6_C5p3<Scalar>() * m_f_internal1.col(idx) - MIRK6_D5p1<Scalar>() * m_f_internal2.col(idx);
  //     auto y2prime = MIRK6_A5p2<Scalar>() * (m_y.col(idx + 1) - m_y.col(idx)) * hinv - MIRK6_B5p2<Scalar>() * (m_f.col(idx + 1) + m_f.col(idx)) - MIRK6_C5p2<Scalar>() * (m_f_internal3.col(idx) + m_f_internal1.col(idx)) - third<Scalar>() * m_f_internal2.col(idx);
  //     auto y3prime = MIRK6_A5p1<Scalar>() * (m_y.col(idx + 1) - m_y.col(idx)) * hinv - MIRK6_B5p1<Scalar>() * (m_f.col(idx + 1) + m_f.col(idx)) - MIRK6_C5p3<Scalar>() * m_f_internal3.col(idx) - sixth<Scalar>() * m_f_internal1.col(idx) - MIRK6_D5p1<Scalar>() * m_f_internal2.col(idx);
  //     Scalar r1 = m_h(idx) * ((y1prime - m_f_internal1.col(idx)) / (1 + m_y_internal1.col(idx).abs())).matrix().stableNorm();
  //     Scalar r2 = m_h(idx) * ((y2prime - m_f_internal2.col(idx)) / (1 + m_y_internal2.col(idx).abs())).matrix().stableNorm();
  //     Scalar r3 = m_h(idx) * ((y3prime - m_f_internal3.col(idx)) / (1 + m_y_internal3.col(idx).abs())).matrix().stableNorm();
  //     result(idx) = std::max({r1, r2, r3});
  //   }
  //   return result;
  // }

  bool residues_converged(Scalar a_tol, Scalar r_tol) const {
    bool success = true;
    for (Index idx = 0; idx < m_cols - 1; ++idx) {
      Scalar hinv = 1 / m_h(idx);
      auto y1prime = MIRK6_A5p1<Scalar>() * (m_y.col(idx + 1) - m_y.col(idx)) * hinv - MIRK6_B5p1<Scalar>() * (m_f.col(idx + 1) + m_f.col(idx)) - sixth<Scalar>() * m_f_internal3.col(idx) - MIRK6_C5p3<Scalar>() * m_f_internal1.col(idx) - MIRK6_D5p1<Scalar>() * m_f_internal2.col(idx);
      auto y2prime = MIRK6_A5p2<Scalar>() * (m_y.col(idx + 1) - m_y.col(idx)) * hinv - MIRK6_B5p2<Scalar>() * (m_f.col(idx + 1) + m_f.col(idx)) - MIRK6_C5p2<Scalar>() * (m_f_internal3.col(idx) + m_f_internal1.col(idx)) - third<Scalar>() * m_f_internal2.col(idx);
      auto y3prime = MIRK6_A5p1<Scalar>() * (m_y.col(idx + 1) - m_y.col(idx)) * hinv - MIRK6_B5p1<Scalar>() * (m_f.col(idx + 1) + m_f.col(idx)) - MIRK6_C5p3<Scalar>() * m_f_internal3.col(idx) - sixth<Scalar>() * m_f_internal1.col(idx) - MIRK6_D5p1<Scalar>() * m_f_internal2.col(idx);
      auto r1 = MIRK6R_factor1<Scalar>() * m_h(idx) * (y1prime - m_f_internal1.col(idx)).abs();
      auto r2 = MIRK6R_factor2<Scalar>() * m_h(idx) * (y2prime - m_f_internal2.col(idx)).abs();
      auto r3 = MIRK6R_factor1<Scalar>() * m_h(idx) * (y3prime - m_f_internal3.col(idx)).abs();
      if ((r1 > a_tol + r_tol * m_y_internal1.col(idx).abs()).any() || (r2 > a_tol + r_tol * m_y_internal2.col(idx).abs()).any() || (r3 > a_tol + r_tol * m_y_internal3.col(idx).abs()).any()) {
        success = false;
        break;
      }
    }
    return success;
  }

  auto y_half(Index j) const {
    return m_y_internal2.col(j);
  }

  auto y_third(Index j) const {
    return MIRK6M_A51<Scalar>() * m_y.col(j + 1) + MIRK6M_A53<Scalar>() * m_y.col(j) - m_h(j) * (MIRK6M_B51<Scalar>() * m_f.col(j + 1) - MIRK6M_B53<Scalar>() * m_f.col(j) + MIRK6M_C51<Scalar>() * m_f_internal3.col(j) - MIRK6M_C53<Scalar>() * m_f_internal1.col(j) + MIRK6M_D51<Scalar>() * m_f_internal2.col(j));
  }

  auto y_two_thirds(Index j) const {
    return MIRK6M_A53<Scalar>() * m_y.col(j + 1) + MIRK6M_A51<Scalar>() * m_y.col(j) - m_h(j) * (MIRK6M_B53<Scalar>() * m_f.col(j + 1) - MIRK6M_B51<Scalar>() * m_f.col(j) + MIRK6M_C53<Scalar>() * m_f_internal3.col(j) - MIRK6M_C51<Scalar>() * m_f_internal1.col(j) - MIRK6M_D51<Scalar>() * m_f_internal2.col(j));
  }

  void calculate_residuals();

  void construct_global_jacobian();

  Array<Scalar, RowsAtCompileTime, 1> S(Index i, Scalar w) const {
    return A(w) * m_y.col(i + 1) + A(1 - w) * m_y.col(i) - m_h(i) * (B(w) * m_f.col(i + 1) - B(1 - w) * m_f.col(i) + C(w) * m_f_internal3.col(i) - C(1 - w) * m_f_internal1.col(i) + D(w) * m_f_internal2.col(i));
  }

  template <typename Derived1, typename Derived2>
  void modify_mesh(const ArrayBase<Derived1>& x_new, const ArrayBase<Derived2>& y_new) {
    Base::modify_mesh(x_new, y_new);
    m_x_internal2 = m_x.head(m_cols - 1) + half<Scalar>() * m_h;
    m_x_internal1 = m_x_internal2 - MIRK6_alpha<Scalar>() * m_h;
    m_x_internal3 = m_x_internal2 + MIRK6_alpha<Scalar>() * m_h;
    m_y_internal1.resize(m_rows, m_cols - 1);
    m_y_internal2.resize(m_rows, m_cols - 1);
    m_y_internal3.resize(m_rows, m_cols - 1);
    m_f_internal1.resize(m_rows, m_cols - 1);
    m_f_internal2.resize(m_rows, m_cols - 1);
    m_f_internal3.resize(m_rows, m_cols - 1);
    calculate();
  }

  typedef interpolators::MIRK6_interpolator<Scalar, RowsAtCompileTime, ColsAtCompileTime> InterpolatorType;
  friend InterpolatorType;

  static constexpr Scalar residual_order = 6;
  static constexpr int internal_points = 3;

private:
  void calculate_internal() {
    for (Index idx = 0; idx < m_cols - 1; ++idx) {
      m_y_internal1.col(idx) = MIRK6_A31<Scalar>() * m_y.col(idx + 1) + MIRK6_A33<Scalar>() * m_y.col(idx) - m_h(idx) * (MIRK6_B31<Scalar>() * m_f.col(idx + 1) - MIRK6_B33<Scalar>() * m_f.col(idx));
      m_y_internal2.col(idx) = half<Scalar>() * (m_y.col(idx + 1) + m_y.col(idx)) - m_h(idx) * eighth<Scalar>() * (m_f.col(idx + 1) - m_f.col(idx));
      m_y_internal3.col(idx) = MIRK6_A33<Scalar>() * m_y.col(idx + 1) + MIRK6_A31<Scalar>() * m_y.col(idx) - m_h(idx) * (MIRK6_B33<Scalar>() * m_f.col(idx + 1) - MIRK6_B31<Scalar>() * m_f.col(idx));
    }
    if constexpr (ParamsAtCompileTime == 0) {
# ifdef COLLOCATION_FUNCTION_RETURN
      m_f_internal1 = m_fun(m_x_internal1, m_y_internal1);
      m_f_internal2 = m_fun(m_x_internal2, m_y_internal2);
      m_f_internal3 = m_fun(m_x_internal3, m_y_internal3);
# else
      m_fun(m_x_internal1, m_y_internal1, m_f_internal1);
      m_fun(m_x_internal2, m_y_internal2, m_f_internal2);
      m_fun(m_x_internal3, m_y_internal3, m_f_internal3);
# endif
    }
    else {
# ifdef COLLOCATION_FUNCTION_RETURN
      m_f_internal1 = m_fun(m_x_internal1, m_y_internal1, this->m_p);
      m_f_internal2 = m_fun(m_x_internal2, m_y_internal2, this->m_p);
      m_f_internal3 = m_fun(m_x_internal3, m_y_internal3, this->m_p);
# else
      m_fun(m_x_internal1, m_y_internal1, this->m_p, m_f_internal1);
      m_fun(m_x_internal2, m_y_internal2, this->m_p, m_f_internal2);
      m_fun(m_x_internal3, m_y_internal3, this->m_p, m_f_internal3);
# endif
    }

    /* boostrap the internal points to higher order */
    for (Index idx = 0; idx < m_cols - 1; ++idx) {
      m_y_internal1.col(idx) = MIRK6_A51<Scalar>() * m_y.col(idx + 1) + MIRK6_A53<Scalar>() * m_y.col(idx) - m_h(idx) * (MIRK6_B51<Scalar>() * m_f.col(idx + 1) - MIRK6_B53<Scalar>() * m_f.col(idx) - MIRK6_C53<Scalar>() * m_f_internal1.col(idx) + MIRK6_D51<Scalar>() * m_f_internal2.col(idx));
      m_y_internal2.col(idx) = half<Scalar>() * (m_y.col(idx + 1) + m_y.col(idx)) - m_h(idx) * (MIRK6_B52<Scalar>() * (m_f.col(idx + 1) - m_f.col(idx)) + MIRK6_C52<Scalar>() * (m_f_internal3.col(idx) - m_f_internal1.col(idx)));
      m_y_internal3.col(idx) = MIRK6_A53<Scalar>() * m_y.col(idx + 1) + MIRK6_A51<Scalar>() * m_y.col(idx) - m_h(idx) * (MIRK6_B53<Scalar>() * m_f.col(idx + 1) - MIRK6_B51<Scalar>() * m_f.col(idx) + MIRK6_C53<Scalar>() * m_f_internal3.col(idx) - MIRK6_D51<Scalar>() * m_f_internal2.col(idx));
    }
    if constexpr (ParamsAtCompileTime == 0) {
# ifdef COLLOCATION_FUNCTION_RETURN
      m_f_internal1 = m_fun(m_x_internal1, m_y_internal1);
      m_f_internal2 = m_fun(m_x_internal2, m_y_internal2);
      m_f_internal3 = m_fun(m_x_internal3, m_y_internal3);
# else
      m_fun(m_x_internal1, m_y_internal1, m_f_internal1);
      m_fun(m_x_internal2, m_y_internal2, m_f_internal2);
      m_fun(m_x_internal3, m_y_internal3, m_f_internal3);
# endif
    }
    else {
# ifdef COLLOCATION_FUNCTION_RETURN
      m_f_internal1 = m_fun(m_x_internal1, m_y_internal1, this->m_p);
      m_f_internal2 = m_fun(m_x_internal2, m_y_internal2, this->m_p);
      m_f_internal3 = m_fun(m_x_internal3, m_y_internal3, this->m_p);
# else
      m_fun(m_x_internal1, m_y_internal1, this->m_p, m_f_internal1);
      m_fun(m_x_internal2, m_y_internal2, this->m_p, m_f_internal2);
      m_fun(m_x_internal3, m_y_internal3, this->m_p, m_f_internal3);
# endif
    }

    /* boostrap the internal points again for superconvergence */
    for (Index idx = 0; idx < m_cols - 1; ++idx) {
      m_y_internal1.col(idx) = MIRK6_A51<Scalar>() * m_y.col(idx + 1) + MIRK6_A53<Scalar>() * m_y.col(idx) - m_h(idx) * (MIRK6_B51<Scalar>() * m_f.col(idx + 1) - MIRK6_B53<Scalar>() * m_f.col(idx) - MIRK6_C53<Scalar>() * m_f_internal1.col(idx) + MIRK6_D51<Scalar>() * m_f_internal2.col(idx));
      m_y_internal2.col(idx) = half<Scalar>() * (m_y.col(idx + 1) + m_y.col(idx)) - m_h(idx) * (MIRK6_B52<Scalar>() * (m_f.col(idx + 1) - m_f.col(idx)) + MIRK6_C52<Scalar>() * (m_f_internal3.col(idx) - m_f_internal1.col(idx)));
      m_y_internal3.col(idx) = MIRK6_A53<Scalar>() * m_y.col(idx + 1) + MIRK6_A51<Scalar>() * m_y.col(idx) - m_h(idx) * (MIRK6_B53<Scalar>() * m_f.col(idx + 1) - MIRK6_B51<Scalar>() * m_f.col(idx) + MIRK6_C53<Scalar>() * m_f_internal3.col(idx) - MIRK6_D51<Scalar>() * m_f_internal2.col(idx));
    }
    if constexpr (ParamsAtCompileTime == 0) {
# ifdef COLLOCATION_FUNCTION_RETURN
      m_f_internal1 = m_fun(m_x_internal1, m_y_internal1);
      m_f_internal2 = m_fun(m_x_internal2, m_y_internal2);
      m_f_internal3 = m_fun(m_x_internal3, m_y_internal3);
# else
      m_fun(m_x_internal1, m_y_internal1, m_f_internal1);
      m_fun(m_x_internal2, m_y_internal2, m_f_internal2);
      m_fun(m_x_internal3, m_y_internal3, m_f_internal3);
# endif
    }
    else {
# ifdef COLLOCATION_FUNCTION_RETURN
      m_f_internal1 = m_fun(m_x_internal1, m_y_internal1, this->m_p);
      m_f_internal2 = m_fun(m_x_internal2, m_y_internal2, this->m_p);
      m_f_internal3 = m_fun(m_x_internal3, m_y_internal3, this->m_p);
# else
      m_fun(m_x_internal1, m_y_internal1, this->m_p, m_f_internal1);
      m_fun(m_x_internal2, m_y_internal2, this->m_p, m_f_internal2);
      m_fun(m_x_internal3, m_y_internal3, this->m_p, m_f_internal3);
# endif
    }
  }

  void calculate_residues() {
    for (Index idx = 0; idx < m_cols - 1; ++idx) {
      if constexpr (RowsAtCompileTime == Dynamic) {
        m_residues.segment(m_rows * idx, m_rows).array() = m_y.col(idx + 1) - m_y.col(idx) - m_h(idx) * (MIRK6_beta1<Scalar>() * (m_f.col(idx + 1) + m_f.col(idx)) + MIRK6_beta2<Scalar>() * (m_f_internal1.col(idx) + m_f_internal3.col(idx)) + MIRK6_beta3<Scalar>() * m_f_internal2.col(idx));
      }
      else {
        m_residues.template segment<RowsAtCompileTime>(m_rows * idx).array() = m_y.col(idx + 1) - m_y.col(idx) - m_h(idx) * (MIRK6_beta1<Scalar>() * (m_f.col(idx + 1) + m_f.col(idx)) + MIRK6_beta2<Scalar>() * (m_f_internal1.col(idx) + m_f_internal3.col(idx)) + MIRK6_beta3<Scalar>() * m_f_internal2.col(idx));
      }
    }
    if constexpr (BCsAtCompileTime == Dynamic) {
      if constexpr (ParamsAtCompileTime == 0) {
# ifdef COLLOCATION_BC_RETURN
        m_residues.tail(m_rows).array() = m_bc(m_y.col(0), m_y.col(m_cols - 1));
# else
        m_bc(m_y.col(0), m_y.col(m_cols - 1), m_residues.tail(m_rows).array());
# endif
      }
      else {
# ifdef COLLOCATION_BC_RETURN
        m_residues.tail(m_rows + m_params).array() = m_bc(m_y.col(0), m_y.col(m_cols - 1), this->m_p);
# else
        m_bc(m_y.col(0), m_y.col(m_cols - 1), this->m_p, m_residues.tail(m_rows + m_params).array());
# endif
      }
    }
    else {
      if constexpr (ParamsAtCompileTime == 0) {
# ifdef COLLOCATION_BC_RETURN
        m_residues.template tail<BCsAtCompileTime>().array() = m_bc(m_y.col(0), m_y.col(m_cols - 1));
# else
        m_bc(m_y.col(0), m_y.col(m_cols - 1), m_residues.template tail<BCsAtCompileTime>().array());
# endif
      }
      else {
# ifdef COLLOCATION_BC_RETURN
        m_residues.template tail<BCsAtCompileTime>().array() = m_bc(m_y.col(0), m_y.col(m_cols - 1), this->m_p);
# else
        m_bc(m_y.col(0), m_y.col(m_cols - 1), this->m_p, m_residues.template tail<BCsAtCompileTime>().array());
# endif
      }
    }
  }

  // Interpolation
  Scalar A(Scalar w) const {
    Scalar temp = w * w;
    return w * temp * (10 - 15 * w + 6 * temp);
  }
  Scalar B(Scalar w) const {
    Scalar temp = w * w;
    return half<Scalar>() * temp * (1 - w) * (1 - 4 * w + 5 * temp);
  }
  Scalar C(Scalar w) const {
    Scalar temp = 1 - w;
    return sixth<Scalar>() * 49 * w * w * temp * temp * (w - half<Scalar>() + MIRK6_alpha<Scalar>());
  }
  Scalar D(Scalar w) const {
    Scalar temp = 1 - w;
    return third<Scalar>() * 8 * w * w * temp * temp * (1 - 2 * w);
  }
};

template <typename F, typename BC, typename FJ, typename BCJ, typename Derived1, typename Derived2, typename Derived3>
MIRK6(F& fun, BC& bc, FJ& fun_jac, BCJ& bc_jac, const ArrayBase<Derived1>& x, const ArrayBase<Derived2>&y, const ArrayBase<Derived3>& p) -> MIRK6<typename Derived2::Scalar, Derived2::RowsAtCompileTime, Derived2::ColsAtCompileTime, Derived3::SizeAtCompileTime, F, BC, FJ, BCJ>;

template <typename F, typename BC, typename FJ, typename BCJ, typename Derived1, typename Derived2>
MIRK6(F& fun, BC& bc, FJ& fun_jac, BCJ& bc_jac, const ArrayBase<Derived1>& x, const ArrayBase<Derived2>&y) -> MIRK6<typename Derived2::Scalar, Derived2::RowsAtCompileTime, Derived2::ColsAtCompileTime, 0, F, BC, FJ, BCJ>;

template <typename Scalar, Index RowsAtCompileTime, Index ColsAtCompileTime, Index ParamsAtCompileTime, typename F, typename BC, typename FJ, typename BCJ>
void MIRK6<Scalar, RowsAtCompileTime, ColsAtCompileTime, ParamsAtCompileTime, F, BC, FJ, BCJ>::calculate_residuals() {
  Array<Scalar, IntervalsAtCompileTime, 1> x1(m_cols - 1);
  Array<Scalar, IntervalsAtCompileTime, 1> x2(m_cols - 1);
  Array<Scalar, IntervalsAtCompileTime, 1> x3(m_cols - 1);
  Array<Scalar, IntervalsAtCompileTime, 1> x4(m_cols - 1);
  if constexpr (IntervalsAtCompileTime == Dynamic) {
    x1 = m_x.head(m_cols - 1) + MIRK6R_w1<Scalar>() * m_h;
    x2 = m_x.head(m_cols - 1) + MIRK6R_w2<Scalar>() * m_h;
    x3 = m_x.head(m_cols - 1) + MIRK6R_w3<Scalar>() * m_h;
    x4 = m_x.head(m_cols - 1) + MIRK6R_w4<Scalar>() * m_h;
  }
  else {
    x1 = m_x.template head<IntervalsAtCompileTime>() + MIRK6R_w1<Scalar>() * m_h;
    x2 = m_x.template head<IntervalsAtCompileTime>() + MIRK6R_w2<Scalar>() * m_h;
    x3 = m_x.template head<IntervalsAtCompileTime>() + MIRK6R_w3<Scalar>() * m_h;
    x4 = m_x.template head<IntervalsAtCompileTime>() + MIRK6R_w4<Scalar>() * m_h;
  }
  Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> y1(m_rows, m_cols - 1);
  Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> y2(m_rows, m_cols - 1);
  Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> y3(m_rows, m_cols - 1);
  Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> y4(m_rows, m_cols - 1);
  Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> y1prime(m_rows, m_cols - 1);
  Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> y2prime(m_rows, m_cols - 1);
  Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> y3prime(m_rows, m_cols - 1);
  Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> y4prime(m_rows, m_cols - 1);
  Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> f1(m_rows, m_cols - 1);
  Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> f2(m_rows, m_cols - 1);
  Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> f3(m_rows, m_cols - 1);
  Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> f4(m_rows, m_cols - 1);

  for (Index idx = 0; idx < m_cols - 1; ++idx) {
    Scalar hinv = 1 / m_h(idx);

    y1.col(idx) = MIRK6R_A51<Scalar>() * m_y.col(idx + 1) + MIRK6R_A54<Scalar>() * m_y.col(idx) - m_h(idx) * (MIRK6R_B51<Scalar>() * m_f.col(idx + 1) - MIRK6R_B54<Scalar>() * m_f.col(idx) + MIRK6R_C51<Scalar>() * m_f_internal3.col(idx) - MIRK6R_C54<Scalar>() * m_f_internal1.col(idx) + MIRK6R_D51<Scalar>() * m_f_internal2.col(idx));
    y2.col(idx) = MIRK6R_A52<Scalar>() * m_y.col(idx + 1) + MIRK6R_A53<Scalar>() * m_y.col(idx) - m_h(idx) * (MIRK6R_B52<Scalar>() * m_f.col(idx + 1) - MIRK6R_B53<Scalar>() * m_f.col(idx) + MIRK6R_C52<Scalar>() * m_f_internal3.col(idx) - MIRK6R_C53<Scalar>() * m_f_internal1.col(idx) + MIRK6R_D52<Scalar>() * m_f_internal2.col(idx));
    y3.col(idx) = MIRK6R_A53<Scalar>() * m_y.col(idx + 1) + MIRK6R_A52<Scalar>() * m_y.col(idx) - m_h(idx) * (MIRK6R_B53<Scalar>() * m_f.col(idx + 1) - MIRK6R_B52<Scalar>() * m_f.col(idx) + MIRK6R_C53<Scalar>() * m_f_internal3.col(idx) - MIRK6R_C52<Scalar>() * m_f_internal1.col(idx) - MIRK6R_D52<Scalar>() * m_f_internal2.col(idx));
    y4.col(idx) = MIRK6R_A54<Scalar>() * m_y.col(idx + 1) + MIRK6R_A51<Scalar>() * m_y.col(idx) - m_h(idx) * (MIRK6R_B54<Scalar>() * m_f.col(idx + 1) - MIRK6R_B51<Scalar>() * m_f.col(idx) + MIRK6R_C54<Scalar>() * m_f_internal3.col(idx) - MIRK6R_C51<Scalar>() * m_f_internal1.col(idx) - MIRK6R_D51<Scalar>() * m_f_internal2.col(idx));

    y1prime.col(idx) = MIRK6R_A5p1<Scalar>() * (m_y.col(idx + 1) - m_y.col(idx)) * hinv - MIRK6R_B5p1<Scalar>() * m_f.col(idx + 1) - MIRK6R_B5p4<Scalar>() * m_f.col(idx) - MIRK6R_C5p1<Scalar>() * m_f_internal3.col(idx) - MIRK6R_C5p4<Scalar>() * m_f_internal1.col(idx) - MIRK6R_D5p1<Scalar>() * m_f_internal2.col(idx);
    y2prime.col(idx) = MIRK6R_A5p2<Scalar>() * (m_y.col(idx + 1) - m_y.col(idx)) * hinv - MIRK6R_B5p2<Scalar>() * m_f.col(idx + 1) - MIRK6R_B5p3<Scalar>() * m_f.col(idx) - MIRK6R_C5p2<Scalar>() * m_f_internal3.col(idx) - MIRK6R_C5p3<Scalar>() * m_f_internal1.col(idx) - MIRK6R_D5p2<Scalar>() * m_f_internal2.col(idx);
    y3prime.col(idx) = MIRK6R_A5p2<Scalar>() * (m_y.col(idx + 1) - m_y.col(idx)) * hinv - MIRK6R_B5p3<Scalar>() * m_f.col(idx + 1) - MIRK6R_B5p2<Scalar>() * m_f.col(idx) - MIRK6R_C5p3<Scalar>() * m_f_internal3.col(idx) - MIRK6R_C5p2<Scalar>() * m_f_internal1.col(idx) - MIRK6R_D5p2<Scalar>() * m_f_internal2.col(idx);
    y4prime.col(idx) = MIRK6R_A5p1<Scalar>() * (m_y.col(idx + 1) - m_y.col(idx)) * hinv - MIRK6R_B5p4<Scalar>() * m_f.col(idx + 1) - MIRK6R_B5p1<Scalar>() * m_f.col(idx) - MIRK6R_C5p4<Scalar>() * m_f_internal3.col(idx) - MIRK6R_C5p1<Scalar>() * m_f_internal1.col(idx) - MIRK6R_D5p1<Scalar>() * m_f_internal2.col(idx);
  }
  if constexpr (ParamsAtCompileTime == 0) {
# ifdef COLLOCATION_FUNCTION_RETURN
    f1 = m_fun(x1, y1);
    f2 = m_fun(x2, y2);
    f3 = m_fun(x3, y3);
    f4 = m_fun(x4, y4);
# else
    m_fun(x1, y1, f1);
    m_fun(x2, y2, f2);
    m_fun(x3, y3, f3);
    m_fun(x4, y4, f4);
# endif
  }
  else {
# ifdef COLLOCATION_FUNCTION_RETURN
    f1 = m_fun(x1, y1, this->m_p);
    f2 = m_fun(x2, y2, this->m_p);
    f3 = m_fun(x3, y3, this->m_p);
    f4 = m_fun(x4, y4, this->m_p);
# else
    m_fun(x1, y1, this->m_p, f1);
    m_fun(x2, y2, this->m_p, f2);
    m_fun(x3, y3, this->m_p, f3);
    m_fun(x4, y4, this->m_p, f4);
# endif
  }

  f1 = (y1prime - f1).abs();
  f2 = (y2prime - f2).abs();
  f3 = (y3prime - f3).abs();
  f4 = (y4prime - f4).abs();

  /* Loop through the arrays and located the maximum residual */
  for (Index i = 0; i < m_rows; ++i) {
    for (Index j = 0; j < m_cols - 1; ++j) {
      std::array<Scalar, 4> r{f1(i, j), f2(i, j), f3(i, j), f4(i, j)};
      std::array<Scalar, 4> y{y1(i, j), y2(i, j), y3(i, j), y4(i, j)};
      int idx = std::distance(r.begin(), std::max_element(r.begin(), r.end()));
      m_residuals(i, j) = m_h(idx) * r[idx];
      residual_norms(i, j) = abs(y[idx]);
    }
  }

  // x1 = ((y1prime - f1) / (1 + y1.abs())).matrix().colwise().stableNorm().array();
  // x2 = ((y2prime - f2) / (1 + y2.abs())).matrix().colwise().stableNorm().array();
  // x3 = ((y3prime - f3) / (1 + y3.abs())).matrix().colwise().stableNorm().array();
  // x4 = ((y4prime - f4) / (1 + y4.abs())).matrix().colwise().stableNorm().array();
  // m_residuals.array() = x1.max(x2.max(x3.max(x4))) * m_h;
}

template <typename Scalar, Index RowsAtCompileTime, Index ColsAtCompileTime, Index ParamsAtCompileTime, typename F, typename BC, typename FJ, typename BCJ>
void MIRK6<Scalar, RowsAtCompileTime, ColsAtCompileTime, ParamsAtCompileTime, F, BC, FJ, BCJ>::construct_global_jacobian() {
  Tensor<Scalar, 3> df_dy(m_rows, m_rows, m_cols);
  Tensor<Scalar, 3> df_dy_internal1(m_rows, m_rows, m_cols - 1);
  Tensor<Scalar, 3> df_dy_internal2(m_rows, m_rows, m_cols - 1);
  Tensor<Scalar, 3> df_dy_internal3(m_rows, m_rows, m_cols - 1);
  Array<Scalar, BCsAtCompileTime, RowsAtCompileTime> dbc_dya(m_rows + m_params, m_rows), dbc_dyb(m_rows + m_params, m_rows);

  Array<Scalar, Dynamic, 1> values((2 * m_rows + m_params) * (m_cols * m_rows + m_params));
  if constexpr (ParamsAtCompileTime == 0) {
# ifdef COLLOCATION_FUNCTION_JACOBIAN_RETURN
    df_dy = m_fun_jac(m_x, m_y);
    df_dy_internal1 = m_fun_jac(m_x_internal1, m_y_internal1);
    df_dy_internal2 = m_fun_jac(m_x_internal2, m_y_internal2);
    df_dy_internal3 = m_fun_jac(m_x_internal3, m_y_internal3);
# else
    m_fun_jac(m_x, m_y, df_dy);
    m_fun_jac(m_x_internal1, m_y_internal1, df_dy_internal1);
    m_fun_jac(m_x_internal2, m_y_internal2, df_dy_internal2);
    m_fun_jac(m_x_internal3, m_y_internal3, df_dy_internal3);
# endif
# ifdef COLLOCATION_BC_JACOBIAN_RETURN
    std::tie(dbc_dya, dbc_dyb) = m_bc_jac(m_y.col(0), m_y.col(m_cols - 1));
# else
    m_bc_jac(m_y.col(0), m_y.col(m_cols - 1), dbc_dya, dbc_dyb);
# endif

    for (Index idx = 0; idx < m_cols - 1; ++idx) {
      Map<const Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>> Ji(df_dy.data() + m_rows * m_rows * idx, m_rows, m_rows);
      Map<const Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>> Jip1(df_dy.data() + m_rows * m_rows * (idx + 1), m_rows, m_rows);
      Map<const Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>> J1(df_dy_internal1.data() + m_rows * m_rows * idx, m_rows, m_rows);
      Map<const Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>> J2(df_dy_internal2.data() + m_rows * m_rows * idx, m_rows, m_rows);
      Map<const Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>> J3(df_dy_internal3.data() + m_rows * m_rows * idx, m_rows, m_rows);

      Map<Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>> m1(values.data() + m_rows * m_rows * idx, m_rows, m_rows); // diagonal entries
      Map<Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>> m2(values.data() + (m_cols - 1) * m_rows * m_rows + m_rows * m_rows * idx, m_rows, m_rows); // off-diagonal entries

      m1.noalias() = - Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>::Identity(m_rows, m_rows) - m_h(idx) * (MIRK6_beta1<Scalar>() * Ji + MIRK6J_a12<Scalar>() * J3 + MIRK6J_a20<Scalar>() * J1 + MIRK6J_a19<Scalar>() * J2) + m_h(idx) * m_h(idx) * (J3 * (- MIRK6J_a4<Scalar>() * Ji + MIRK6J_a5<Scalar>() * J3 - MIRK6J_a8<Scalar>() * J2 + m_h(idx) * (MIRK6J_a1<Scalar>() * J3 - MIRK6J_a2<Scalar>() * J2) * Ji) + J1 * (- MIRK6J_a13<Scalar>() * Ji - MIRK6J_a14<Scalar>() * J1 + MIRK6J_a8<Scalar>() * J2 + m_h(idx) * (- MIRK6J_a6<Scalar>() * J1 + MIRK6J_a2<Scalar>() * J2) * Ji) + J2 * (- MIRK6J_a9<Scalar>() * Ji + MIRK6J_a7<Scalar>() * J3 - MIRK6J_a17<Scalar>() * J1 + m_h(idx) * (MIRK6J_a3<Scalar>() * J3 - MIRK6J_a10<Scalar>() * J1) * Ji));

      m2.noalias() = Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>::Identity(m_rows, m_rows) - m_h(idx) * (MIRK6_beta1<Scalar>() * Jip1 + MIRK6J_a20<Scalar>() * J3 + MIRK6J_a12<Scalar>() * J1 + MIRK6J_a19<Scalar>() * J2) + m_h(idx) * m_h(idx) * (J3 * (MIRK6J_a13<Scalar>() * Jip1 + MIRK6J_a14<Scalar>() * J3 - MIRK6J_a8<Scalar>() * J2 + m_h(idx) * (- MIRK6J_a6<Scalar>() * J3 + MIRK6J_a2<Scalar>() * J2) * Jip1) + J1 * (MIRK6J_a4<Scalar>() * Jip1 - MIRK6J_a5<Scalar>() * J1 + MIRK6J_a8<Scalar>() * J2 + m_h(idx) * (MIRK6J_a1<Scalar>() * J1 - MIRK6J_a2<Scalar>() * J2) * Jip1) + J2 * (MIRK6J_a9<Scalar>() * Jip1 + MIRK6J_a17<Scalar>() * J3 - MIRK6J_a7<Scalar>() * J1 + m_h(idx) * (- MIRK6J_a10<Scalar>() * J3 + MIRK6J_a3<Scalar>() * J1) * Jip1));
    }
    if constexpr (BCsAtCompileTime == Dynamic) {
      values.segment(2 * m_rows * m_rows * (m_cols - 1), m_rows * m_rows) = dbc_dya.reshaped();
      values.segment((2 * m_cols -1) * m_rows * m_rows, m_rows * m_rows) = dbc_dyb.reshaped();
    }
    else {
      values.template segment<RowsAtCompileTime * RowsAtCompileTime>(2 * m_rows * m_rows * (m_cols - 1)) = dbc_dya.reshaped();
      values.template segment<RowsAtCompileTime * RowsAtCompileTime>((2 * m_cols -1) * m_rows * m_rows) = dbc_dyb.reshaped();
    }
  }
  else {
    Tensor<Scalar, 3> df_dp(m_rows, m_params, m_cols);
    Tensor<Scalar, 3> df_dp_internal1(m_rows, m_params, m_cols - 1);
    Tensor<Scalar, 3> df_dp_internal2(m_rows, m_params, m_cols - 1);
    Tensor<Scalar, 3> df_dp_internal3(m_rows, m_params, m_cols - 1);
    Array<Scalar, BCsAtCompileTime, ParamsAtCompileTime> dbc_dp(m_rows + m_params, m_params);

# ifdef COLLOCATION_FUNCTION_JACOBIAN_RETURN
    std::tie(df_dy, df_dp) = m_fun_jac(m_x, m_y, this->m_p);
    std::tie(df_dy_internal1, df_dp_internal1) = m_fun_jac(m_x_internal1, m_y_internal1, this->m_p);
    std::tie(df_dy_internal2, df_dp_internal2) = m_fun_jac(m_x_internal2, m_y_internal2, this->m_p);
    std::tie(df_dy_internal3, df_dp_internal3) = m_fun_jac(m_x_internal3, m_y_internal3, this->m_p);
# else
    m_fun_jac(m_x, m_y, this->m_p, df_dy, df_dp);
    m_fun_jac(m_x_internal1, m_y_internal1, this->m_p, df_dy_internal1, df_dp_internal1);
    m_fun_jac(m_x_internal2, m_y_internal2, this->m_p, df_dy_internal2, df_dp_internal2);
    m_fun_jac(m_x_internal3, m_y_internal3, this->m_p, df_dy_internal3, df_dp_internal3);
# endif
# ifdef COLLOCATION_BC_JACOBIAN_RETURN
    std::tie(dbc_dya, dbc_dyb, dbc_dp) = m_bc_jac(m_y.col(0), m_y.col(m_cols - 1), this->m_p);
# else
    m_bc_jac(m_y.col(0), m_y.col(m_cols - 1), this->m_p, dbc_dya, dbc_dyb, dbc_dp);
# endif

    for (Index idx = 0; idx < m_cols - 1; ++idx) {
      Map<const Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>> Ji(df_dy.data() + m_rows * m_rows * idx, m_rows, m_rows);
      Map<const Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>> Jip1(df_dy.data() + m_rows * m_rows * (idx + 1), m_rows, m_rows);
      Map<const Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>> J1(df_dy_internal1.data() + m_rows * m_rows * idx, m_rows, m_rows);
      Map<const Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>> J2(df_dy_internal2.data() + m_rows * m_rows * idx, m_rows, m_rows);
      Map<const Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>> J3(df_dy_internal3.data() + m_rows * m_rows * idx, m_rows, m_rows);
      Map<const Matrix<Scalar, RowsAtCompileTime, ParamsAtCompileTime>> Ki(df_dp.data() + m_rows * m_params * idx, m_rows, m_params);
      Map<const Matrix<Scalar, RowsAtCompileTime, ParamsAtCompileTime>> Kip1(df_dp.data() + m_rows * m_params * (idx + 1), m_rows, m_params);
      Map<const Matrix<Scalar, RowsAtCompileTime, ParamsAtCompileTime>> K1(df_dp_internal1.data() + m_rows * m_params * idx, m_rows, m_params);
      Map<const Matrix<Scalar, RowsAtCompileTime, ParamsAtCompileTime>> K2(df_dp_internal2.data() + m_rows * m_params * idx, m_rows, m_params);
      Map<const Matrix<Scalar, RowsAtCompileTime, ParamsAtCompileTime>> K3(df_dp_internal3.data() + m_rows * m_params * idx, m_rows, m_params);

      Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime> temp1 = - MIRK6J_a6<Scalar>() * J3 + MIRK6J_a2<Scalar>() * J2;
      Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime> temp2 = MIRK6J_a1<Scalar>() * J3 - MIRK6J_a2<Scalar>() * J2;
      Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime> temp3 = MIRK6J_a1<Scalar>() * J1 - MIRK6J_a2<Scalar>() * J2;
      Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime> temp4 = - MIRK6J_a6<Scalar>() * J1 + MIRK6J_a2<Scalar>() * J2;
      Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime> temp5 = - MIRK6J_a10<Scalar>() * J3 + MIRK6J_a3<Scalar>() * J1;
      Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime> temp6 = MIRK6J_a3<Scalar>() * J3 - MIRK6J_a10<Scalar>() * J1;

      Map<Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>> m1(values.data() + m_rows * m_rows * idx, m_rows, m_rows); // diagonal entries
      Map<Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>> m2(values.data() + (m_cols - 1) * m_rows * m_rows + m_rows * m_rows * idx, m_rows, m_rows); // off-diagonal entries
      Map<Matrix<Scalar, RowsAtCompileTime, ParamsAtCompileTime>> m3(values.data() + 2 * (m_cols - 1) * m_rows * m_rows + m_rows * m_params * idx, m_rows, m_params); // parameter entries

      m1.noalias() = - Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>::Identity(m_rows, m_rows) - m_h(idx) * (MIRK6_beta1<Scalar>() * Ji + MIRK6J_a12<Scalar>() * J3 + MIRK6J_a20<Scalar>() * J1 + MIRK6J_a19<Scalar>() * J2) + m_h(idx) * m_h(idx) * (J3 * (- MIRK6J_a4<Scalar>() * Ji + MIRK6J_a5<Scalar>() * J3 - MIRK6J_a8<Scalar>() * J2 + m_h(idx) * temp2 * Ji) + J1 * (- MIRK6J_a13<Scalar>() * Ji - MIRK6J_a14<Scalar>() * J1 + MIRK6J_a8<Scalar>() * J2 + m_h(idx) * temp4 * Ji) + J2 * (- MIRK6J_a9<Scalar>() * Ji + MIRK6J_a7<Scalar>() * J3 - MIRK6J_a17<Scalar>() * J1 + m_h(idx) * temp6 * Ji));

      m2.noalias() = Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>::Identity(m_rows, m_rows) - m_h(idx) * (MIRK6_beta1<Scalar>() * Jip1 + MIRK6J_a20<Scalar>() * J3 + MIRK6J_a12<Scalar>() * J1 + MIRK6J_a19<Scalar>() * J2) + m_h(idx) * m_h(idx) * (J3 * (MIRK6J_a13<Scalar>() * Jip1 + MIRK6J_a14<Scalar>() * J3 - MIRK6J_a8<Scalar>() * J2 + m_h(idx) * temp1 * Jip1) + J1 * (MIRK6J_a4<Scalar>() * Jip1 - MIRK6J_a5<Scalar>() * J1 + MIRK6J_a8<Scalar>() * J2 + m_h(idx) * temp3 * Jip1) + J2 * (MIRK6J_a9<Scalar>() * Jip1 + MIRK6J_a17<Scalar>() * J3 - MIRK6J_a7<Scalar>() * J1 + m_h(idx) * temp5 * Jip1));

      m3.noalias() = - m_h(idx) * (MIRK6_beta1<Scalar>() * (Kip1 + Ki) + MIRK6_beta2<Scalar>() * (K3 + K1) + MIRK6_beta3<Scalar>() * K2) + m_h(idx) * m_h(idx) * (
        J3 * (MIRK6J_a13<Scalar>() * Kip1 - MIRK6J_a4<Scalar>() * Ki + MIRK6J_a15<Scalar>() * K3 - MIRK6J_a11<Scalar>() * K2 + m_h(idx) * (temp1 * Kip1 + temp2 * Ki)) + J1 * (MIRK6J_a4<Scalar>() * Kip1 - MIRK6J_a13<Scalar>() * Ki - MIRK6J_a15<Scalar>() * K1 + MIRK6J_a11<Scalar>() * K2 + m_h(idx) * (temp3 * Kip1 + temp4 * Ki)) + J2 * (MIRK6J_a9<Scalar>() * (Kip1 - Ki) + MIRK6J_a18<Scalar>() * (K3 - K1) + m_h(idx) * (temp5 * Kip1 + temp6 * Ki)));
    }
    if constexpr (BCsAtCompileTime == Dynamic) {
      values.segment((2 * m_rows + m_params) * (m_cols - 1) * m_rows, m_rows * (m_rows + m_params)) = dbc_dya.reshaped();
      values.segment((2 * m_cols -1) * m_rows * m_rows + m_params * m_cols * m_rows, m_rows * (m_rows + m_params)) = dbc_dyb.reshaped();
      values.tail(m_params * (m_rows + m_params)) = dbc_dp.reshaped();
    }
    else {
      values.template segment<RowsAtCompileTime * BCsAtCompileTime>((2 * m_rows + m_params) * (m_cols - 1) * m_rows) = dbc_dya.reshaped();
      values.template segment<RowsAtCompileTime * BCsAtCompileTime>((2 * m_cols -1) * m_rows * m_rows + m_params * m_cols * m_rows) = dbc_dyb.reshaped();
      values.template tail<ParamsAtCompileTime * BCsAtCompileTime>() = dbc_dp.reshaped();
    }
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
} // namespace nonlinearbvp

#endif // MIRK6_HPP
