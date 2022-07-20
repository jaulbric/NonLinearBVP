#ifndef MIRK_HPP
#define MIRK_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "NonLinearBVP/collocation/methods/MIRK_Traits.hpp"

using Eigen::Array;
using Eigen::ArrayXi;
using Eigen::Matrix;
using Eigen::MatrixBase;
using Eigen::Map;
using Eigen::ArrayBase;
using Eigen::DenseBase;
using Eigen::SparseMatrix;
using Eigen::Dynamic;
using Eigen::Index;

namespace nonlinearbvp { namespace collocation { namespace methods {

// Base class for monoimplicit collocation methods.
template <typename _Scalar, Index _RowsAtCompileTime, Index _ColsAtCompileTime, Index _ParamsAtCompileTime, typename F, typename BC, typename FJ, typename BCJ>
class MIRK : public MIRK_Traits<_Scalar, _RowsAtCompileTime, _ColsAtCompileTime, _ParamsAtCompileTime> {
public:
  using Traits = MIRK_Traits<_Scalar, _RowsAtCompileTime, _ColsAtCompileTime, _ParamsAtCompileTime>;
  using Scalar = typename Traits::Scalar;
  using Traits::RowsAtCompileTime;
  using Traits::ColsAtCompileTime;
  using Traits::SizeAtCompileTime;
  using Traits::ParamsAtCompileTime;
  using Traits::IntervalsAtCompileTime;
  using Traits::ResiduesAtCompileTime;
  using Traits::BCsAtCompileTime;

  typedef F Functor;
  typedef BC BCFunctor;
  typedef FJ FunctorJacobian;
  typedef BCJ BCFunctorJacobian;

  template <typename Derived1, typename Derived2, typename Derived3>
  MIRK(F& fun, BC& bc, FJ& fun_jac, BCJ& bc_jac, const ArrayBase<Derived1>& x_input, const ArrayBase<Derived2>&y_input, const ArrayBase<Derived3>& p_input) : Traits(p_input),
  m_fun{fun}, m_bc{bc}, m_fun_jac{fun_jac}, m_bc_jac{bc_jac},
  m_rows{y_input.rows()}, m_cols{y_input.cols()}, m_params{p_input.size()},
  m_x{x_input},
  m_h{m_cols - 1},
  m_y{y_input},
#ifdef COLLOCATION_FUNCTION_RETURN
  m_f{m_fun(m_x, m_y, p_input)},
#else
  m_f(m_rows, m_cols),
#endif
  m_residues{m_rows * m_cols + m_params}, m_residuals{m_rows, m_cols - 1},
  m_i_jac{(2 * m_rows + m_params) * (m_cols * m_rows + m_params)}, m_j_jac{(2 * m_rows + m_params) * (m_cols * m_rows + m_params)},
  m_jacobian{m_rows * m_cols + m_params, m_rows * m_cols + m_params},
  x{m_x}, h{m_h}, y{m_y}, f{m_f}, residues{m_residues}, residuals{m_residuals}, residual_norms{m_rows, m_cols - 1}, jacobian{m_jacobian},
  collocation_residues{m_residues.data(), m_rows, m_cols - 1}, bc_residues{m_residues.data() + m_rows * (m_cols - 1), m_rows + m_params} {
#ifndef COLLOCATION_FUNCTION_RETURN
    m_fun(m_x, m_y, p_input, m_f);
#endif
    assert(m_x.size() == m_cols);
    if constexpr (IntervalsAtCompileTime == Dynamic) {
      m_h = m_x.tail(m_cols - 1) - m_x.head(m_cols - 1);
    }
    else {
      m_h = m_x.template tail<IntervalsAtCompileTime>() - m_x.template head<IntervalsAtCompileTime>();
    }
    assert((m_h >= 0).all());
    compute_jacobian_indices();
  };

  template <typename Derived1, typename Derived2>
  MIRK(F& fun, BC& bc, FJ& fun_jac, BCJ& bc_jac, const ArrayBase<Derived1>& x_input, const ArrayBase<Derived2>&y_input) : Traits(),
  m_fun{fun}, m_bc{bc}, m_fun_jac{fun_jac}, m_bc_jac{bc_jac},
  m_rows{y_input.rows()}, m_cols{y_input.cols()}, m_params{0},
  m_x{x_input},
  m_h{m_cols - 1},
  m_y{y_input},
#ifdef COLLOCATION_FUNCTION_RETURN
  m_f{m_fun(m_x, m_y)},
#else
  m_f(m_rows, m_cols),
#endif
  m_residues{m_rows * m_cols}, m_residuals{m_rows, m_cols - 1},
  m_i_jac{2 * m_rows * m_cols * m_rows}, m_j_jac{2 * m_rows * m_cols * m_rows},
  m_jacobian{m_rows * m_cols, m_rows * m_cols},
  x{m_x}, h{m_h}, y{m_y}, f{m_f}, residues{m_residues}, residuals{m_residuals}, residual_norms{m_rows, m_cols - 1}, jacobian{m_jacobian},
  collocation_residues{m_residues.data(), m_rows, m_cols - 1}, bc_residues{m_residues.data() + m_rows * (m_cols - 1), m_rows} {
#ifndef COLLOCATION_FUNCTION_RETURN
    m_fun(m_x, m_y, m_f);
#endif
    assert(m_x.size() == m_cols);
    if constexpr (IntervalsAtCompileTime == Dynamic) {
      m_h = m_x.tail(m_cols - 1) - m_x.head(m_cols - 1);
    }
    else {
      m_h = m_x.template tail<IntervalsAtCompileTime>() - m_x.template head<IntervalsAtCompileTime>();
    }
    assert((m_h >= 0).all());
    compute_jacobian_indices();
  };

  ~MIRK() = default;

protected:
  F& m_fun;
  BC& m_bc;
  FJ& m_fun_jac;
  BCJ& m_bc_jac;

  const Index m_rows;
  Index m_cols;
  const Index m_params;

  Array<Scalar, ColsAtCompileTime, 1> m_x;
  Array<Scalar, IntervalsAtCompileTime, 1> m_h;
  Array<Scalar, RowsAtCompileTime, ColsAtCompileTime> m_y;
  Array<Scalar, RowsAtCompileTime, ColsAtCompileTime> m_f;

  Matrix<Scalar, ResiduesAtCompileTime, 1> m_residues;
  Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> m_residuals;
  Array<int, Dynamic, 1> m_i_jac;
  Array<int, Dynamic, 1> m_j_jac;
  SparseMatrix<Scalar> m_jacobian;

  void compute_jacobian_indices();

public:
  // Read only access to internal data structures
  const Array<Scalar, ColsAtCompileTime, 1>& x;
  const Array<Scalar, IntervalsAtCompileTime, 1>& h;
  const Array<Scalar, RowsAtCompileTime, ColsAtCompileTime>& y;
  const Array<Scalar, RowsAtCompileTime, ColsAtCompileTime>& f;
  const Matrix<Scalar, ResiduesAtCompileTime, 1>& residues;
  const Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime>& residuals;
  Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime> residual_norms;
  const SparseMatrix<Scalar>& jacobian;
  Map<const Array<Scalar, RowsAtCompileTime, IntervalsAtCompileTime>> collocation_residues;
  Map<const Array<Scalar, BCsAtCompileTime, 1>> bc_residues;

  Matrix<Scalar, ResiduesAtCompileTime, 1> Y() const {
    Matrix<Scalar, ResiduesAtCompileTime, 1> result(m_rows * m_cols + m_params);
    if constexpr (ParamsAtCompileTime == 0) {
      result.array() = m_y.reshaped();
    }
    else {
      if constexpr (SizeAtCompileTime == Dynamic) {
        result.head(m_rows * m_cols).array() = m_y.reshaped();
      }
      else {
        result.template head<SizeAtCompileTime>().array() = m_y.reshaped();
      }
      if constexpr (ParamsAtCompileTime == Dynamic) {
        result.tail(m_params).array() = this->m_p;
      }
      else {
        result.template tail<ParamsAtCompileTime>().array() = this->m_p;
      }
    }
    return result;
  }

  template <typename Derived>
  void step(const DenseBase<Derived>& dY) {
    assert(dY.size() == m_rows * m_cols + m_params);
    Map<const Array<Scalar, RowsAtCompileTime, ColsAtCompileTime>> y_step(dY.derived().data(), m_rows, m_cols);
    m_y += y_step;
    if constexpr (ParamsAtCompileTime != 0) {
      if constexpr (ParamsAtCompileTime == Dynamic) {
        this->m_p += dY.tail(m_params).array();
      }
      else {
        this->m_p += dY.template tail<ParamsAtCompileTime>().array();
      }
#ifdef COLLOCATION_FUNCTION_RETURN
      m_f = m_fun(m_x, m_y, this->m_p);
#else
      m_fun(m_x, m_y, this->m_p, m_f);
#endif
    }
    else {
#ifdef COLLOCATION_FUNCTION_RETURN
      m_f = m_fun(m_x, m_y);
#else
      m_fun(m_x, m_y, m_f);
#endif
    }
  }

  template <typename Derived>
  void apply_singular(const MatrixBase<Derived>& B) {
    assert(B.rows() == m_rows);
    assert(B.cols() == m_rows);
    m_y.col(0).matrix() = B * m_y.col(0).matrix();
  }

  template <typename Derived1, typename Derived2>
  void modify_mesh(const ArrayBase<Derived1>& x_new, const ArrayBase<Derived2>& y_new) {
    static_assert(ColsAtCompileTime == Dynamic);
    assert(x_new.size() == y_new.cols());
    m_cols = y_new.cols();
    m_x = x_new;
    m_y = y_new;
    m_h = x_new.tail(m_cols - 1) - x_new.head(m_cols - 1);
    if constexpr (ParamsAtCompileTime == 0) {
#ifdef COLLOCATION_FUNCTION_RETURN
      m_f = m_fun(m_x, m_y);
#else
      m_f.resize(m_rows, m_cols);
      m_fun(m_x, m_y, m_f);
#endif
    }
    else {
#ifdef COLLOCATION_FUNCTION_RETURN
      m_f = m_fun(m_x, m_y, this->m_p);
#else
      m_f.resize(m_rows, m_cols);
      m_fun(m_x, m_y, this->m_p, m_f);
#endif
    }
    m_residues.resize(m_rows * m_cols + m_params);
    m_residuals.resize(m_rows, m_cols - 1);
    residual_norms.resize(m_rows, m_cols - 1);
    m_i_jac.resize((2 * m_rows + m_params) * (m_cols * m_rows + m_params));
    m_j_jac.resize((2 * m_rows + m_params) * (m_cols * m_rows + m_params));
    m_jacobian.resize(m_rows * m_cols + m_params, m_rows * m_cols + m_params);
    // new (&collocation_residues) Map<Array<Scalar, RowsAtCompileTime, Dynamic>>(m_residues.data(), m_rows, m_cols - 1);
    // new (&bc_residues) Map<Array<Scalar, BCsAtCompileTime, 1>>(m_residues.data() + m_rows * (m_cols - 1), m_rows + m_params);
    compute_jacobian_indices();
  }

  Index rows() const {return m_rows;}
  Index cols() const {return m_cols;}
  Index params() const {return m_params;}
};

template <typename F, typename BC, typename FJ, typename BCJ, typename Derived1, typename Derived2, typename Derived3>
MIRK(F& fun, BC& bc, FJ& fun_jac, BCJ& bc_jac, const ArrayBase<Derived1>& x_input, const ArrayBase<Derived2>&y_input, const ArrayBase<Derived3>& p_input) -> MIRK<typename Derived2::Scalar, Derived2::RowsAtCompileTime, Derived2::ColsAtCompileTime, Derived3::SizeAtCompileTime, F, BC, FJ, BCJ>;

template <typename F, typename BC, typename FJ, typename BCJ, typename Derived1, typename Derived2>
MIRK(F& fun, BC& bc, FJ& fun_jac, BCJ& bc_jac, const ArrayBase<Derived1>& x_input, const ArrayBase<Derived2>&y_input) -> MIRK<typename Derived2::Scalar, Derived2::RowsAtCompileTime, Derived2::ColsAtCompileTime, 0, F, BC, FJ, BCJ>;

template <typename _Scalar, Index _RowsAtCompileTime, Index _ColsAtCompileTime, Index _ParamsAtCompileTime, typename F, typename BC, typename FJ, typename BCJ>
void MIRK<_Scalar, _RowsAtCompileTime, _ColsAtCompileTime, _ParamsAtCompileTime, F, BC, FJ, BCJ>::compute_jacobian_indices() {
  auto i_col = ArrayXi::LinSpaced(m_rows, 0, m_rows - 1).replicate(m_rows * (m_cols - 1), 1) + m_rows * Eigen::ArrayXi::LinSpaced((m_cols - 1) * m_rows * m_rows, 0, m_cols - 2);
  auto j_col = ArrayXi::LinSpaced((m_cols - 1) * m_rows * m_rows, 0, (m_cols - 1) * m_rows - 1);

  auto i_bc = ArrayXi::LinSpaced(m_rows + m_params, (m_cols - 1) * m_rows, m_cols * m_rows + m_params - 1).replicate(m_rows, 1);
  auto j_bc = ArrayXi::LinSpaced(m_rows * (m_rows + m_params), 0, m_rows - 1);

  auto i_p_col = ArrayXi::LinSpaced((m_cols - 1) * m_rows, 0, (m_cols - 1) * m_rows - 1).replicate(m_params, 1);
  auto j_p_col = ArrayXi::LinSpaced((m_cols - 1) * m_rows * m_params, m_cols * m_rows, m_cols * m_rows + m_params - 1);

  auto i_p_bc = ArrayXi::LinSpaced(m_rows + m_params, (m_cols - 1) * m_rows, m_cols * m_rows + m_params - 1).replicate(m_params, 1);
  auto j_p_bc = ArrayXi::LinSpaced((m_rows + m_params) * m_params, m_cols * m_rows, m_cols * m_rows + m_params - 1);

  m_i_jac << i_col, i_col, i_p_col, i_bc, i_bc, i_p_bc;
  m_j_jac << j_col, j_col + m_rows, j_p_col, j_bc, j_bc + (m_cols - 1) * m_rows, j_p_bc;
}

} // namespace methods
} // namespace collocation
} // namespace nonlinearbvp

#endif // MIRK_HPP
