#ifndef BVP4C_HPP
#define BVP4C_HPP

#include <Eigen/Dense>
#include <Eigen/QR>

#include "methods/MIRK4.hpp"
#include "detail/Generic_Solver.hpp"
#include "detail/wrapped_fun.hpp"
#include "detail/estimate_fun_jac.hpp"
#include "detail/estimate_bc_jac.hpp"

namespace collocation {

using detail::BVP_wrapped_fun;
using detail::BVP_wrapped_fun_jac;
using detail::estimate_fun_jac;
using detail::estimate_bc_jac;

// Class containing static methods for solver with unknown parameters and singular term
template <typename Scalar, Index RowsAtCompileTime, Index ParamsAtCompileTime>
class bvp4c {
  public:
    // type alias for method
    template <typename F, typename BC, typename FJ, typename BCJ>
    using Method = typename methods::MIRK4<Scalar, RowsAtCompileTime, Dynamic, ParamsAtCompileTime, F, BC, FJ, BCJ>;

    template <typename F, typename BC, typename FJ, typename BCJ>
    using MethodS = typename methods::MIRK4<Scalar, RowsAtCompileTime, Dynamic, ParamsAtCompileTime, BVP_wrapped_fun<F, Scalar, RowsAtCompileTime, ParamsAtCompileTime>, BC, BVP_wrapped_fun_jac<FJ, Scalar, RowsAtCompileTime, ParamsAtCompileTime>, BCJ>;

    template <typename F, typename BC>
    using MethodE = typename methods::MIRK4<Scalar, RowsAtCompileTime, Dynamic, ParamsAtCompileTime, F, BC, estimate_fun_jac<F, Scalar, RowsAtCompileTime, ParamsAtCompileTime, false>, estimate_bc_jac<BC, Scalar, RowsAtCompileTime, ParamsAtCompileTime>>;

    template <typename F, typename BC>
    using MethodSE = typename methods::MIRK4<Scalar, RowsAtCompileTime, Dynamic, ParamsAtCompileTime, BVP_wrapped_fun<F, Scalar, RowsAtCompileTime, ParamsAtCompileTime>, BC, estimate_fun_jac<F, Scalar, RowsAtCompileTime, ParamsAtCompileTime, true>, estimate_bc_jac<BC, Scalar, RowsAtCompileTime, ParamsAtCompileTime>>;

    // Most generic solver
    template <typename F, typename BC, typename FJ, typename BCJ, typename Derived1, typename Derived2, typename Derived3, typename Derived4>
    static BVP_Result<MethodS<F, BC, FJ, BCJ>>
    solve(F& fun, BC& bc, FJ& fun_jac, BCJ& bc_jac, const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y, const ArrayBase<Derived3>& p, const MatrixBase<Derived4>& S, Scalar tol, Scalar bc_tol, int max_nodes);

    // No singular term
    template <typename F, typename BC, typename FJ, typename BCJ, typename Derived1, typename Derived2, typename Derived3>
    static BVP_Result<Method<F, BC, FJ, BCJ>>
    solve(F& fun, BC& bc, FJ& fun_jac, BCJ& bc_jac, const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y, const ArrayBase<Derived3>& p, Scalar tol, Scalar bc_tol, int max_nodes);

    // No jacobians, just a redirect to the general case
    template <typename F, typename BC, typename Derived1, typename Derived2, typename Derived3, typename Derived4>
    static BVP_Result<MethodSE<F, BC>>
    solve(F& fun, BC& bc, const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y, const ArrayBase<Derived3>& p, const MatrixBase<Derived4>& S, Scalar tol, Scalar bc_tol, int max_nodes);

    // No jacobians and no singular term, just a redirect to the general case with no singular term
    template <typename F, typename BC, typename Derived1, typename Derived2, typename Derived3>
    static BVP_Result<MethodE<F, BC>>
    solve(F& fun, BC& bc, const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y, const ArrayBase<Derived3>& p, Scalar tol, Scalar bc_tol, int max_nodes);
};

// Function definitions
// Singular, Jacobians supplied
template <typename Scalar, Index RowsAtCompileTime, Index ParamsAtCompileTime>
template <typename F, typename BC, typename FJ, typename BCJ, typename Derived1, typename Derived2, typename Derived3, typename Derived4>
auto bvp4c<Scalar, RowsAtCompileTime, ParamsAtCompileTime>::solve(F& fun, BC& bc, FJ& fun_jac, BCJ& bc_jac, const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y, const ArrayBase<Derived3>& p, const MatrixBase<Derived4>& S, Scalar tol, Scalar bc_tol, int max_nodes) -> BVP_Result<MethodS<F, BC, FJ, BCJ>> {
  using SType = Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>;
  SType B = SType::Identity(S.rows(), S.cols()) - S.completeOrthogonalDecomposition().pseudoInverse() * S;
  const_cast<ArrayBase<Derived2>&>(y).col(0).matrix() = B * y.col(0).matrix();
  SType D = (SType::Identity(S.rows(), S.cols()) - S).completeOrthogonalDecomposition().pseudoInverse();
  detail::BVP_wrapped_fun<F, Scalar, RowsAtCompileTime, ParamsAtCompileTime> fun_wrapped(fun, x(0), S, D);
  detail::BVP_wrapped_fun_jac<FJ, Scalar, RowsAtCompileTime, ParamsAtCompileTime> fun_jac_wrapped(fun_jac, x(0), S, D);
  detail::Generic_Solver<MethodS<F, BC, FJ, BCJ>, true> algorithm(fun_wrapped, bc, fun_jac_wrapped, bc_jac, x, y, p, B);
  return algorithm.solve(tol, bc_tol, max_nodes);
}

// Not singular, Jacobians supplied
template <typename Scalar, Index RowsAtCompileTime, Index ParamsAtCompileTime>
template <typename F, typename BC, typename FJ, typename BCJ, typename Derived1, typename Derived2, typename Derived3>
auto bvp4c<Scalar, RowsAtCompileTime, ParamsAtCompileTime>::solve(F& fun, BC& bc, FJ& fun_jac, BCJ& bc_jac, const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y, const ArrayBase<Derived3>& p, Scalar tol, Scalar bc_tol, int max_nodes) -> BVP_Result<Method<F, BC, FJ, BCJ>> {
  detail::Generic_Solver<Method<F, BC, FJ, BCJ>, false> algorithm(fun, bc, fun_jac, bc_jac, x, y, p);
  return algorithm.solve(tol, bc_tol, max_nodes);
}

// Singular, estimate Jacobians
template <typename Scalar, Index RowsAtCompileTime, Index ParamsAtCompileTime>
template <typename F, typename BC, typename Derived1, typename Derived2, typename Derived3, typename Derived4>
auto bvp4c<Scalar, RowsAtCompileTime, ParamsAtCompileTime>::solve(F& fun, BC& bc, const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y, const ArrayBase<Derived3>& p, const MatrixBase<Derived4>& S, Scalar tol, Scalar bc_tol, int max_nodes) -> BVP_Result<MethodSE<F, BC>> {
  using SType = Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>;
  SType B = SType::Identity(S.rows(), S.cols()) - S.completeOrthogonalDecomposition().pseudoInverse() * S;
  const_cast<ArrayBase<Derived2>&>(y).col(0).matrix() = B * y.col(0).matrix();
  SType D = (SType::Identity() - S).completeOrthogonalDecomposition().pseudoInverse();
  detail::BVP_wrapped_fun<F, Scalar, RowsAtCompileTime, ParamsAtCompileTime> fun_wrapped(fun, x(0), S, D);
  detail::estimate_fun_jac<F, Scalar, RowsAtCompileTime, ParamsAtCompileTime, true> fun_jac(fun, x(0), S, D);
  detail::estimate_bc_jac<BC, Scalar, RowsAtCompileTime, ParamsAtCompileTime> bc_jac(bc);
  detail::Generic_Solver<MethodSE<F, BC>, true> algorithm(fun_wrapped, bc, fun_jac, bc_jac, x, y, p, B);
  return algorithm.solve(tol, bc_tol, max_nodes);
}

// Not singular, estimate Jacobians
template <typename Scalar, Index RowsAtCompileTime, Index ParamsAtCompileTime>
template <typename F, typename BC, typename Derived1, typename Derived2, typename Derived3>
auto bvp4c<Scalar, RowsAtCompileTime, ParamsAtCompileTime>::solve(F& fun, BC& bc, const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y, const ArrayBase<Derived3>& p, Scalar tol, Scalar bc_tol, int max_nodes) -> BVP_Result<MethodE<F, BC>> {
  detail::estimate_fun_jac<F, Scalar, RowsAtCompileTime, ParamsAtCompileTime, false> fun_jac(fun);
  detail::estimate_bc_jac<BC, Scalar, RowsAtCompileTime, ParamsAtCompileTime> bc_jac(bc);
  detail::Generic_Solver<MethodE<F, BC>, false> algorithm(fun, bc, fun_jac, bc_jac, x, y, p);
  return algorithm.solve(tol, bc_tol, max_nodes);
}

// Partial specialization for case with no unknown parameters
template <typename Scalar, Index RowsAtCompileTime>
class bvp4c<Scalar, RowsAtCompileTime, 0> {
  public:
    // type alias for method
    template <typename F, typename BC, typename FJ, typename BCJ>
    using Method = typename methods::MIRK4<Scalar, RowsAtCompileTime, Dynamic, 0, F, BC, FJ, BCJ>;

    template <typename F, typename BC, typename FJ, typename BCJ>
    using MethodS = typename methods::MIRK4<Scalar, RowsAtCompileTime, Dynamic, 0, BVP_wrapped_fun<F, Scalar, RowsAtCompileTime, 0>, BC, BVP_wrapped_fun_jac<FJ, Scalar, RowsAtCompileTime, 0>, BCJ>;

    template <typename F, typename BC>
    using MethodE = typename methods::MIRK4<Scalar, RowsAtCompileTime, Dynamic, 0, F, BC, estimate_fun_jac<F, Scalar, RowsAtCompileTime, 0, false>, estimate_bc_jac<BC, Scalar, RowsAtCompileTime, 0>>;

    template <typename F, typename BC>
    using MethodSE = typename methods::MIRK4<Scalar, RowsAtCompileTime, Dynamic, 0, BVP_wrapped_fun<F, Scalar, RowsAtCompileTime, 0>, BC, estimate_fun_jac<F, Scalar, RowsAtCompileTime, 0, true>, estimate_bc_jac<BC, Scalar, RowsAtCompileTime, 0>>;

    // Most generic solver
    template <typename F, typename BC, typename FJ, typename BCJ, typename Derived1, typename Derived2, typename Derived3>
    static BVP_Result<MethodS<F, BC, FJ, BCJ>>
    solve(F& fun, BC& bc, FJ& fun_jac, BCJ& bc_jac, const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y, const MatrixBase<Derived3>& S, Scalar tol, Scalar bc_tol, int max_nodes);

    // No singular term
    template <typename F, typename BC, typename FJ, typename BCJ, typename Derived1, typename Derived2>
    static BVP_Result<Method<F, BC, FJ, BCJ>>
    solve(F& fun, BC& bc, FJ& fun_jac, BCJ& bc_jac, const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y, Scalar tol, Scalar bc_tol, int max_nodes);

    // No jacobians, just a redirect to the general case
    template <typename F, typename BC, typename Derived1, typename Derived2, typename Derived3>
    static BVP_Result<MethodSE<F, BC>>
    solve(F& fun, BC& bc, const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y, const MatrixBase<Derived3>& S, Scalar tol, Scalar bc_tol, int max_nodes);

    // No jacobians and no singular term, just a redirect to the general case with no singular term
    template <typename F, typename BC, typename Derived1, typename Derived2>
    static BVP_Result<MethodE<F, BC>>
    solve(F& fun, BC& bc, const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y, Scalar tol, Scalar bc_tol, int max_nodes);
};

// Function definitions
// Singular, Jacobians supplied
template <typename Scalar, Index RowsAtCompileTime>
template <typename F, typename BC, typename FJ, typename BCJ, typename Derived1, typename Derived2, typename Derived3>
auto bvp4c<Scalar, RowsAtCompileTime, 0>::solve(F& fun, BC& bc, FJ& fun_jac, BCJ& bc_jac, const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y, const MatrixBase<Derived3>& S, Scalar tol, Scalar bc_tol, int max_nodes) -> BVP_Result<MethodS<F, BC, FJ, BCJ>> {
  using SType = Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>;
  SType B = SType::Identity(S.rows(), S.cols()) - S.completeOrthogonalDecomposition().pseudoInverse() * S;
  const_cast<ArrayBase<Derived2>&>(y).col(0).matrix() = B * y.col(0).matrix();
  SType D = (SType::Identity(S.rows(), S.cols()) - S).completeOrthogonalDecomposition().pseudoInverse();
  detail::BVP_wrapped_fun<F, Scalar, RowsAtCompileTime, 0> fun_wrapped(fun, x(0), S, D);
  detail::BVP_wrapped_fun_jac<FJ, Scalar, RowsAtCompileTime, 0> fun_jac_wrapped(fun_jac, x(0), S, D);
  detail::Generic_Solver<MethodS<F, BC, FJ, BCJ>, true> algorithm(fun_wrapped, bc, fun_jac_wrapped, bc_jac, x, y, B);
  return algorithm.solve(tol, bc_tol, max_nodes);
}

// Not singular, Jacobians supplied
template <typename Scalar, Index RowsAtCompileTime>
template <typename F, typename BC, typename FJ, typename BCJ, typename Derived1, typename Derived2>
auto bvp4c<Scalar, RowsAtCompileTime, 0>::solve(F& fun, BC& bc, FJ& fun_jac, BCJ& bc_jac, const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y, Scalar tol, Scalar bc_tol, int max_nodes) -> BVP_Result<Method<F, BC, FJ, BCJ>> {
  detail::Generic_Solver<Method<F, BC, FJ, BCJ>, false> algorithm(fun, bc, fun_jac, bc_jac, x, y);
  return algorithm.solve(tol, bc_tol, max_nodes);
}

// Singular, estimate jacobians
template <typename Scalar, Index RowsAtCompileTime>
template <typename F, typename BC, typename Derived1, typename Derived2, typename Derived3>
auto bvp4c<Scalar, RowsAtCompileTime, 0>::solve(F& fun, BC& bc, const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y, const MatrixBase<Derived3>& S, Scalar tol, Scalar bc_tol, int max_nodes) -> BVP_Result<MethodSE<F, BC>> {
  using SType = Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime>;
  SType B = SType::Identity(S.rows(), S.cols()) - S.completeOrthogonalDecomposition().pseudoInverse() * S;
  const_cast<ArrayBase<Derived2>&>(y).col(0).matrix() = B * y.col(0).matrix();
  SType D = (SType::Identity(S.rows(), S.cols()) - S).completeOrthogonalDecomposition().pseudoInverse();
  detail::BVP_wrapped_fun<F, Scalar, RowsAtCompileTime, 0> fun_wrapped(fun, x(0), S, D);
  detail::estimate_fun_jac<F, Scalar, RowsAtCompileTime, 0, true> fun_jac(fun, x(0), S, D);
  detail::estimate_bc_jac<BC, Scalar, RowsAtCompileTime, 0> bc_jac(bc);
  detail::Generic_Solver<MethodSE<F, BC>, true> algorithm(fun_wrapped, bc, fun_jac, bc_jac, x, y, B);
  return algorithm.solve(tol, bc_tol, max_nodes);
}

// Not singular, estimate Jacobians
template <typename Scalar, Index RowsAtCompileTime>
template <typename F, typename BC, typename Derived1, typename Derived2>
auto bvp4c<Scalar, RowsAtCompileTime, 0>::solve(F& fun, BC& bc, const ArrayBase<Derived1>& x, const ArrayBase<Derived2>& y, Scalar tol, Scalar bc_tol, int max_nodes) -> BVP_Result<MethodE<F, BC>> {
  detail::estimate_fun_jac<F, Scalar, RowsAtCompileTime, 0, false> fun_jac(fun);
  detail::estimate_bc_jac<BC, Scalar, RowsAtCompileTime, 0> bc_jac(bc);
  detail::Generic_Solver<MethodE<F, BC>, false> algorithm(fun, bc, fun_jac, bc_jac, x, y);
  return algorithm.solve(tol, bc_tol, max_nodes);
}

} // namespace collocation

#endif // BVP4C_HPP
