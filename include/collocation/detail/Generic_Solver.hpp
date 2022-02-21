#ifndef COLLOCATION_DETAIL_GENERIC_SOLVER_HPP
#define COLLOCATION_DETAIL_GENERIC_SOLVER_HPP

#ifndef COLLOCATION_AUTODIFF
#define COLLOCATION_AUTODIFF 0
#endif

#ifndef COLLOCATION_VERBOSITY
#define COLLOCATION_VERBOSITY 0
#endif

#ifndef COLLOCATION_SOLVER
#define COLLOCATION_SOLVER 0
#endif

#include <boost/math/tools/precision.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <limits>

#include "collocation/BVP_Result.hpp"
#include "NonLinearOptimization/PowellHybrid.hpp"
#include "NonLinearOptimization/NewtonRaphson.hpp"
#include "constants/constants.hpp"
#include "tools.hpp"

#ifndef BVPSOLVERSPACE
#define BVPSOLVERSPACE
namespace BVPSolverSpace {
  enum Status {Success = 1,
               MaximumNodesExceeded = 2,
               SingularJacobian = 3,
               TolTooSmall = 4,
               NotMakingProgressResiduals = 5
              };
}
#endif
#include "print_functions.hpp"

namespace collocation { namespace detail {

using Eigen::Array;
using Eigen::ArrayBase;
using Eigen::Map;
using Eigen::Matrix;
using Eigen::MatrixBase;
using Eigen::Index;
using Eigen::Dynamic;
using Eigen::all;

using boost::math::constants::half;
using boost::math::constants::third;
using boost::math::constants::two_thirds;

template <typename Method, bool Singular>
class Generic_Solver {
  public:
    typedef typename Method::Scalar Scalar;
    typedef typename Method::Functor Functor;
    typedef typename Method::BCFunctor BCFunctor;
    typedef typename Method::FunctorJacobian FunctorJacobian;
    typedef typename Method::BCFunctorJacobian BCFunctorJacobian;
    // typedef class NewtonRaphson<Method, Singular> Solver;
    static constexpr Index RowsAtCompileTime = Method::RowsAtCompileTime;
    static constexpr Index ParamsAtCompileTime = Method::ParamsAtCompileTime;

    template <typename Derived1, typename Derived2, typename Derived3, typename Derived4>
    Generic_Solver(Functor& fun,
        BCFunctor& bc,
        FunctorJacobian& fun_jac,
        BCFunctorJacobian& bc_jac,
        const ArrayBase<Derived1>& x_input,
        const ArrayBase<Derived2>& y_input,
        const ArrayBase<Derived3>& p_input,
        const MatrixBase<Derived4>& B) : m_method{fun, bc, fun_jac, bc_jac, x_input, y_input, p_input}, m_solver{m_method, B}, m_nodes_added{0}, m_nodes_removed{0} {
      static_assert(Singular == true, "Singular == false but a singular term has been supplied.");
      static_assert(ParamsAtCompileTime == Derived3::SizeAtCompileTime, "Declared number of unknown parameters does not match supplied number of parameters");
    }

    template <typename Derived1, typename Derived2, typename Derived3>
    Generic_Solver(Functor& fun,
        BCFunctor& bc,
        FunctorJacobian& fun_jac,
        BCFunctorJacobian& bc_jac,
        const ArrayBase<Derived1>& x_input,
        const ArrayBase<Derived2>& y_input,
        const ArrayBase<Derived3>& p_input) : m_method{fun, bc, fun_jac, bc_jac, x_input, y_input, p_input}, m_solver{m_method}, m_nodes_added{0}, m_nodes_removed{0} {
      static_assert(Singular == false, "Singular == true but no singular term has been supplied.");
      static_assert(ParamsAtCompileTime == Derived3::SizeAtCompileTime, "Declared number of unknown parameters does not match supplied number of parameters");
    }

    template <typename Derived1, typename Derived2, typename Derived3>
    Generic_Solver(Functor& fun,
        BCFunctor& bc,
        FunctorJacobian& fun_jac,
        BCFunctorJacobian& bc_jac,
        const ArrayBase<Derived1>& x_input,
        const ArrayBase<Derived2>& y_input,
        const MatrixBase<Derived3>& B) : m_method{fun, bc, fun_jac, bc_jac, x_input, y_input}, m_solver{m_method, B}, m_nodes_added{0}, m_nodes_removed{0} {
      static_assert(Singular == true, "Singular == false but a singular term has been supplied");
      static_assert(ParamsAtCompileTime == 0, "Declared number of unknown parameters does not match supplied number of parameters");
    }

    template <typename Derived1, typename Derived2>
    Generic_Solver(Functor& fun,
        BCFunctor& bc,
        FunctorJacobian& fun_jac,
        BCFunctorJacobian& bc_jac,
        const ArrayBase<Derived1>& x_input,
        const ArrayBase<Derived2>& y_input) : m_method{fun, bc, fun_jac, bc_jac, x_input, y_input}, m_solver{m_method}, m_nodes_added{0}, m_nodes_removed{0} {
      static_assert(Singular == false, "Singular == true but no singular term has been supplied.");
      static_assert(ParamsAtCompileTime == 0, "Declared number of unknown parameters does not match supplied number of parameters");
    }

    BVP_Result<Method> solve(const Scalar tol, const Scalar bc_tol, const int max_nodes);

  private:
    Method m_method;
#if COLLOCATION_SOLVER == 0
    PowellHybrid<Method, Singular> m_solver;
#else
    NewtonRaphson<Method, Singular> m_solver;
#endif
    static constexpr Scalar EPS = boost::math::tools::epsilon<Scalar>();
    static constexpr int m_max_iterations = 10; // Max number of iterations of the main loop
    int m_nodes_added;
    int m_nodes_removed;
    static constexpr Scalar m_tol_factor = 2 * tools::three_raise_x(Scalar(Method::residual_order));

    void modify_mesh(Scalar tol);
};

template <typename Method, bool Singular>
BVP_Result<Method> Generic_Solver<Method, Singular>::solve(Scalar tol, const Scalar bc_tol, const int max_nodes) {
  if (tol < 100 * EPS) {
    // const_cast<Scalar>(tol) = 100 * EPS;
    tol = 100 * EPS;
  }
#if COLLOCATION_VERBOSITY > 1
  print_iteration_header();
#endif
  m_solver.parameters.tol = tenth<Scalar>() * tol;
  m_solver.parameters.bc_tol = bc_tol;
  m_solver.parameters.max_njev = 4;

  BVPSolverSpace::Status status = BVPSolverSpace::TolTooSmall;
  Scalar max_bc_res = std::numeric_limits<Scalar>::max();
  Scalar max_res = std::numeric_limits<Scalar>::max();
  int iteration;
  for (iteration = 0; iteration < m_max_iterations; ++iteration) {

#if COLLOCATION_SOLVER == 0
    PowellHybridSolverSpace::Status info;
#else
    NewtonRaphsonSolverSpace::Status info;
#endif
    info = m_solver.solve();

#if COLLOCATION_SOLVER == 0
    if (info == PowellHybridSolverSpace::SingularJacobian) {
#else
    if (info == NewtonRaphsonSolverSpace::SingularJacobian) {
#endif
      status = BVPSolverSpace::SingularJacobian;
      iteration++;
      break;
    }

    max_bc_res = m_method.bc_residues.abs().maxCoeff();
    m_method.calculate_residuals();
    Scalar new_max_res = m_method.residuals.maxCoeff();
    if (max_res < new_max_res) {
      status = BVPSolverSpace::NotMakingProgressResiduals;
      max_res = new_max_res;
      ++iteration;
      break;
    }
    max_res = new_max_res;

    auto insert_1 = tools::between(m_method.residuals, tol, m_tol_factor * tol, 0);
    auto insert_2 = tools::greater(m_method.residuals, m_tol_factor * tol, 1);
    m_nodes_added = insert_1.size() + 2 * insert_2.size();
    m_nodes_removed = 0;

    if (m_nodes_added > 0) {
      modify_mesh(tol);
      m_method.calculate();
      m_solver.resize(m_method.rows() * m_method.cols() + m_method.params());
      if (m_method.cols() + m_nodes_added - m_nodes_removed > max_nodes) {
        status = BVPSolverSpace::MaximumNodesExceeded;
#if COLLOCATION_VERBOSITY > 1
        print_iteration_progress(iteration + 1, max_res, max_bc_res, m_method.cols(), m_nodes_added, m_nodes_removed);
#endif
        ++iteration;
        break;
      }
    }
    else if (max_bc_res <= bc_tol) {
      status = BVPSolverSpace::Success;
#if COLLOCATION_VERBOSITY > 1
      print_iteration_progress(iteration + 1, max_res, max_bc_res, m_method.cols(), m_nodes_added, m_nodes_removed);
#endif
      ++iteration;
      break;
    }
#if COLLOCATION_VERBOSITY > 1
      print_iteration_progress(iteration + 1, max_res, max_bc_res, m_method.cols(), m_nodes_added, m_nodes_removed);
#endif
  } // main loop

#if COLLOCATION_VERBOSITY > 0
  print_result(status, iteration, max_res, max_bc_res, m_method.cols());
#endif

  return BVP_Result(m_method, iteration, status, Termination_Messages(status), status == BVPSolverSpace::Success);
}

template <typename Method, bool Singular>
void Generic_Solver<Method, Singular>::modify_mesh(Scalar tol) {
  Array<Scalar, Dynamic, 1> x_new(m_method.cols() + m_nodes_added);
  Array<Scalar, RowsAtCompileTime, Dynamic> y_new(m_method.rows(), m_method.cols() + m_nodes_added);
  Index idx_new = 1;
  x_new(0) = m_method.x(0);
  y_new.col(0) = m_method.y.col(0);
  for (Index idx = 0; idx < m_method.cols() - 1; ++idx) {
    if (m_method.residuals(idx) <= tol) {// Try to remove nodes
#ifdef COLLOCATION_DONT_REMOVE_NODES
      x_new(idx_new) = m_method.x(idx + 1);
      y_new.col(idx_new) = m_method.y.col(idx + 1);
      idx_new += 1;
#else
      // We first determine the number of consecutive intervals for which the
      // residual is below tolerance.
      int intervals = 1;
      for (Index i = idx + 1; i < m_method.cols() - 1; ++i) {
        if (m_method.residuals(i) < tol) {
          intervals += 1;
        }
        else {
          break;
        }
      }
      // If there are more than two consecutive intervals for which the
      // rms residulas is below tolerance we try removing the interior nodes
      // with the smallest residuals and redistribute the mesh.
      if (intervals > 2) {
        int consecutive_intervals = intervals;
        constexpr auto order = m_method.residual_order;
        constexpr Scalar order_inv = static_cast<Scalar>(1) / m_method.residual_order;
        Map<const Array<Scalar, Dynamic, 1>> x_sub(m_method.x.data() + idx, intervals);
        Array<Scalar, Dynamic, 1> res_new = m_method.residuals.segment(idx, intervals);
        Array<Scalar, Dynamic, 1> h_new = m_method.h.segment(idx, intervals);
        Index i_min;
        res_new.segment(1, intervals - 2).minCoeff(&i_min); // Locate the interior interval with the minimum residual.
        i_min += 1;
        Scalar hsum = h_new.template segment<3>(i_min - 1).sum();
        Scalar rim1_pinv = pow(res_new(i_min - 1), order_inv);
        Scalar rip1_pinv = pow(res_new(i_min + 1), order_inv);
        Scalar him1 = h_new(i_min - 1) * (hsum / (h_new(i_min - 1) + h_new(i_min + 1) * (rim1_pinv / rip1_pinv)));
        Scalar hip1 = h_new(i_min + 1) * (hsum / (h_new(i_min + 1) + h_new(i_min - 1) * (rip1_pinv / rim1_pinv)));
        Scalar r_new = res_new(i_min - 1) * res_new(i_min + 1) * pow(hsum / (h_new(i_min + 1) * rim1_pinv + h_new(i_min - 1) * rip1_pinv), order);
        res_new(i_min - 1) = r_new;
        res_new(i_min + 1) = r_new;
        res_new.segment(i_min, intervals - i_min - 1) = res_new.tail(intervals - i_min - 1);
        res_new.conservativeResize(intervals - 1);
        while (r_new < half<Scalar>() * tol) {
          h_new(i_min - 1) = him1;
          h_new(i_min + 1) = hip1;
          h_new.segment(i_min, intervals - i_min - 1) = h_new.tail(intervals - i_min - 1);
          h_new.conservativeResize(intervals - 1);
          intervals -= 1;
          m_nodes_removed += 1;
          if (intervals > 2) {
            res_new.segment(1, intervals - 2).minCoeff(&i_min);
            i_min += 1;
            hsum = h_new.template segment<3>(i_min - 1).sum();
            rim1_pinv = pow(res_new(i_min - 1), order_inv);
            rip1_pinv = pow(res_new(i_min + 1), order_inv);
            him1 = h_new(i_min - 1) * (hsum / (h_new(i_min - 1) + h_new(i_min + 1) * (rim1_pinv / rip1_pinv)));
            hip1 = h_new(i_min + 1) * (hsum / (h_new(i_min + 1) + h_new(i_min - 1) * (rip1_pinv / rim1_pinv)));
            r_new = res_new(i_min - 1) * res_new(i_min + 1) * pow(hsum / (h_new(i_min + 1) * rim1_pinv + h_new(i_min - 1) * rip1_pinv), order);
            res_new(i_min - 1) = r_new;
            res_new(i_min + 1) = r_new;
            res_new.segment(i_min, intervals - i_min - 1) = res_new.tail(intervals - i_min - 1);
            res_new.conservativeResize(intervals - 1);
          }
          else {
            break;
          }
        }
        for (Index i = 0; i < h_new.size(); ++i) {
          x_new(idx_new + i) = x_new(idx_new + i - 1) + h_new(i);
          auto it = std::upper_bound(x_sub.begin(), x_sub.end(), x_new(idx_new + i));
          auto ii = std::distance(x_sub.begin(), it) - 1;
          Scalar w = (x_new(idx_new + i) - m_method.x(idx + ii)) / m_method.h(idx + ii);
          y_new.col(idx_new + i) = m_method.S(idx + ii, w);
        }
        idx_new += h_new.size();
        idx += consecutive_intervals - 1;
      }
      else {// Don't remove nodes
        x_new.segment(idx_new, intervals) = m_method.x.segment(idx + 1, intervals);
        y_new.middleCols(idx_new, intervals) = m_method.y.middleCols(idx + 1, intervals);
        idx_new += intervals;
        idx += intervals - 1;
      }
#endif
    }
#ifndef COLLOCATION_LIMIT_ADD_NODES
    else if (m_method.residuals(idx) >= m_tol_factor * tol) {// Add two nodes
      x_new(idx_new) = m_method.x(idx) + third<Scalar>() * m_method.h(idx);
      x_new(idx_new + 1) = m_method.x(idx) + two_thirds<Scalar>() * m_method.h(idx);
      x_new(idx_new + 2) = m_method.x(idx + 1);
      y_new.col(idx_new) = m_method.y_third(idx);
      y_new.col(idx_new + 1) = m_method.y_two_thirds(idx);
      y_new.col(idx_new + 2) = m_method.y.col(idx + 1);
      idx_new += 3;
    }
#endif
    else {// Add one node
      x_new(idx_new) = m_method.x(idx) + half<Scalar>() * m_method.h(idx);
      x_new(idx_new + 1) = m_method.x(idx + 1);
      y_new.col(idx_new) = m_method.y_half(idx);
      y_new.col(idx_new + 1) = m_method.y.col(idx + 1);
      idx_new += 2;
    }
  }
  // m_method.modify_mesh(x_new.segment(0, m_method.cols() + m_nodes_added - m_nodes_removed), y_new.block(0, 0, m_method.rows(), m_method.cols() + m_nodes_added - m_nodes_removed));
  m_method.modify_mesh(x_new.segment(0, m_method.cols() + m_nodes_added - m_nodes_removed), y_new.leftCols(m_method.cols() + m_nodes_added - m_nodes_removed));
}


} // namespace detail
} // namespace collocation

#endif // BVP_SOLVER_HPP
