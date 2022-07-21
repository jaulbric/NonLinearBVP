#ifndef COLLOCATION_DETAIL_GENERIC_SOLVER_HPP
#define COLLOCATION_DETAIL_GENERIC_SOLVER_HPP

#ifdef COLLOCATION_INTERNAL_BREAKPOINTS
#include <fstream>
#endif

#ifdef COLLOCATION_MODIFY_MESH_LOG
#include <fstream>
#endif

#ifndef COLLOCATION_AUTODIFF
#define COLLOCATION_AUTODIFF 0
#endif

#ifndef COLLOCATION_VERBOSITY
#define COLLOCATION_VERBOSITY 0
#endif

#ifndef COLLOCATION_SOLVER
#define COLLOCATION_SOLVER 0
#endif

#ifndef COLLOCATION_FAIL_TRIGGER
#define COLLOCATION_FAIL_TRIGGER 2
#endif

#include <boost/math/tools/precision.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <string>

#include "NonLinearBVP/collocation/BVP_Result.hpp"
#if COLLOCATION_SOLVER == 0
#include "NonLinearBVP/NonLinearOptimization/PowellHybrid.hpp"
#else
#include "NonLinearBVP/NonLinearOptimization/NewtonRaphson.hpp"
#endif
#include "NonLinearBVP/constants/constants.hpp"
#include "NonLinearBVP/collocation/detail/tools.hpp"

#if COLLOCATION_VERBOSITY > 0
#include "NonLinearBVP/collocation/detail/print_functions.hpp"
#endif

namespace nonlinearbvp {

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

std::string Termination_Messages(const BVPSolverSpace::Status status) {
  std::string message;
  switch (status) {
    case BVPSolverSpace::Success : {
      message = "The algorithm converged to the desired accuracy";
      break;
    }
    case BVPSolverSpace::MaximumNodesExceeded: {
      message = "The maximum number of mesh nodes was exceeded.";
      break;
    }
    case BVPSolverSpace::SingularJacobian: {
      message = "A singular Jacobian was encountered when solving the collocation system.";
      break;
    }
    case BVPSolverSpace::TolTooSmall: {
      message = "The solver was unable to satisfy boundary conditions on final iteration.";
      break;
    }
    case BVPSolverSpace::NotMakingProgressResiduals: {
      message = "Maximum residual did not decrease after two iterations.";
      break;
    }
    default: {
      message = "Unknown error.";
    }
  }
  return message;
}

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
        const MatrixBase<Derived4>& B) : m_method{fun, bc, fun_jac, bc_jac, x_input, y_input, p_input}, m_solver{m_method, B}, m_nodes_added{0}, m_nodes_removed{0}, nslow{0} {
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
        const ArrayBase<Derived3>& p_input) : m_method{fun, bc, fun_jac, bc_jac, x_input, y_input, p_input}, m_solver{m_method}, m_nodes_added{0}, m_nodes_removed{0}, nslow{0} {
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
        const MatrixBase<Derived3>& B) : m_method{fun, bc, fun_jac, bc_jac, x_input, y_input}, m_solver{m_method, B}, m_nodes_added{0}, m_nodes_removed{0}, nslow{0} {
      static_assert(Singular == true, "Singular == false but a singular term has been supplied");
      static_assert(ParamsAtCompileTime == 0, "Declared number of unknown parameters does not match supplied number of parameters");
    }

    template <typename Derived1, typename Derived2>
    Generic_Solver(Functor& fun,
        BCFunctor& bc,
        FunctorJacobian& fun_jac,
        BCFunctorJacobian& bc_jac,
        const ArrayBase<Derived1>& x_input,
        const ArrayBase<Derived2>& y_input) : m_method{fun, bc, fun_jac, bc_jac, x_input, y_input}, m_solver{m_method}, m_nodes_added{0}, m_nodes_removed{0}, nslow{0} {
      static_assert(Singular == false, "Singular == true but no singular term has been supplied.");
      static_assert(ParamsAtCompileTime == 0, "Declared number of unknown parameters does not match supplied number of parameters");
    }

    BVP_Result<Method> solve(const Scalar a_tol, Scalar r_tol, const Scalar bc_tol, const int max_nodes);

  private:
    Method m_method;
#if COLLOCATION_SOLVER == 0
    PowellHybrid<Method, Singular> m_solver;
#else
    NewtonRaphson<Method, Singular> m_solver;
#endif
    // static constexpr Scalar EPS = boost::math::tools::epsilon<Scalar>();
    static constexpr Scalar EPS = std::numeric_limits<Scalar>::epsilon();
    static constexpr int m_max_iterations = 10; // Max number of iterations of the main loop
    int m_nodes_added;
    int m_nodes_removed;
    static constexpr Scalar m_tol_factor = 2 * tools::three_raise_x(Scalar(Method::residual_order));
    int nslow;
    Array<int, Dynamic, 1> intervals;

    void modify_mesh(const Scalar a_tol, const Scalar r_tol);
};

template <typename Method, bool Singular>
BVP_Result<Method> Generic_Solver<Method, Singular>::solve(const Scalar a_tol, Scalar r_tol, const Scalar bc_tol, const int max_nodes) {

#ifdef COLLOCATION_INTERNAL_BREAKPOINTS // Save the mesh nodes and solutions
  std::ofstream file("output.csv");
  std::cout << "I'm going to save the output at each step to output.csv." << std::endl;
  const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
  if (file.is_open()) {
    file << m_method.x.transpose().format(CSVFormat) << std::endl;
    file << m_method.y.format(CSVFormat) << std::endl;
  }
#endif

  if (r_tol < 100 * EPS) {
    // const_cast<Scalar>(tol) = 100 * EPS;
    r_tol = 100 * EPS;
  }
#if COLLOCATION_VERBOSITY > 1
  print_iteration_header();
#endif
  m_solver.parameters.a_tol = 10 * std::numeric_limits<Scalar>::min();
  m_solver.parameters.r_tol = 100 * std::numeric_limits<Scalar>::epsilon();
  // m_solver.parameters.a_tol = a_tol;
  // m_solver.parameters.r_tol = r_tol;
  m_solver.parameters.bc_tol = bc_tol;
  m_solver.parameters.max_njev = 4;

  BVPSolverSpace::Status status = BVPSolverSpace::TolTooSmall;
  Scalar max_bc_res = std::numeric_limits<Scalar>::max();
  Scalar max_res = std::numeric_limits<Scalar>::max();
  int iteration;
  nslow = 0;
  Index i_res_max;
  Index j_res_max;
  for (iteration = 0; iteration < m_max_iterations; ++iteration) {
    intervals = Array<int, Dynamic, 1>::Zero(m_method.cols() - 1);
#if COLLOCATION_SOLVER == 0
    PowellHybridSolverSpace::Status info;
#else
    NewtonRaphsonSolverSpace::Status info;
#endif
    info = m_solver.solve();
    // std::cout << info << std::endl;
#ifdef COLLOCATION_INTERNAL_BREAKPOINTS
    if (file.is_open()) {
      file << m_method.x.transpose().format(CSVFormat) << std::endl;
      file << m_method.y.format(CSVFormat) << std::endl;
    }
#endif

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
    Scalar new_max_res = m_method.residuals.maxCoeff(&i_res_max, &j_res_max);
    // std::cout << "Maximum residual " << m_method.residuals.col(j_res_max).transpose() << " located at x = " << m_method.x(j_res_max + 1) << std::endl;
    if (max_res < new_max_res) {
      ++nslow;
    }
    else {
      nslow = 0;
    }
    max_res = new_max_res;
    if (nslow > COLLOCATION_FAIL_TRIGGER) {
      status = BVPSolverSpace::NotMakingProgressResiduals;
      ++iteration;
      break;
    }

    Index offset = 0;
    m_nodes_added = 0;
    for (Index idx = 0; idx < m_method.cols() - 1; ++idx) {
      if ((m_method.residuals.col(idx) <= a_tol + r_tol * m_method.residual_norms.col(idx)).all()) {
        intervals(offset) += 1;
      }
#ifdef COLLOCATION_LIMIT_ADD_NODES
      else {
        intervals(idx) = -1;
        m_nodes_added += 1;
        offset = idx + 1;
      }
#else
      else if ((m_method.residuals.col(idx) < m_tol_factor * (a_tol + r_tol * m_method.residual_norms.col(idx))).all()) {
        intervals(idx) = -1;
        m_nodes_added += 1;
        offset = idx + 1;
      }
      else {
        intervals(idx) = -2;
        m_nodes_added += 2;
        offset = idx + 1;
      }
#endif
    }
    m_nodes_removed = 0;

    if (m_nodes_added > 0) {
      modify_mesh(a_tol, r_tol);
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
  print_result(status, iteration, max_res / m_method.residual_norms(i_res_max, j_res_max), max_bc_res, m_method.cols());
#endif

#ifdef COLLOCATION_INTERNAL_BREAKPOINTS
  file.close();
#endif

  return BVP_Result(m_method, iteration, status, Termination_Messages(status), status == BVPSolverSpace::Success);
}

template <typename Method, bool Singular>
void Generic_Solver<Method, Singular>::modify_mesh(const Scalar a_tol, const Scalar r_tol) {
#ifdef COLLOCATION_MODIFY_MESH_LOG
  std::ofstream mesh_log("mesh_log.txt", std::ios_base::app);
  if (mesh_log.is_open()) {
    mesh_log << "Initial number of nodes: " << m_method.cols() << std::endl;
    mesh_log << "Number of nodes to be added: " << m_nodes_added << std::endl;
  }
#endif
  Array<Scalar, Dynamic, 1> x_new(m_method.cols() + m_nodes_added);
  Array<Scalar, RowsAtCompileTime, Dynamic> y_new(m_method.rows(), m_method.cols() + m_nodes_added);
  Index idx_new = 1;
  x_new(0) = m_method.x(0);
  y_new.col(0) = m_method.y.col(0);
  for (Index idx = 0; idx < m_method.cols() - 1; ++idx) {
    if (intervals(idx) >= 0) {// Try to remove nodes
#ifdef COLLOCATION_DONT_REMOVE_NODES
      x_new.segment(idx_new, intervals(idx)) = m_method.x.segment(idx + 1, intervals(idx));
      y_new.middleCols(idx_new, intervals(idx)) = m_method.y.middleCols(idx + 1, intervals(idx));
      idx_new += intervals(idx);
      idx += intervals(idx) - 1
#else
      if (nslow > 0) { // Don't try to remove nodes if the max residual increased from the previous step.
        x_new.segment(idx_new, intervals(idx)) = m_method.x.segment(idx + 1, intervals(idx));
        y_new.middleCols(idx_new, intervals(idx)) = m_method.y.middleCols(idx + 1, intervals(idx));
        idx_new += intervals(idx);
        idx += intervals(idx) - 1;
        continue;
      }

      int consecutive_intervals = intervals(idx);
      // If there are more than two consecutive intervals for which the
      // residulas is below tolerance we try removing the interior nodes
      // with the smallest residuals and redistribute the mesh.
      if (consecutive_intervals > 2) {
        constexpr auto order = m_method.residual_order;
        constexpr Scalar order_inv = static_cast<Scalar>(1) / m_method.residual_order;
        Map<const Array<Scalar, Dynamic, 1>> x_sub(m_method.x.data() + idx, intervals(idx) + 1);
        Array<Scalar, Dynamic, 1> res_new = (m_method.residuals.middleCols(idx, intervals(idx)) / (a_tol + r_tol * m_method.residual_norms.middleCols(idx, intervals(idx)))).eval().colwise().maxCoeff();
        Array<Scalar, Dynamic, 1> h_new = m_method.h.segment(idx, intervals(idx));
        Index i_min;
        Scalar r_min;
        Array<bool, Dynamic, 1> can_be_merged = Array<bool, Dynamic, 1>::Constant(intervals(idx), true);
        Scalar hsum, him1, hip1, r_new, rim1_pinv, rip1_pinv;
        int state = 3;
        while (true) {
          /* Find the first interval that can be merged */
          i_min = - 1;
          for (Index i = 1; i < intervals(idx) - 1; ++i) {
            switch (state) {
              case 0: { // we do not know if any of the intervals can be merged
                if (!can_be_merged(i + 1)) {
                  i += 2;
                  state = 0;
                  continue;
                }
                if (!can_be_merged(i)) {
                  i += 1;
                  state = 1;
                  continue;
                }
                if (!can_be_merged(i - 1)) {
                  state = 2;
                  continue;
                }
                break;
              }
              case 1: { // we know that the first interval can be merged
                if (!can_be_merged(i + 1)) {
                  i += 2;
                  state = 0;
                  continue;
                }
                if (!can_be_merged(i)) {
                  i += 1;
                  state = 1;
                  continue;
                }
                break;
              }
              case 2: { // we know that the first and middle interval can be merged
                if (!can_be_merged(i + 1)) {
                  i += 2;
                  state = 0;
                  continue;
                }
                break;
              }
              case 3: {
                break;
              }
              default: {
                break;
              }
            }
            i_min = i;
            r_min = res_new(i);
            break;
          }

          if (i_min == -1) { // We end the loop when no intervals can be merged
            break;
          }

          /* Find the interval with the smallest residual in the subset of
          consecutive intervals for which the residual is below tolerance. We
          will remove this interval first */
          state = 2;
          for (Index i = i_min + 1; i < intervals(idx) - 1; ++i) {
            switch (state) {
              case 0: {
                if (!can_be_merged(i + 1)) {
                  i += 2;
                  state = 0;
                  continue;
                }
                if (!can_be_merged(i)) {
                  i += 1;
                  state = 1;
                  continue;
                }
                if (!can_be_merged(i - 1)) {
                  state = 2;
                  continue;
                }
                break;
              }
              case 1: {
                if (!can_be_merged(i + 1)) {
                  i += 2;
                  state = 0;
                  continue;
                }
                if (!can_be_merged(i)) {
                  i += 1;
                  state = 1;
                  continue;
                }
                break;
              }
              case 2: {
                if (!can_be_merged(i + 1)) {
                  i += 2;
                  state = 0;
                  continue;
                }
                break;
              }
            }
            if (res_new(i) < r_min) {
              i_min = i;
              r_min = res_new(i);
            }
            state = 2;
          }
          state = 0;

          hsum = h_new.template segment<3>(i_min - 1).sum();
          if ((res_new(i_min - 1) <= EPS) || (res_new(i_min + 1) <= EPS)) { // can't redistribute mesh reliably if the scaled residuals are very small. In this case just bisect the middle interval.
            him1 = h_new(i_min - 1) + half<Scalar>() * h_new(i_min);
            hip1 = h_new(i_min + 1) + half<Scalar>() * h_new(i_min);
            Scalar rim1 = res_new(i_min - 1) * pow(him1 / h_new(i_min - 1), order);
            Scalar rip1 = res_new(i_min + 1) * pow(hip1 / h_new(i_min + 1), order);
            r_new = std::max(rim1, rip1);
          }
          else {
            rim1_pinv = pow(res_new(i_min - 1), order_inv);
            rip1_pinv = pow(res_new(i_min + 1), order_inv);
            Scalar denom = rim1_pinv * h_new(i_min + 1) + rip1_pinv * h_new(i_min - 1);
            him1 = rip1_pinv * hsum * h_new(i_min - 1) / denom;
            hip1 = rim1_pinv * hsum * h_new(i_min + 1) / denom;
            r_new = res_new(i_min - 1) * res_new(i_min + 1) * pow(hsum / denom, order);
          }

          /* Check if the estimate of the new residuals is sufficiently small */
          if (r_new < tenth<Scalar>()) { // merge the 3 intervals into 2
#ifdef COLLOCATION_MODIFY_MESH_LOG
            mesh_log << "REMOVE: removing interval " << i_min << " from sequence [x(" << idx << "), x(" << idx + consecutive_intervals << ")]. Total intervals = " << intervals(idx) << std::endl;
            mesh_log << "\tInterval lengths before removal: {h(" << i_min - 1 << "), h(" << i_min << "), h(" << i_min + 1 << ")} = {" << h_new(i_min - 1) << ", " << h_new(i_min) << ", " << h_new(i_min + 1) << "}" << std::endl;
            mesh_log << "\tResiduals before removal: {r(" << i_min - 1 << "), r(" << i_min << "), r(" << i_min + 1 << ")} = {" << res_new(i_min - 1) << ", " << res_new(i_min) << ", " << res_new(i_min + 1) << "}" << std::endl;
            mesh_log << "\tInterval lengths after removal: {h(" << i_min - 1 << "), h(" << i_min << ")} = {" << him1 << ", " << hip1 << "}" << std::endl;
            mesh_log << "\tResiduals after removal: {r(" << i_min - 1 << "), r(" << i_min << ")} = {" << r_new << ", " << r_new << "}" << std::endl;
#endif
            res_new(i_min - 1) = r_new;
            res_new(i_min + 1) = r_new;
            h_new(i_min - 1) = him1;
            h_new(i_min + 1) = hip1;
            can_be_merged(i_min - 1) = false;
            can_be_merged(i_min + 1) = false;
            res_new.segment(i_min, intervals(idx) - i_min - 1) = res_new.tail(intervals(idx) - i_min - 1);
            h_new.segment(i_min, intervals(idx) - i_min - 1) = h_new.tail(intervals(idx) - i_min - 1);
            can_be_merged.segment(i_min, intervals(idx) - i_min - 1) = can_be_merged.tail(intervals(idx) - i_min - 1);
            res_new.conservativeResize(intervals(idx) - 1);
            h_new.conservativeResize(intervals(idx) - 1);
            can_be_merged.conservativeResize(intervals(idx) - 1);
            intervals(idx) -= 1;
            ++m_nodes_removed;
          }
          else { // Don't merge, and don't try to merge this interval again.
            can_be_merged(i_min) = false;
          }
        }

        for (Index i = 0; i < intervals(idx) - 1; ++i) {
          x_new(idx_new + i) = x_new(idx_new + i - 1) + h_new(i);
          auto it = std::upper_bound(x_sub.begin(), x_sub.end(), x_new(idx_new + i));
          auto ii = std::distance(x_sub.begin(), it) - 1;
          Scalar w = (x_new(idx_new + i) - m_method.x(idx + ii)) / m_method.h(idx + ii);
          y_new.col(idx_new + i) = m_method.S(idx + ii, w);
        }
        x_new(idx_new + intervals(idx) - 1) = m_method.x(idx + consecutive_intervals);
        y_new.col(idx_new + intervals(idx) - 1) = m_method.y.col(idx + consecutive_intervals);
#ifdef COLLOCATION_MODIFY_MESH_LOG
        mesh_log << "\tOld sequence [x(" << idx << "), x(" << idx + consecutive_intervals << ")]:" << std::endl;
        mesh_log << x_sub.transpose() << std::endl;
        mesh_log << "\tNew sequence: [x_new(" << idx_new - 1 << "), x_new(" << idx_new + intervals(idx) - 1 << ")]:" << std::endl;
        mesh_log << x_new.segment(idx_new - 1, intervals(idx) + 1).transpose() << std::endl;
#endif
        idx_new += intervals(idx);
        idx += consecutive_intervals - 1;
      }
      else {// Don't remove nodes
        x_new.segment(idx_new, intervals(idx)) = m_method.x.segment(idx + 1, intervals(idx));
        y_new.middleCols(idx_new, intervals(idx)) = m_method.y.middleCols(idx + 1, intervals(idx));
        idx_new += intervals(idx);
        idx += intervals(idx) - 1;
      }
#endif
    }
#ifndef COLLOCATION_LIMIT_ADD_NODES
    else if (intervals(idx) == -2) {// Add two nodes
      x_new(idx_new) = m_method.x(idx) + third<Scalar>() * m_method.h(idx);
      x_new(idx_new + 1) = m_method.x(idx) + two_thirds<Scalar>() * m_method.h(idx);
      x_new(idx_new + 2) = m_method.x(idx + 1);
      y_new.col(idx_new) = m_method.y_third(idx);
      y_new.col(idx_new + 1) = m_method.y_two_thirds(idx);
      y_new.col(idx_new + 2) = m_method.y.col(idx + 1);
#ifdef COLLOCATION_MODIFY_MESH_LOG
      mesh_log << "ADD: adding two nodes to the interval [x(" << idx << "), x(" << idx + 1 << ")] = [" << m_method.x(idx) << ", " << m_method.x(idx + 1) << "]." << std::endl;
      mesh_log << "\tNew interval {x_new(" << idx_new - 1 << "), x_new(" << idx_new << "), x_new(" << idx_new + 1 << "), x(" << idx_new + 2 << ")} = {" << x_new(idx_new - 1) << ", " << x_new(idx_new) << ", " << x_new(idx_new + 1) << ", " << x_new(idx_new + 2) << "}" << std::endl;
#endif
      idx_new += 3;
    }
#endif
    else if (intervals(idx) == -1){// Add one node
      x_new(idx_new) = m_method.x(idx) + half<Scalar>() * m_method.h(idx);
      x_new(idx_new + 1) = m_method.x(idx + 1);
      y_new.col(idx_new) = m_method.y_half(idx);
      y_new.col(idx_new + 1) = m_method.y.col(idx + 1);
#ifdef COLLOCATION_MODIFY_MESH_LOG
      mesh_log << "ADD: adding one node to the interval [x(" << idx << "), x(" << idx + 1 << ")] = [" << m_method.x(idx) << ", " << m_method.x(idx + 1) << "]." << std::endl;
      mesh_log << "\tNew interval {x_new(" << idx_new - 1 << "), x_new(" << idx_new << "), x_new(" << idx_new + 1 << ")} =  {" << x_new(idx_new - 1) << ", " << x_new(idx_new) << ", " << x_new(idx_new + 1) << "}" << std::endl;
#endif
      idx_new += 2;
    }
    else {
      std::cout << "This shouldn't have happened. intervals(" << idx << ") = " << intervals(idx) << std::endl;
    }
  }
#ifdef COLLOCATION_MODIFY_MESH_LOG
  mesh_log << "Done." << std::endl << std::endl;
  mesh_log.close();
#endif
  // std::cout << x_new(m_method.cols() + m_nodes_added - m_nodes_removed - 1) << std::endl;
  m_method.modify_mesh(x_new.segment(0, m_method.cols() + m_nodes_added - m_nodes_removed), y_new.leftCols(m_method.cols() + m_nodes_added - m_nodes_removed));
}


} // namespace detail
} // namespace collocation
} // namespace nonlinearbvp

#endif // BVP_SOLVER_HPP
