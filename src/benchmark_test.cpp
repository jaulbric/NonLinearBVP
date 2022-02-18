#include <benchmark/benchmark.h>

#include <Eigen/Dense>
#include <Eigen/CXX11/Tensor>
#include <boost/math/constants/constants.hpp>
#include <utility>
#include <tuple>
#include <cmath>
#include <iostream>
#include <complex>
#include <iomanip>
#include <limits>

#include "collocation/collocation.hpp"

#include "collocation/collocation_functions.hpp"
#include "collocation/compute_jac_indices.hpp"
#include "collocation/construct_global_jac.hpp"
#include "collocation/estimate_rms_residuals.hpp"
#include "collocation/modify_mesh.hpp"
#include "collocation/solve_newton.hpp"
#include "collocation/tools.hpp"

#include "interpolators/vector_cubic_hermite.hpp"

#include "benchmark_initial_guess.hpp"

using Eigen::array;
using Eigen::Array;
using Eigen::ArrayBase;
using Eigen::Matrix;
using Eigen::Tensor;
using Eigen::TensorBase;
using Eigen::ReadOnlyAccessors;
using Eigen::TensorRef;
using Eigen::Index;
using Eigen::IndexPair;
using std::sqrt;
using boost::math::constants::pi;
using boost::math::constants::half;
using boost::math::constants::sqrt_three_twentyeighth;
using boost::math::constants::thirtytwo_ninetieth;
using boost::math::constants::fortynine_onehundredeightieth;

template <typename Real>
class BVP_fun {
  public:
    BVP_fun(Real m) : m_{m} {};
    ~BVP_fun() = default;

    template <typename Derived1, typename Derived2, typename Derived3>
    Array<Real, 2, Derived2::ColsAtCompileTime>
    operator()(const ArrayBase<Derived1>& x,
               const ArrayBase<Derived2>& y,
               const ArrayBase<Derived3>& p) const {
      Array<Real, 2, Derived2::ColsAtCompileTime> dydx(2, y.cols());
      dydx.row(0) = y.row(1);
      dydx.row(1) = (m_ * m_ - p.value()) * y.row(0);
      return dydx;
    }
  private:
    Real m_;
};

template <typename Real>
class BVP_bc {
  public:
    BVP_bc(Real m) : m_{m} {};
    ~BVP_bc() = default;

    template <typename Derived1, typename Derived2, typename Derived3>
    Array<Real, 3, 1> operator()(const ArrayBase<Derived1>& ya,
                                 const ArrayBase<Derived2>& yb,
                                 const ArrayBase<Derived3>& p) const {
      return Array<Real, 3, 1>({ya(0), yb(0), ya(1) - sqrt(p.value() - m_ * m_)});
    }
  private:
    Real m_;
};

template <typename Real>
class BVP_fun_jac {
  public:
    BVP_fun_jac(Real m) : m_{m} {};
    ~BVP_fun_jac() = default;

    template <typename Derived1, typename Derived2, typename Derived3>
    std::pair<Tensor<Real, 3>, Tensor<Real, 3>>
    operator()(const ArrayBase<Derived1>& x,
               const ArrayBase<Derived2>& y,
               const ArrayBase<Derived3>& p) const {
      Tensor<Real, 3> dfdy(2, 2, x.size());
      Tensor<Real, 3> dfdp(2, 1, x.size());
      for (Index idx = 0; idx < x.size(); ++idx) {
        dfdy(0, 0, idx) = 0;
        dfdy(0, 1, idx) = 1;
        dfdy(1, 0, idx) = m_ * m_ - p.value();
        dfdy(1, 1, idx) = 0;
        dfdp(0, 0, idx) = 0;
        dfdp(1, 0, idx) = -y(0, idx);
      }
      return std::make_pair(dfdy, dfdp);
    }
  private:
    Real m_;
};

template <typename Real>
class BVP_bc_jac {
  public:
    BVP_bc_jac(Real m) : m_{m} {};
    ~BVP_bc_jac() = default;

    template <typename Derived1, typename Derived2, typename Derived3>
    std::tuple<Array<Real, 3, 2>, Array<Real, 3, 2>, Array<Real, 3, 1>>
    operator()(const ArrayBase<Derived1>& x,
               const ArrayBase<Derived2>& y,
               const ArrayBase<Derived3>& p) const {
      Array<Real, 3, 2> dbc_dya = Array<Real, 3, 2>::Zero();
      Array<Real, 3, 2> dbc_dyb = Array<Real, 3, 2>::Zero();
      Array<Real, 3, 1> dbc_dp = Array<Real, 3, 1>::Zero();
      dbc_dya(0, 0) = 1;
      dbc_dya(2, 1) = 1;
      dbc_dyb(1, 0) = 1;
      dbc_dp(2, 0) = - static_cast<Real>(0.5) / sqrt(p.value() - m_ * m_);
      return std::make_tuple(dbc_dya, dbc_dyb, dbc_dp);
    }
  private:
    Real m_;
};

static void BM_solve(benchmark::State& state) {
  using Real = double;

  Real m = 1;
  Array<Real, 1, 1> p_guess;
  p_guess(0,0) = 1.5;

  Array<Real, 10, 1> x = Array<Real, 10,1>::LinSpaced(10, 0, 2 * pi<Real>());

  Array<Real, 2, 10> y_guess = Array<Real, 2, 10>::Zero();
  y_guess(0, 5) = 1;
  y_guess(1, 0) = 1;
  y_guess(1, 9) = -1;

  BVP_fun fun(m);
  BVP_bc bc(m);
  BVP_fun_jac fun_jac(m);
  BVP_bc_jac bc_jac(m);
  for (auto _ : state) {
    auto result = collocation::collocation_algorithm<Real, 2, 1>::solve(fun, bc, fun_jac, bc_jac, x, y_guess, p_guess, 1.0e-9, 1000, 0, 1.0e-9);
  }
}

static void BM_fun(benchmark::State& state) {
  using Real = double;

  Real m = 1;
  Array<Real, 1, 1> p_guess;
  p_guess(0,0) = 1.5;

  Array<Real, 10, 1> x = Array<Real, 10,1>::LinSpaced(10, 0, 2 * pi<Real>());

  Array<Real, 2, 10> y_guess = Array<Real, 2, 10>::Zero();
  y_guess(0, 5) = 1;
  y_guess(1, 0) = 1;
  y_guess(1, 9) = -1;

  BVP_fun fun(m);
  for (auto _ : state) {
    Array<Real, 2, Dynamic> f = fun(x, y_guess, p_guess);
  }
}

static void BM_bc(benchmark::State& state) {
  using Real = double;

  Real m = 1;
  Array<Real, 1, 1> p_guess;
  p_guess(0,0) = 1.5;

  Array<Real, 10, 1> x = Array<Real, 10,1>::LinSpaced(10, 0, 2 * pi<Real>());

  Array<Real, 2, 10> y_guess = Array<Real, 2, 10>::Zero();
  y_guess(0, 5) = 1;
  y_guess(1, 0) = 1;
  y_guess(1, 9) = -1;

  BVP_bc bc(m);
  for (auto _ : state) {
    Array<Real, 3, 1> bc_res = bc(y_guess(all, 0), y_guess(all, last), p_guess);
  }
}

static void BM_fun_jac(benchmark::State& state) {
  using Real = double;

  Real m = 1;
  Array<Real, 1, 1> p_guess;
  p_guess(0,0) = 1.5;

  Array<Real, 10, 1> x = Array<Real, 10,1>::LinSpaced(10, 0, 2 * pi<Real>());

  Array<Real, 2, 10> y_guess = Array<Real, 2, 10>::Zero();
  y_guess(0, 5) = 1;
  y_guess(1, 0) = 1;
  y_guess(1, 9) = -1;

  BVP_fun_jac fun_jac(m);
  for (auto _ : state) {
    auto [ df_dy, df_dp ] = fun_jac(x, y_guess, p_guess);
  }
}

static void BM_bc_jac(benchmark::State& state) {
  using Real = double;

  Real m = 1;
  Array<Real, 1, 1> p_guess;
  p_guess(0,0) = 1.5;

  Array<Real, 10, 1> x = Array<Real, 10,1>::LinSpaced(10, 0, 2 * pi<Real>());

  Array<Real, 2, 10> y_guess = Array<Real, 2, 10>::Zero();
  y_guess(0, 5) = 1;
  y_guess(1, 0) = 1;
  y_guess(1, 9) = -1;

  BVP_bc_jac bc_jac(m);
  for (auto _ : state) {
    auto [ dbc_dya, dbc_dyb, dbc_dp ] = bc_jac(y_guess(all, 0), y_guess(all, last), p_guess);
  }
}

static void BM_compute_jac_indices(benchmark::State& state) {
  using Real = double;

  Array<Real, 10, 1> x = Array<Real, 10, 1>::LinSpaced(10, 0, 2 * pi<Real>());
  for (auto _ : state)
    auto [ i_jac, j_jac ] = collocation::compute_jac_indices(2, x.size(), 1);
}

static void BM_column_midpoints(benchmark::State& state) {
  using Real = double;

  Real m = 1;

  Array<Real, Dynamic, 1> x = x_init;
  Array<Real, 2, Dynamic> y_guess = y_init;
  Array<Real, 1, 1> p_guess = p_init;

  BVP_fun fun(m);
  Array<Real, Dynamic, 1> h = collocation::tools::diff(x);
  auto f = fun(x, y_guess, p_guess);
  for (auto _ : state) {
    Array<Real, 2, Dynamic> y_middle = collocation::column_midpoints(h, y_guess, f);
  }
}

static void BM_collocation_residues(benchmark::State& state) {
  using Real = double;

  Real m = 1;
  Array<Real, Dynamic, 1> x = x_init;
  Array<Real, 2, Dynamic> y_guess = y_init;
  Array<Real, 1, 1> p_guess = p_init;

  BVP_fun fun(m);
  Array<Real, Dynamic, 1> h = collocation::tools::diff(x);
  Array<Real, Dynamic, 1> x_middle = x(seq(0, last-1)) + 0.5 * h;
  auto f = fun(x, y_guess, p_guess);
  auto y_middle = collocation::column_midpoints(h, y_guess, f);
  auto f_middle = fun(x_middle, y_middle, p_guess);
  for (auto _ : state) {
    Array<Real, 2, Dynamic> res = collocation::collocation_residues(h, y_guess, f, f_middle);
  }
}

static void BM_collocation_residues_scaled(benchmark::State& state) {
  using Real = double;

  Real m = 1;
  Array<Real, Dynamic, 1> x = x_init;
  Array<Real, 2, Dynamic> y_guess = y_init;
  Array<Real, 1, 1> p_guess = p_init;

  BVP_fun fun(m);
  Array<Real, Dynamic, 1> h = collocation::tools::diff(x);
  Array<Real, Dynamic, 1> x_middle = x(seq(0, last-1)) + 0.5 * h;
  auto f = fun(x, y_guess, p_guess);
  auto y_middle = collocation::column_midpoints(h, y_guess, f);
  auto f_middle = fun(x_middle, y_middle, p_guess);
  for (auto _ : state) {
    Array<Real, 2, Dynamic> res = collocation::collocation_residues_scaled(h, y_guess, f, f_middle);
  }
}

static void BM_construct_global_jac(benchmark::State& state) {
  using Real = double;

  Real m = 1;
  Array<Real, Dynamic, 1> x = x_init;
  Array<Real, 2, Dynamic> y_guess = y_init;
  Array<Real, 1, 1> p_guess = p_init;

  BVP_fun fun(m);
  BVP_bc bc(m);
  BVP_fun_jac fun_jac(m);
  BVP_bc_jac bc_jac(m);
  Array<Real, Dynamic, 1> h = collocation::tools::diff(x);
  Array<Real, Dynamic, 1> x_middle = x(seq(0, last-1)) + 0.5 * h;
  auto f = fun(x, y_guess, p_guess);
  auto y_middle = collocation::column_midpoints(h, y_guess, f);
  auto f_middle = fun(x_middle, y_middle, p_guess);
  auto [ df_dy, df_dp ] = fun_jac(x, y_guess, p_guess);
  auto [ df_dy_middle, df_dp_middle ] = fun_jac(x_middle, y_middle, p_guess);
  auto [ dbc_dya, dbc_dyb, dbc_dp ] = bc_jac(y_guess(all, 0), y_guess(all, last), p_guess);
  auto [ i_jac, j_jac ] = collocation::compute_jac_indices(2, x.size(), 1);
  for (auto _ : state) {
    auto jac = collocation::collocation_algorithm<Real, 2, 1>::construct_global_jac(i_jac, j_jac, h, df_dy, df_dy_middle, df_dp, df_dp_middle, dbc_dya, dbc_dyb, dbc_dp);
  }
}

static void BM_solve_compressed(benchmark::State& state) {
  using Real = double;

  Real m = 1;
  Array<Real, Dynamic, 1> x = x_init;
  Array<Real, 2, Dynamic> y_guess = y_init;
  Array<Real, 1, 1> p_guess = p_init;

  BVP_fun fun(m);
  BVP_bc bc(m);
  BVP_fun_jac fun_jac(m);
  BVP_bc_jac bc_jac(m);
  Array<Real, Dynamic, 1> h = collocation::tools::diff(x);
  Array<Real, Dynamic, 1> x_middle = x(seq(0, last-1)) + 0.5 * h;
  auto f = fun(x, y_guess, p_guess);
  auto y_middle = collocation::column_midpoints(h, y_guess, f);
  auto f_middle = fun(x_middle, y_middle, p_guess);
  auto [ df_dy, df_dp ] = fun_jac(x, y_guess, p_guess);
  auto [ df_dy_middle, df_dp_middle ] = fun_jac(x_middle, y_middle, p_guess);
  auto [ dbc_dya, dbc_dyb, dbc_dp ] = bc_jac(y_guess(all, 0), y_guess(all, last), p_guess);
  auto [ i_jac, j_jac ] = collocation::compute_jac_indices(2, x.size(), 1);

  Array<Real, 2, Dynamic> col_res = collocation::collocation_residues_scaled(h, y_guess, f, f_middle);

  Array<Real, 3, 1> bc_res = bc(y_guess(all, 0), y_guess(all, last), p_guess);

  Matrix<Real, Dynamic, 1> res(2 * x.size() + 1);
  res << col_res.reshaped().matrix(), bc_res.matrix();

  SparseMatrix<Real> jac(2 * x.size() + 1, 2 * x.size() + 1);
  SparseLU<SparseMatrix<Real>, COLAMDOrdering<int>> solver;
  for (auto _ : state) {
    jac = collocation::collocation_algorithm<Real, 2, 1>::construct_global_jac(i_jac, j_jac, h, df_dy, df_dy_middle, df_dp, df_dp_middle, dbc_dya, dbc_dyb, dbc_dp);
    jac.makeCompressed();
    // jac.prune(static_cast<Real>(0));

    solver.compute(jac);
    Matrix<Real, Dynamic, 1> step = solver.solve(res);
  }
}

static void BM_solve_pruned(benchmark::State& state) {
  using Real = double;

  Real m = 1;
  Array<Real, Dynamic, 1> x = x_init;
  Array<Real, 2, Dynamic> y_guess = y_init;
  Array<Real, 1, 1> p_guess = p_init;

  BVP_fun fun(m);
  BVP_bc bc(m);
  BVP_fun_jac fun_jac(m);
  BVP_bc_jac bc_jac(m);
  Array<Real, Dynamic, 1> h = collocation::tools::diff(x);
  Array<Real, Dynamic, 1> x_middle = x(seq(0, last-1)) + 0.5 * h;
  auto f = fun(x, y_guess, p_guess);
  auto y_middle = collocation::column_midpoints(h, y_guess, f);
  auto f_middle = fun(x_middle, y_middle, p_guess);
  auto [ df_dy, df_dp ] = fun_jac(x, y_guess, p_guess);
  auto [ df_dy_middle, df_dp_middle ] = fun_jac(x_middle, y_middle, p_guess);
  auto [ dbc_dya, dbc_dyb, dbc_dp ] = bc_jac(y_guess(all, 0), y_guess(all, last), p_guess);
  auto [ i_jac, j_jac ] = collocation::compute_jac_indices(2, x.size(), 1);

  Array<Real, 2, Dynamic> col_res = collocation::collocation_residues_scaled(h, y_guess, f, f_middle);

  Array<Real, 3, 1> bc_res = bc(y_guess(all, 0), y_guess(all, last), p_guess);

  Matrix<Real, Dynamic, 1> res(2 * x.size() + 1);
  res << col_res.reshaped().matrix(), bc_res.matrix();

  SparseMatrix<Real> jac(2 * x.size() + 1, 2 * x.size() + 1);
  SparseLU<SparseMatrix<Real>, COLAMDOrdering<int>> solver;
  for (auto _ : state) {
    jac = collocation::collocation_algorithm<Real, 2, 1>::construct_global_jac(i_jac, j_jac, h, df_dy, df_dy_middle, df_dp, df_dp_middle, dbc_dya, dbc_dyb, dbc_dp);
    // jac.makeCompressed();
    jac.prune(static_cast<Real>(0));

    solver.compute(jac);
    Matrix<Real, Dynamic, 1> step = solver.solve(res);
  }
}

static void BM_solve_factorized(benchmark::State& state) {
  using Real = double;

  Real m = 1;
  Array<Real, Dynamic, 1> x = x_init;
  Array<Real, 2, Dynamic> y_guess = y_init;
  Array<Real, 1, 1> p_guess = p_init;

  BVP_fun fun(m);
  BVP_bc bc(m);
  BVP_fun_jac fun_jac(m);
  BVP_bc_jac bc_jac(m);
  Array<Real, Dynamic, 1> h = collocation::tools::diff(x);
  Array<Real, Dynamic, 1> x_middle = x(seq(0, last-1)) + 0.5 * h;
  auto f = fun(x, y_guess, p_guess);
  auto y_middle = collocation::column_midpoints(h, y_guess, f);
  auto f_middle = fun(x_middle, y_middle, p_guess);
  auto [ df_dy, df_dp ] = fun_jac(x, y_guess, p_guess);
  auto [ df_dy_middle, df_dp_middle ] = fun_jac(x_middle, y_middle, p_guess);
  auto [ dbc_dya, dbc_dyb, dbc_dp ] = bc_jac(y_guess(all, 0), y_guess(all, last), p_guess);
  auto [ i_jac, j_jac ] = collocation::compute_jac_indices(2, x.size(), 1);

  Array<Real, 2, Dynamic> col_res = collocation::collocation_residues_scaled(h, y_guess, f, f_middle);

  Array<Real, 3, 1> bc_res = bc(y_guess(all, 0), y_guess(all, last), p_guess);

  Matrix<Real, Dynamic, 1> res(2 * x.size() + 1);
  res << col_res.reshaped().matrix(), bc_res.matrix();

  SparseMatrix<Real> jac(2 * x.size() + 1, 2 * x.size() + 1);
  SparseLU<SparseMatrix<Real>, COLAMDOrdering<int>> solver;
  jac = collocation::collocation_algorithm<Real, 2, 1>::construct_global_jac(i_jac, j_jac, h, df_dy, df_dy_middle, df_dp, df_dp_middle, dbc_dya, dbc_dyb, dbc_dp);
  solver.analyzePattern(jac);
  for (auto _ : state) {
    jac = collocation::collocation_algorithm<Real, 2, 1>::construct_global_jac(i_jac, j_jac, h, df_dy, df_dy_middle, df_dp, df_dp_middle, dbc_dya, dbc_dyb, dbc_dp);
    jac.makeCompressed();
    // jac.prune(static_cast<Real>(0));

    solver.factorize(jac);
    Matrix<Real, Dynamic, 1> step = solver.solve(res);
  }
}

static void BM_matmul_TensorBase(benchmark::State& state) {
  using Real = double;

  Real m = 1;
  Array<Real, Dynamic, 1> x = x_init;
  Array<Real, 2, Dynamic> y_guess = y_init;
  Array<Real, 1, 1> p_guess = p_init;

  BVP_fun fun(m);
  BVP_bc bc(m);
  BVP_fun_jac fun_jac(m);
  BVP_bc_jac bc_jac(m);
  Array<Real, Dynamic, 1> h = collocation::tools::diff(x);
  Array<Real, Dynamic, 1> x_middle = x(seq(0, last-1)) + 0.5 * h;
  auto f = fun(x, y_guess, p_guess);
  auto y_middle = collocation::column_midpoints(h, y_guess, f);
  auto f_middle = fun(x_middle, y_middle, p_guess);
  auto [ df_dy, df_dp ] = fun_jac(x, y_guess, p_guess);
  auto [ df_dy_middle, df_dp_middle ] = fun_jac(x_middle, y_middle, p_guess);
  for (auto _ : state) {
    auto df_dy_shuffled = df_dy.shuffle(array<Index, 3> {2, 0, 1});
    auto df_dy_middle_shuffled = df_dy_middle.shuffle(array<Index, 3>{2, 0, 1});
    auto T0 = collocation::tools::matmul(df_dy_middle_shuffled, df_dy_shuffled.slice(array<Index, 3>({0,0,0}), array<Index, 3>({x.size() - 1, 2, 2})));
    auto T1 = collocation::tools::matmul(df_dy_middle_shuffled, df_dy_shuffled.slice(array<Index, 3>({1,0,0}), array<Index, 3>({x.size() - 1, 2, 2})));
  }
}

static void BM_matmul_Tensor(benchmark::State& state) {
  using Real = double;

  Real m = 1;
  Array<Real, Dynamic, 1> x = x_init;
  Array<Real, 2, Dynamic> y_guess = y_init;
  Array<Real, 1, 1> p_guess = p_init;

  BVP_fun fun(m);
  BVP_bc bc(m);
  BVP_fun_jac fun_jac(m);
  BVP_bc_jac bc_jac(m);
  Array<Real, Dynamic, 1> h = collocation::tools::diff(x);
  Array<Real, Dynamic, 1> x_middle = x(seq(0, last-1)) + 0.5 * h;
  auto f = fun(x, y_guess, p_guess);
  auto y_middle = collocation::column_midpoints(h, y_guess, f);
  auto f_middle = fun(x_middle, y_middle, p_guess);
  auto [ df_dy, df_dp ] = fun_jac(x, y_guess, p_guess);
  auto [ df_dy_middle, df_dp_middle ] = fun_jac(x_middle, y_middle, p_guess);
  for (auto _ : state) {
    auto df_dy_shuffled = df_dy.shuffle(array<Index, 3> {2, 0, 1});
    Tensor<Real, 3> df_dy_middle_shuffled = df_dy_middle.shuffle(array<Index, 3>{2, 0, 1});
    Tensor<Real, 3> df_dy_first = df_dy_shuffled.slice(array<Index, 3>({0,0,0}), array<Index, 3>({x.size() - 1, 2, 2}));
    Tensor<Real, 3> df_dy_last = df_dy_shuffled.slice(array<Index, 3>({1,0,0}), array<Index, 3>({x.size() - 1, 2, 2}));
    Tensor<Real, 3> T0 = collocation::tools::matmul(df_dy_middle_shuffled, df_dy_first);
    Tensor<Real, 3> T1 = collocation::tools::matmul(df_dy_middle_shuffled, df_dy_first);
  }
}

static void BM_matmul_loop(benchmark::State& state) {
  using Real = double;

  Real m = 1;
  Array<Real, Dynamic, 1> x = x_init;
  Array<Real, 2, Dynamic> y_guess = y_init;
  Array<Real, 1, 1> p_guess = p_init;

  BVP_fun fun(m);
  BVP_bc bc(m);
  BVP_fun_jac fun_jac(m);
  BVP_bc_jac bc_jac(m);
  Array<Real, Dynamic, 1> h = collocation::tools::diff(x);
  Array<Real, Dynamic, 1> x_middle = x(seq(0, last-1)) + 0.5 * h;
  auto f = fun(x, y_guess, p_guess);
  auto y_middle = collocation::column_midpoints(h, y_guess, f);
  auto f_middle = fun(x_middle, y_middle, p_guess);
  auto [ df_dy, df_dp ] = fun_jac(x, y_guess, p_guess);
  auto [ df_dy_middle, df_dp_middle ] = fun_jac(x_middle, y_middle, p_guess);
  for (auto _ : state) {
    Tensor<Real, 3> T0(x.size() - 1, 2, 2);
    Tensor<Real, 3> T1(x.size() - 1, 2, 2);
    for (Index idx = 0; idx < x.size() - 1; ++idx) {
      auto a_first = df_dy.chip(idx, 2);
      auto a_last = df_dy.chip(idx + 1, 2);
      auto b = df_dy_middle.chip(idx, 2);
      T0.chip(idx, 0) = b.contract(a_first, array<IndexPair<Index>, 1>({IndexPair<Index>(1, 0)}));
      T1.chip(idx, 0) = b.contract(a_last, array<IndexPair<Index>, 1>({IndexPair<Index>(1, 0)}));
    }
  }
}

static void BM_matmul_loop2(benchmark::State& state) {
  using Real = double;

  Real m = 1;
  Array<Real, Dynamic, 1> x = x_init;
  Array<Real, 2, Dynamic> y_guess = y_init;
  Array<Real, 1, 1> p_guess = p_init;

  BVP_fun fun(m);
  BVP_bc bc(m);
  BVP_fun_jac fun_jac(m);
  BVP_bc_jac bc_jac(m);
  Array<Real, Dynamic, 1> h = collocation::tools::diff(x);
  Array<Real, Dynamic, 1> x_middle = x(seq(0, last-1)) + 0.5 * h;
  auto f = fun(x, y_guess, p_guess);
  auto y_middle = collocation::column_midpoints(h, y_guess, f);
  auto f_middle = fun(x_middle, y_middle, p_guess);
  auto [ df_dy, df_dp ] = fun_jac(x, y_guess, p_guess);
  auto [ df_dy_middle, df_dp_middle ] = fun_jac(x_middle, y_middle, p_guess);
  for (auto _ : state) {
    Tensor<Real, 3> T0(2, 2, x.size() - 1);
    Tensor<Real, 3> T1(2, 2, x.size() - 1);
    for (Index idx = 0; idx < x.size() - 1; ++idx) {
      // auto a_first = df_dy.chip(idx, 2);
      // auto a_last = df_dy.chip(idx + 1, 2);
      // auto b = df_dy_middle.chip(idx, 2);
      // T0.chip(idx, 0) = b.contract(a_first, array<IndexPair<Index>, 1>({IndexPair<Index>(1, 0)}));
      // T1.chip(idx, 0) = b.contract(a_last, array<IndexPair<Index>, 1>({IndexPair<Index>(1, 0)}));
      Map<const Matrix<Real, 2, 2>> m1(df_dy.data() + idx * 4);
      Map<const Matrix<Real, 2, 2>> m2(df_dy.data() + (idx + 1) * 4);
      Map<const Matrix<Real, 2, 2>> m3(df_dy_middle.data() + idx * 4);
      Map<Matrix<Real, 2, 2>> m4(T0.data() + idx * 4);
      Map<Matrix<Real, 2, 2>> m5(T1.data() + idx * 4);
      m4.noalias() = m3 * m1;
      m5.noalias() = m3 * m2;
    }
  }
}

static void BM_estimate_rms_residuals(benchmark::State& state) {
  using Real = double;

  Real m = 1;
  Array<Real, Dynamic, 1> x = x_init;
  Array<Real, 2, Dynamic> y_guess = y_init;
  Array<Real, 1, 1> p_guess = p_init;

  BVP_fun fun(m);
  Array<Real, Dynamic, 1> h = collocation::tools::diff(x);
  auto f = fun(x, y_guess, p_guess);
  for (auto _ : state) {
    Array<Real, Dynamic, 1> rms_res = collocation::collocation_algorithm<Real, 2, 1>::estimate_rms_residuals(fun, x, h, y_guess, f, p_guess);
  }
}

static void BM_solve_newton(benchmark::State& state) {
  using Real = double;

  Real m = 1;
  Array<Real, Dynamic, 1> x = x_init;
  Array<Real, 2, Dynamic> y_guess = y_init;
  Array<Real, 1, 1> p_guess = p_init;

  BVP_fun fun(m);
  BVP_bc bc(m);
  BVP_fun_jac fun_jac(m);
  BVP_bc_jac bc_jac(m);
  Array<Real, Dynamic, 1> h = collocation::tools::diff(x);
  Array<Real, 2, Dynamic> y(2, 10);
  Array<Real, 1, 1> p;
  bool singular;
  for (auto _ : state) {
    std::tie(y, p, singular) = collocation::collocation_algorithm<Real, 2, 1>::solve_newton(fun, bc, fun_jac, bc_jac, x, h, y_guess, p_guess, 1.0e-9, 1.0e-9);
  }
}

static void BM_matrix_matrix_multiplication(benchmark::State& state) {
  Eigen::Matrix<double, Dynamic, Dynamic> m1 = Eigen::Matrix<double, Dynamic, Dynamic>::Random(100, 100);
  Eigen::Matrix<double, Dynamic, Dynamic> m2 = Eigen::Matrix<double, Dynamic, Dynamic>::Random(100, 100);
  Eigen::Matrix<double, Dynamic, Dynamic> m3(100, 100);
  for (auto _ : state) {
    m3.noalias() = m1 * m2;
  }
}

static void BM_tensor_matrix_multiplication(benchmark::State& state) {
  Tensor<double, 2> t1(100, 100);
  Tensor<double, 2> t2(100, 100);
  Tensor<double, 2> t3(100, 100);
  t1.setRandom();
  t2.setRandom();
  for (auto _ : state) {
    // Map<Matrix<double, Dynamic, Dynamic>> m1(t1.data(), 100, 100);
    // Map<Matrix<double, Dynamic, Dynamic>> m2(t2.data(), 100, 100);
    // Map<Matrix<double, Dynamic, Dynamic>> m3(t3.data(), 100, 100);
    // m3.noalias() = m1 * m2;
    t3 = t1.contract(t2, array<IndexPair<Index>, 1>({IndexPair<Index>(1, 0)}));
  }
}

BENCHMARK(BM_solve)->Name("solve")->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fun)->Name("fun")->Unit(benchmark::kNanosecond);
BENCHMARK(BM_bc)->Name("bc")->Unit(benchmark::kNanosecond);
BENCHMARK(BM_fun_jac)->Name("fun_jac")->Unit(benchmark::kNanosecond);
BENCHMARK(BM_bc_jac)->Name("bc_jac")->Unit(benchmark::kNanosecond);
BENCHMARK(BM_compute_jac_indices)->Name("compute_jac_indices")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_column_midpoints)->Name("column_midpoints")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_collocation_residues)->Name("collocation_residues")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_collocation_residues_scaled)->Name("collocation_residues_scaled")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_construct_global_jac)->Name("construct_global_jac")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_solve_compressed)->Name("Solve Jacobian compressed")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_solve_pruned)->Name("Solve Jacobian pruned")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_solve_factorized)->Name("Solve Jacobian factorized")->Unit(benchmark::kMicrosecond);
// BENCHMARK(BM_matmul_TensorBase)->Name("matmul TensorBase")->Unit(benchmark::kMicrosecond);
// BENCHMARK(BM_matmul_Tensor)->Name("matmul Tensor")->Unit(benchmark::kMicrosecond);
// BENCHMARK(BM_matmul_loop)->Name("matmul loop")->Unit(benchmark::kMicrosecond);
// BENCHMARK(BM_matmul_loop2)->Name("matmul loop2")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_estimate_rms_residuals)->Name("estimate_rms_residuals")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_solve_newton)->Name("solve_newton")->Unit(benchmark::kMicrosecond);
// BENCHMARK(BM_matrix_matrix_multiplication)->Name("matrix-matrix multiplication");
// BENCHMARK(BM_tensor_matrix_multiplication)->Name("tensor-matrix multiplication");

BENCHMARK_MAIN();
