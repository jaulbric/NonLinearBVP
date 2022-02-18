#ifndef COLLOCATION_ESTIMATE_BC_JAC_HPP
#define COLLOCATION_ESTIMATE_BC_JAC_HPP

namespace collocation { namespace detail {

template <typename BC, typename Real, Index n, Index k>
class estimate_bc_jac {
  public:
    estimate_bc_jac(BC& bc) : bc_{bc} {};
    ~estimate_bc_jac() = default;

    template <typename Derived1, typename Derived2, typename Derived3, typename Derived4>
    std::tuple<Array<Real, n + k, n>, Array<Real, n + k, n>, Array<Real, n + k, k>> operator()(const ArrayBase<Derived1>& ya,
                                                                                               const ArrayBase<Derived2>& yb,
                                                                                               const ArrayBase<Derived3>& p,
                                                                                               const ArrayBase<Derived4>& bc0) const {
      Array<Real, n + k, n> dbc_dya;
      Array<Real, n + k, n> dbc_dyb;
      Array<Real, n + k, k> dbc_dp;

      Array<Real, n, 1> y_new;
      Array<Real, k, 1> p_new;

      Array<Real, n + k, 1> bc_new;
      Array<Real, n, 1> h;
      Array<Real, k, 1> hp;

      h = sqrt_EPS * (1 + ya.abs());
      for (Index j = 0; j < n; ++j) {
        y_new = ya;
        y_new(j) += h(j);
        bc_new = bc_(y_new, yb, p);
        Real hj = y_new(j) - ya(j);
        for (Index i = 0; i < n + k; ++i) {
         dbc_dya(i, j) = (bc_new(i) - bc0(i)) / hj;
        }
      }

      h = sqrt_EPS * (1 + yb.abs());
      for (Index j = 0; j < n; ++j) {
        y_new = yb;
        y_new(j) += h(j);
        Real hj = y_new(j) - yb(j);
        bc_new = bc_(ya, y_new, p);
        for (int i = 0; i < n + k; i++) {
          dbc_dyb(i, j) = (bc_new(i) - bc0(i)) / hj;
        }
      }

      hp = sqrt_EPS * (1 + p.abs());
      for (Index j = 0; j < k; ++j) {
        p_new = p;
        p_new(j) += hp(j);
        Real hj = p_new(j) - p(j);
        bc_new = bc_(ya, yb, p_new);
        for (Index i = 0; i < n + k; ++i) {
          dbc_dp(i, j) = (bc_new(i) - bc0(i)) / hj;
        }
      }

      return std::make_tuple(dbc_dya, dbc_dyb, dbc_dp);
    };

    template <typename Derived1, typename Derived2, typename Derived3>
    inline std::tuple<Array<Real, n + k, n>, Array<Real, n + k, n>, Array<Real, n + k, k>> operator()(const ArrayBase<Derived1>& x,
                                                                                                      const ArrayBase<Derived2>& y,
                                                                                                      const ArrayBase<Derived3>& p) const {
      auto bc0 = bc_(x, y, p);
      return (*this)(x, y, p, bc0);
    };
  private:
    BC& bc_;
    static constexpr Real sqrt_EPS = sqrt(boost::math::tools::epsilon<Real>());
};

// Same thing but with no unknown parameters
template <typename BC, typename Real, Index n>
class estimate_bc_jac<BC, Real, n, 0> {
  public:
    estimate_bc_jac(BC& bc) : bc_{bc} {};
    ~estimate_bc_jac() = default;

    template <typename Derived1, typename Derived2, typename Derived3>
    std::tuple<Array<Real, n, n>, Array<Real, n, n>> operator()(const ArrayBase<Derived1>& ya,
                                                                const ArrayBase<Derived2>& yb,
                                                                const ArrayBase<Derived3>& bc0) const {
      Array<Real, n, n> dbc_dya;
      Array<Real, n, n> dbc_dyb;

      Array<Real, n, 1> y_new;

      Array<Real, n, 1> bc_new;
      Array<Real, n, 1> h;

      h = sqrt_EPS * (1 + ya.abs());
      for (Index j = 0; j < n; ++j) {
        y_new = ya;
        y_new(j) += h(j);
        bc_new = bc_(y_new, yb);
        Real hj = y_new(j) - ya(j);
        for (Index i = 0; i < n; ++i) {
         dbc_dya(i, j) = (bc_new(i) - bc0(i)) / hj;
        }
      }

      h = sqrt_EPS * (1 + yb.abs());
      for (Index j = 0; j < n; ++j) {
        y_new = yb;
        y_new(j) += h(j);
        Real hj = y_new(j) - yb(j);
        bc_new = bc_(ya, y_new);
        for (Index i = 0; i < n; ++i) {
          dbc_dyb(i, j) = (bc_new(i) - bc0(i)) / hj;
        }
      }

      return std::make_tuple(dbc_dya, dbc_dyb);
    };

    template <typename Derived1, typename Derived2>
    inline std::tuple<Array<Real, n, n>, Array<Real, n, n>> operator()(const ArrayBase<Derived1>& x,
                                                                       const ArrayBase<Derived2>& y) const {
      auto bc0 = bc_(x, y);
      return (*this)(x, y, bc0);
    };
  private:
    BC& bc_;
    static constexpr Real sqrt_EPS = sqrt(boost::math::tools::epsilon<Real>());
};

} // namespace detail
} // namespace collocation

#endif // COLLOCATION_DETAIL_ESTIMATE_BC_JAC
