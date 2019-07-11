#include <boost/math/constants/constants.hpp>
#include <camera/camera.hpp>
#include <unsupported/Eigen/NumericalDiff>

#define BOOST_TEST_MODULE UnitTests
#include <boost/test/unit_test.hpp>

static constexpr auto PI = boost::math::constants::pi<double>();

BOOST_AUTO_TEST_CASE(Eval)
{
  struct non_fundamental
  {
    int eval() const { return 2; }
  };
  BOOST_CHECK_EQUAL(1.0, differentiable_camera::detail::eval(1.0));
  BOOST_CHECK_EQUAL(2, differentiable_camera::detail::eval(non_fundamental{}));
}

BOOST_AUTO_TEST_CASE(InvertibleNC2WND)
{
  for (int i = 0; i < 100; ++i)
  {
    const Eigen::Vector2d nc    = Eigen::Vector2d::Random();
    const Eigen::Vector2d focal = Eigen::Vector2d::Random();
    const Eigen::Vector2d cop   = Eigen::Vector2d::Random();

    const Eigen::Vector2d wnd =
      differentiable_camera::normalized_coordinates_to_window_coordinates(
        nc, focal, cop);
    const Eigen::Vector2d nc_ =
      differentiable_camera::window_coordinates_to_normalized_coordinates(
        wnd, focal, cop);

    BOOST_CHECK_LT((nc - nc_).norm(), 1e-9);
  }
}

BOOST_AUTO_TEST_CASE(InvertibleDist2Undist)
{
  for (int i = 0; i < 100; ++i)
  {
    const Eigen::Vector2d nc = Eigen::Vector2d::Random();
    const Eigen::Vector<double, 12> k =
      Eigen::Vector<double, 12>::Random() * 0.01;

    const auto nc_ =
      differentiable_camera::distort_normalized_coordinates(nc, k);
    const auto nc__ =
      differentiable_camera::undistort_normalized_coordinates(nc_, k);

    BOOST_CHECK_LT((nc - nc__).norm(), 1e-9);
  }
}

template <typename Selector_>
struct JacFunctor
{
  using Scalar       = double;
  using InputType    = Eigen::Vector<Scalar, 14>;
  using ValueType    = Eigen::Vector2d;
  using JacobianType = Eigen::Matrix<double, 2, 14>;
  using Selector     = Selector_;
  enum
  {
    InputsAtCompileTime = 1,
    ValuesAtCompileTime = 1
  };

  static auto values() { return 2; }

  void operator()(const InputType& x, ValueType& out) const
  {
    out = Selector::execute(x.head<2>(), x.tail<12>());
  }
};

struct DistortSelector
{
  template <typename X, typename Y>
  static auto execute(X&& x, Y&& y)
  {
    return differentiable_camera::distort_normalized_coordinates(x, y);
  }
};

struct UndistortSelector
{
  template <typename X, typename Y>
  static auto execute(X&& x, Y&& y)
  {
    return differentiable_camera::undistort_normalized_coordinates(x, y);
  }
};

using JacFunctors =
  std::tuple<JacFunctor<DistortSelector>, JacFunctor<UndistortSelector>>;

BOOST_AUTO_TEST_CASE_TEMPLATE(DistortGradient, JacFunctor, JacFunctors)
{
  using Selector = typename JacFunctor::Selector;

  Eigen::NumericalDiff<JacFunctor, Eigen::Central> functor;

  for (int i = 0; i < 100; ++i)
  {
    const auto input = (JacFunctor::InputType::Random() * 0.1).eval();
    JacFunctor::JacobianType jac;
    functor.df(input, jac);

    using DScalar = Eigen::AutoDiffScalar<Eigen::Vector<double, 14>>;
    using DVec2   = Eigen::Vector<DScalar, 2>;
    using DVec12  = Eigen::Vector<DScalar, 12>;

    const auto out = Selector::execute(
      DVec2{DScalar{input(0), 14, 0}, DScalar{input(1), 14, 1}},
      DVec12{DScalar{input(2), 14, 2},
             DScalar{input(3), 14, 3},
             DScalar{input(4), 14, 4},
             DScalar{input(5), 14, 5},
             DScalar{input(6), 14, 6},
             DScalar{input(7), 14, 7},
             DScalar{input(8), 14, 8},
             DScalar{input(9), 14, 9},
             DScalar{input(10), 14, 10},
             DScalar{input(11), 14, 11},
             DScalar{input(12), 14, 12},
             DScalar{input(13), 14, 13}});

    for (Eigen::Index i = 0; i < 2; ++i)
    {
      BOOST_CHECK_LT(
        (jac.row(i).transpose() - out(i).derivatives()).cwiseAbs().maxCoeff(),
        1e-6);
    }
  }
}

BOOST_AUTO_TEST_CASE(Wnd2Ray)
{
  const Eigen::Vector2d f{1, 1};
  const Eigen::Vector2d c{0, 0};
  Eigen::Vector<double, 12> k;
  k.setZero();

  {
    const auto ray =
      differentiable_camera::window_to_ray(Eigen::Vector2d{0, 0}, f, c, k);
    BOOST_CHECK_CLOSE(ray.x(), 0, 1e-9);
    BOOST_CHECK_CLOSE(ray.y(), 0, 1e-9);
    BOOST_CHECK_CLOSE(ray.z(), 1, 1e-9);
  }

  {
    const auto ray =
      differentiable_camera::window_to_ray(Eigen::Vector2d{1, 0}, f, c, k);
    BOOST_CHECK_CLOSE(ray.x(), cos(PI / 4), 1e-9);
    BOOST_CHECK_CLOSE(ray.y(), 0, 1e-9);
    BOOST_CHECK_CLOSE(ray.z(), sqrt(1 - pow(cos(PI / 4), 2)), 1e-9);
  }

  {
    const auto ray =
      differentiable_camera::window_to_ray(Eigen::Vector2d{0, 1}, f, c, k);
    BOOST_CHECK_CLOSE(ray.x(), 0, 1e-9);
    BOOST_CHECK_CLOSE(ray.y(), cos(PI / 4), 1e-9);
    BOOST_CHECK_CLOSE(ray.z(), sqrt(1 - pow(cos(PI / 4), 2)), 1e-9);
  }

  {
    const auto ray =
      differentiable_camera::window_to_ray(Eigen::Vector2d{1, 1}, f, c, k);
    BOOST_CHECK_CLOSE(ray.x(), 1 / sqrt(3), 1e-9);
    BOOST_CHECK_CLOSE(ray.x(), 1 / sqrt(3), 1e-9);
    BOOST_CHECK_CLOSE(ray.x(), 1 / sqrt(3), 1e-9);
  }
}