#include <camera/camera.hpp>

#define BOOST_TEST_MODULE UnitTests
#include <boost/test/unit_test.hpp>

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