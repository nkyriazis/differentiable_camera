#include <camera/camera.hpp>

int main(int argc, char **argv)
{
  // const auto x0 = Eigen::MakeAutoDiffScalar(1.0, Eigen::Vector3d{1, 1, 1});
  // const auto x1 = Eigen::MakeAutoDiffScalar(2.0, Eigen::Vector3d{1, 1, 1});
  // const auto x2 = differentiable_camera::detail::eval(x0 + x1);

  Eigen::Vector2d nc{0.2, 0.3};
  Eigen::Vector<double, 12> k;
  k.setZero();
  k.head<5>().fill(1.);

  // const auto res =
  const auto dist =
    differentiable_camera::distort_normalized_coordinates(nc, k);
  const auto undist = differentiable_camera::undistort_normalized_coordinates(
    dist, k, [](const auto &nc, const auto &k) {
      return differentiable_camera::distort_normalized_coordinates(nc, k);
      ;
    });
  return 0;
}