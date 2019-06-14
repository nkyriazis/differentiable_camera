#include <camera/camera.hpp>

int main(int argc, char **argv)
{
  // const auto x0 = Eigen::MakeAutoDiffScalar(1.0, Eigen::Vector3d{1, 1, 1});
  // const auto x1 = Eigen::MakeAutoDiffScalar(2.0, Eigen::Vector3d{1, 1, 1});
  // const auto x2 = differentiable_camera::detail::eval(x0 + x1);

  using Jet        = Eigen::Vector<double, 2 + 12>;
  using Scalar     = Eigen::AutoDiffScalar<Jet>;
  using Vec2       = Eigen::Vector<Scalar, 2>;
  using DistCoeffs = Eigen::Vector<Scalar, 12>;

  Vec2 nc{Scalar{0.2, 14, 0}, Scalar{0.3, 14, 1}};
  DistCoeffs k{Scalar{1, 14, 2},
               Scalar{1, 14, 3},
               Scalar{1, 14, 4},
               Scalar{1, 14, 5},
               Scalar{1, 14, 6},
               Scalar{0, 14, 7},
               Scalar{0, 14, 8},
               Scalar{0, 14, 9},
               Scalar{0, 14, 10},
               Scalar{0, 14, 11},
               Scalar{0, 14, 12},
               Scalar{0, 14, 13}};

  // const auto res =
  const auto dist =
    differentiable_camera::distort_normalized_coordinates(nc, k);
  const auto undist =
    differentiable_camera::undistort_normalized_coordinates(dist, k);
  return 0;
}