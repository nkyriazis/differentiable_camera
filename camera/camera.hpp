#pragma once

#include <Eigen/Dense>
#include <type_traits>
#include <unsupported/Eigen/AutoDiff>

namespace differentiable_camera
{
namespace detail
{

template <typename T>
auto eval(const T &t, std::enable_if_t<std::is_fundamental_v<T>> * = nullptr)
{
  return t;
}

template <typename T>
auto eval(const T &t, std::enable_if_t<!std::is_fundamental_v<T>> * = nullptr)
{
  return t.eval();
}

template <typename Expr>
auto eval(const Eigen::AutoDiffScalar<Expr> &t)
{
  const auto value       = eval(t.value());
  const auto derivatives = eval(t.derivatives());
  return Eigen::MakeAutoDiffScalar(value, derivatives);
}

template <typename Scalar>
auto make_vec2(const Scalar &x, const Scalar &y)
{
  return Eigen::Vector<Scalar, 2>{x, y};
}

template <typename Scalar>
auto make_mat2x2(const Scalar &x,
                 const Scalar &y,
                 const Scalar &z,
                 const Scalar &w)
{
  Eigen::Matrix<Scalar, 2, 2> ret;
  ret(0, 0) = x;
  ret(0, 1) = y;
  ret(1, 0) = z;
  ret(1, 1) = w;
  return ret;
}

} // namespace detail

template <typename NC, typename Focal, typename CoP>
auto normalized_coordinates_to_window_coordinates(
  const Eigen::MatrixBase<NC> &nc,
  const Eigen::MatrixBase<Focal> &focal,
  const Eigen::MatrixBase<CoP> &cop)
{
  return nc.cwiseProduct(focal) + cop;
}

template <typename WND, typename Focal, typename CoP>
auto window_coordinates_to_normalized_coordinates(
  const Eigen::MatrixBase<WND> &wnd,
  const Eigen::MatrixBase<Focal> &focal,
  const Eigen::MatrixBase<CoP> &cop)
{
  return (wnd - cop).cwiseProduct(focal.cwiseInverse());
}

template <typename NC, typename DistCoeffs>
auto distort_normalized_coordinates(const Eigen::MatrixBase<NC> &nc,
                                    const Eigen::MatrixBase<DistCoeffs> &k)
{
  // ripped off from OpenCV source
  const auto r2 = nc.squaredNorm();
  const auto r4 = r2 * r2;
  const auto r6 = r2 * r4;

  const auto a1 = 2 * nc.x() * nc.y();
  const auto a2 = r2 + 2 * nc.x() * nc.x();
  const auto a3 = r2 + 2 * nc.y() * nc.y();

  const auto cdist   = 1 + k(0) * r2 + k(1) * r4 + k(4) * r6;
  const auto icdist2 = 1. / (1 + k(5) * r2 + k(6) * r4 + k(7) * r6);
  const auto ud0     = detail::eval(nc.x() * cdist * icdist2 + k(2) * a1 +
                                k(3) * a2 + k(8) * r2 + k(9) * r4);
  const auto vd0     = detail::eval(nc.y() * cdist * icdist2 + k(2) * a3 +
                                k(3) * a1 + k(10) * r2 + k(11) * r4);

  return detail::make_vec2(ud0, vd0);
}

namespace detail
{

template <typename NC, typename DistCoeffs, typename Distort>
auto undistort_normalized_coordinates_impl(
  const Eigen::MatrixBase<NC> &nc,
  const Eigen::MatrixBase<DistCoeffs> &k,
  Distort &&distort)
{
  using Scalar  = typename decltype(distort(nc, k))::Scalar;
  using DJac    = Eigen::Vector<Scalar, 2>;
  using DScalar = Eigen::AutoDiffScalar<DJac>;
  using DVec2   = Eigen::Vector<DScalar, 2>;

  auto ret = nc.eval();

  for (int i = 0; i < 10; ++i)
  {
    const DVec2 nc_diff = {DScalar{ret.x(), DJac{1, 0}},
                           DScalar{ret.y(), DJac{0, 1}}};

    const auto f_and_J = distort(nc_diff, k) - nc; // f(x) and Df(x)/Dx

    const auto f = detail::make_vec2(f_and_J.x().value(), f_and_J.y().value());
    const auto J = detail::make_mat2x2(f_and_J.x().derivatives()(0),
                                       f_and_J.x().derivatives()(1),
                                       f_and_J.y().derivatives()(0),
                                       f_and_J.y().derivatives()(1));
    const auto update = (J.inverse() * f).eval();
    ret -= update;
  }

  return ret;
}
} // namespace detail

template <typename NC, typename DistCoeffs>
auto undistort_normalized_coordinates(const Eigen::MatrixBase<NC> &nc,
                                      const Eigen::MatrixBase<DistCoeffs> &k)
{
  return detail::undistort_normalized_coordinates_impl(
    nc, k, [](const auto &nc, const auto &k) {
      return distort_normalized_coordinates(nc, k);
    });
}

} // namespace differentiable_camera