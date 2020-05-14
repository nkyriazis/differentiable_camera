#pragma once

#include "detail.hpp"

namespace differentiable_camera
{

template <typename NC, typename Focal, typename CoP>
auto normalized_coordinates_to_window_coordinates(
  const Eigen::MatrixBase<NC> &nc,
  const Eigen::MatrixBase<Focal> &focal,
  const Eigen::MatrixBase<CoP> &cop)
{
  return (nc.cwiseProduct(focal) + cop).eval();
}

template <typename WND, typename Focal, typename CoP>
auto window_coordinates_to_normalized_coordinates(
  const Eigen::MatrixBase<WND> &wnd,
  const Eigen::MatrixBase<Focal> &focal,
  const Eigen::MatrixBase<CoP> &cop)
{
  return ((wnd - cop).cwiseProduct(focal.cwiseInverse())).eval();
}

template <typename NC, typename DistCoeffs>
auto distort_normalized_coordinates(const Eigen::MatrixBase<NC> &nc,
                                    const Eigen::MatrixBase<DistCoeffs> &k)
{
  // ripped off from OpenCV source
  const auto r2 = nc.squaredNorm();
  const auto r4 = r2 * r2;
  const auto r6 = r2 * r4;

  const auto a1 = 2. * nc.x() * nc.y();
  const auto a2 = r2 + 2. * nc.x() * nc.x();
  const auto a3 = r2 + 2. * nc.y() * nc.y();

  const auto cdist   = 1. + k(0) * r2 + k(1) * r4 + k(4) * r6;
  const auto icdist2 = 1. / (1. + k(5) * r2 + k(6) * r4 + k(7) * r6);
  const auto ud0     = detail::eval(nc.x() * cdist * icdist2 + k(2) * a1 +
                                k(3) * a2 + k(8) * r2 + k(9) * r4);
  const auto vd0     = detail::eval(nc.y() * cdist * icdist2 + k(2) * a3 +
                                k(3) * a1 + k(10) * r2 + k(11) * r4);

  return detail::make_vec2(ud0, vd0);
}

template <typename NC, typename DistCoeffs>
auto undistort_normalized_coordinates(const Eigen::MatrixBase<NC> &nc,
                                      const Eigen::MatrixBase<DistCoeffs> &k)
{
  return detail::undistort_normalized_coordinates_impl(
    nc, k, [](const auto &nc, const auto &k) {
      return distort_normalized_coordinates(nc, k);
    });
}

template <typename WND, typename Focal, typename CoP>
auto window_to_ray_unnormalized(const Eigen::MatrixBase<WND> &wnd,
                                const Eigen::MatrixBase<Focal> &f,
                                const Eigen::MatrixBase<CoP> &c)
{
  return window_coordinates_to_normalized_coordinates(wnd, f, c)
    .homogeneous()
    .eval();
}

template <typename WND, typename Focal, typename CoP>
auto window_to_ray(const Eigen::MatrixBase<WND> &wnd,
                   const Eigen::MatrixBase<Focal> &f,
                   const Eigen::MatrixBase<CoP> &c)
{
  return window_to_ray_unnormalized(wnd, f, c).normalized();
}

template <typename WND, typename Focal, typename CoP, typename DistCoeffs>
auto window_to_ray(const Eigen::MatrixBase<WND> &wnd,
                   const Eigen::MatrixBase<Focal> &f,
                   const Eigen::MatrixBase<CoP> &c,
                   const Eigen::MatrixBase<DistCoeffs> &k)
{
  return undistort_normalized_coordinates(
           window_coordinates_to_normalized_coordinates(wnd, f, c), k)
    .homogeneous()
    .normalized();
}

template <typename RayOrigin,
          typename RayDirection,
          typename PlaneOrigin,
          typename PlaneDirection>
auto find_ray_plane_intersection_time(
  const Eigen::MatrixBase<RayOrigin> &o,
  const Eigen::MatrixBase<RayDirection> &d,
  const Eigen::MatrixBase<PlaneOrigin> &oo,
  const Eigen::MatrixBase<PlaneDirection> &dd)
{
  return dd.dot(oo - o) / dd.dot(d);
}

template <typename RayOrigin,
          typename RayDirection,
          typename PatchOrigin,
          typename PatchXAxis,
          typename PatchYAxis>
auto find_normalized_ray_patch_intersection(
  const Eigen::MatrixBase<RayOrigin> &o,
  const Eigen::MatrixBase<RayDirection> &d,
  const Eigen::MatrixBase<PatchOrigin> &oo,
  const Eigen::MatrixBase<PatchXAxis> &x,
  const Eigen::MatrixBase<PatchYAxis> &y)
{
  const auto z            = x.cross(y).normalized();
  const auto t            = find_ray_plane_intersection_time(o, d, oo, z);
  const auto intersection = o + t * d;

  Eigen::Matrix<typename decltype(z)::Scalar, 3, 3> R;
  R << x, y, z;
  return (R.inverse() * (intersection - oo)).template head<2>().eval();
}

} // namespace differentiable_camera