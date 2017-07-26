#ifndef CAT_TORCH_CUT_QA_H
#define CAT_TORCH_CUT_QA_H

#include <pcl_ros/point_cloud.h>

namespace cat_laser_scan_qa
{

struct TorchCutQAParameters
{
  double surface_tolerance; // (m) Max acceptable distance (+/-) to the ideal top surface
  double plane_fit_ratio; // Percentage of the surface tolerance used to fit the plane
};


struct TorchCutQAResult
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Vector4f surface_plane_model;
  std::vector<double> point_to_plane_distances; // (m) signed distance from every point in input to plane
  pcl::PointIndices high_point_indices; // Indices (from 'input') of points judged to be 'out of spec'
};

/**
 * @brief runQualityAssurance
 * @param cloud
 * @param params
 * @return
 */
TorchCutQAResult runQualityAssurance(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                                     const TorchCutQAParameters& params);

}

#endif // CAT_TORCH_CUT_QA_H
