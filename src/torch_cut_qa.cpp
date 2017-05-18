#include "cat_laser_scan_qa/torch_cut_qa.h"

#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/segmentation/sac_segmentation.h>

namespace
{

const int NUM_MAX_ITER_RANSAC = 4000;

Eigen::Vector4f fitPlaneManually(const pcl::PointCloud<pcl::PointXYZ>& cloud)
{
  Eigen::MatrixXd lhs (cloud.size(), 3);
  Eigen::VectorXd rhs (cloud.size());

  for (size_t i = 0; i < cloud.size(); ++i)
  {
    const auto& pt = cloud.points[i];
    lhs(i, 0) = pt.x;
    lhs(i, 1) = pt.y;
    lhs(i, 2) = 1.0;

    rhs(i) = -1.0 * pt.z;
  }

  Eigen::Vector3d params = lhs.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(rhs);

  Eigen::Vector3d normal (params(0), params(1), 1.0);
  auto length = normal.norm();

  normal /= length;
  params(2) /= length;

  return {normal(0), normal(1), normal(2), params(2)};
}

Eigen::Vector4f computeSurfacePlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                    const double surface_tolerance)
{
  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZ> seg;

  // Mandatory
  seg.setInputCloud(cloud);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(NUM_MAX_ITER_RANSAC);
  seg.setDistanceThreshold(surface_tolerance);
  // Optional
  seg.setOptimizeCoefficients(true);

  // Segment the largest planar component from the remaining cloud
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
  seg.segment (*inliers, *coefficients);

  assert(coefficients->values.size() == 4); // A plane should have 4 parameters
  auto vector = Eigen::Vector4f(coefficients->values.data());

  // Refine plane with custom plane fitter
  pcl::PointCloud<pcl::PointXYZ> inlier_points;
  pcl::copyPointCloud(*cloud, inliers->indices, inlier_points);

  return fitPlaneManually(inlier_points);
}

std::vector<double> computeDistances(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                     const Eigen::Vector4f& plane_model)
{
  std::vector<double> results (cloud->size());

  std::transform(cloud->begin(), cloud->end(), results.begin(), [plane_model] (const pcl::PointXYZ& pt)
  {
    return pcl::pointToPlaneDistanceSigned(pt, plane_model);
  });

  return results;
}

pcl::PointIndices extractHighPoints(const std::vector<double>& plane_distances, const double tolerance)
{
  pcl::PointIndices high_points;

  // Forgive the cast, 'indices' is a vector of ints so I use an int here
  for (int i = 0; i < static_cast<int>(plane_distances.size()); ++i)
  {
    if (plane_distances[i] > tolerance)
    {
      high_points.indices.push_back(i);
    }
  }
  return high_points;
}

} // end of anonymous namespace

cat_laser_scan_qa::TorchCutQAResult
cat_laser_scan_qa::runQualityAssurance(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                       const TorchCutQAParameters &params)
{
  Eigen::Vector4f surface_plane_model = computeSurfacePlane(cloud, params.surface_tolerance);

  // We define a positive distance as along the plane normal. The plane normal is assume to be pointing
  // out of the surface. So positive distances are points that are "out of spec" and need to be removed.
  // TODO: Test positive direction based on some reference point

  // Compute the distance from each point in cloud to the surface plane
  std::vector<double> distances = computeDistances(cloud, surface_plane_model);

  // Extract the indices of any point that is more than 'surface tolerance' above the nominal plane
  pcl::PointIndices high_indices = extractHighPoints(distances, params.surface_tolerance * 1.0);

  TorchCutQAResult result;
  result.surface_plane_model = surface_plane_model;
  result.point_to_plane_distances = distances;
  result.high_point_indices = high_indices;
  return result;
}
