#include "cat_laser_scan_qa/torch_cut_qa.h"

//#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/ransac.h>

namespace
{


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
  const int NUM_MAX_ITER_RANSAC = 5000;
  seg.setMaxIterations(NUM_MAX_ITER_RANSAC);
  seg.setDistanceThreshold(surface_tolerance);
  seg.setProbability(1);
  seg.setAxis(Eigen::Vector3f(0, 0, 1));
  seg.setEpsAngle(0.3);
  // Optional
//  seg.setOptimizeCoefficients(true);

  // Segment the largest planar component from the remaining cloud
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
  seg.segment (*inliers, *coefficients);

  assert(coefficients->values.size() == 4); // A plane should have 4 parameters
//  auto vector = Eigen::Vector4f(coefficients->values.data());
//  return vector;

  // Refine plane with custom plane fitter
  pcl::PointCloud<pcl::PointXYZ> inlier_points;
  pcl::copyPointCloud(*cloud, inliers->indices, inlier_points);
  auto vector = fitPlaneManually(inlier_points);
  ROS_INFO_STREAM("Plane eq: " << vector.transpose());
  return vector;
}

Eigen::Vector4f computeSurfacePlane2(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                     const double surface_tolerance)
{
  pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr
    model_p (new pcl::SampleConsensusModelPlane<pcl::PointXYZ> (cloud));

  pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model_p);
  ransac.setDistanceThreshold(surface_tolerance);
  ransac.setMaxIterations(5000);
  ransac.setProbability(1);
  ransac.computeModel();

  Eigen::VectorXf vector;
  std::vector<int> inliers;

  ransac.getModelCoefficients(vector);
  ransac.getInliers(inliers);
  ROS_INFO_STREAM("Ransac Plane eq: " << vector.transpose());


  // Segment the largest planar component from the remaining cloud
//  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
//  pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
//  seg.segment (*inliers, *coefficients);

//  assert(coefficients->values.size() == 4); // A plane should have 4 parameters
//  auto vector = Eigen::Vector4f(coefficients->values.data());
//  return vector;

  // Refine plane with custom plane fitter
  pcl::PointCloud<pcl::PointXYZ> inlier_points;
  pcl::copyPointCloud(*cloud, inliers, inlier_points);
  auto vector2 = fitPlaneManually(inlier_points);
  ROS_INFO_STREAM("Plane eq: " << vector2.transpose());
  return vector2;
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

pcl::PointCloud<pcl::PointXYZ>::Ptr filterTable(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                                                const Eigen::Vector4f& plane_model)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr copy (new pcl::PointCloud<pcl::PointXYZ>);

  for (const auto& pt : cloud)
  {
    if (pcl::pointToPlaneDistance(pt, plane_model) > 0.02)
    {
      copy->push_back(pt);
    }
  }
  return copy;
}

Eigen::Vector4f planeCoefficients(const Eigen::Vector3d& point, const Eigen::Vector3d& normal)
{
  Eigen::Vector3f pointf = point.cast<float>();
  Eigen::Vector3f normalf = normal.cast<float>();

  // ax + by + cz + d = 0
  Eigen::Vector4f result;
  result(0) = normalf(0);
  result(1) = normalf(1);
  result(2) = normalf(2);
  result(3) = -(pointf.dot(normalf));

  return result;
}

} // end of anonymous namespace

cat_laser_scan_qa::TorchCutQAResult
cat_laser_scan_qa::runQualityAssurance(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                       const TorchCutQAParameters &params)
{
  // Pre-filter the data
  const Eigen::Vector3d plane_pt {1.40498, 0.0583, 0.92378 - 0.72871};
  const Eigen::Vector3d plane_normal {0, 0, 1.0};

  ROS_INFO_STREAM("Before table removal: " << cloud->size());
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered = filterTable(*cloud, planeCoefficients(plane_pt, plane_normal));
  ROS_INFO_STREAM("After table removal: " << filtered->size());

  // Find the top plane in the part data
  Eigen::Vector4f surface_plane_model = computeSurfacePlane2(filtered,
                                                            params.plane_fit_ratio * params.surface_tolerance);

  // We define a positive distance as along the plane normal. The plane normal is assume to be pointing
  // out of the surface. So positive distances are points that are "out of spec" and need to be removed.
  // TODO: Test positive direction based on some reference point

  // Compute the distance from each point in cloud to the surface plane
  std::vector<double> distances = computeDistances(cloud, surface_plane_model);

  // Extract the indices of any point that is more than 'surface tolerance' above the nominal plane
  pcl::PointIndices high_indices = extractHighPoints(distances, params.surface_tolerance);

  TorchCutQAResult result;
  result.surface_plane_model = surface_plane_model;
  result.point_to_plane_distances = distances;
  result.high_point_indices = high_indices;
  return result;
}
