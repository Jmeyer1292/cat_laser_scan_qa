#include <ros/ros.h>

#include <cat_laser_scan_qa/torch_cut_qa.h>
#include <pcl/filters/extract_indices.h>

// Returns a pair of clouds, first = the outliers of the cloud
//                           second = the inliers of the cloud
std::pair<pcl::PointCloud<pcl::PointXYZ>, pcl::PointCloud<pcl::PointXYZ>>
segmentCloud(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, const pcl::PointIndices& outliers)
{
  std::pair<pcl::PointCloud<pcl::PointXYZ>, pcl::PointCloud<pcl::PointXYZ>> result;

  // Create the filtering object
  pcl::ExtractIndices<pcl::PointXYZ> extract;


  // Extract the inliers
  extract.setInputCloud(cloud);
  extract.setIndices( boost::make_shared<pcl::PointIndices>(outliers) );
  extract.setNegative(false);
  extract.filter(result.first);

  // Create the filtering object
  extract.setNegative (true);
  extract.filter(result.second);

  result.first.header.frame_id = result.second.header.frame_id = cloud->header.frame_id;

  return result;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "torch_cut_qa_test_node");

  ros::NodeHandle pnh ("~");

  // Load file path
  std::string input_pcd_filepath;
  if (!pnh.getParam("pcd_file", input_pcd_filepath))
  {
    ROS_ERROR("Toch Cut QA Node requires that the 'pcd_file' parameter be set.");
    return 2;
  }

  // Load PCD data
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>());
  if (pcl::io::loadPCDFile(input_pcd_filepath, *cloud) == -1)
  {
    ROS_ERROR("Unable to load pcd file from path '%s'.", input_pcd_filepath.c_str());
    return 1;
  }

  // Run algorithm
  cat_laser_scan_qa::TorchCutQAParameters params;
  params.surface_tolerance = pnh.param<double>("surface_tolerance", 0.0005); // (m) A 1mm test tolerance
  params.plane_fit_ratio = pnh.param<double>("plane_fit_ratio", 1.0);
  auto result = cat_laser_scan_qa::runQualityAssurance(cloud, params);

  // Report results
  const auto n_total_points = result.point_to_plane_distances.size();
  const auto n_points_oot = result.high_point_indices.indices.size(); // oot = 'out of tolerance'
  const auto percentage_oot = static_cast<double>(n_points_oot) / n_total_points;

  ROS_INFO("Computed QA results: %lu of %lu (or %f \%) points were out of specification",
           n_points_oot, n_total_points, percentage_oot);

  ROS_INFO_STREAM("Plane parameters: " << result.surface_plane_model.transpose());

  // Publish visualization
  ros::NodeHandle nh;
  ros::Publisher cloud_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZ>>("cloud", 0, true);
  ros::Publisher in_points = nh.advertise<pcl::PointCloud<pcl::PointXYZ>>("in_cloud", 0, true);
  ros::Publisher oot_points = nh.advertise<pcl::PointCloud<pcl::PointXYZ>>("oot_cloud", 0, true);

  cloud->header.frame_id = "base_link";
  auto segment = segmentCloud(cloud, result.high_point_indices);

  cloud_pub.publish(cloud);
  in_points.publish(segment.second);
  oot_points.publish(segment.first);

  ros::spin();
}
