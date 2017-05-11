#ifndef CAT_TORCH_CUT_QA_H
#define CAT_TORCH_CUT_QA_H

#include <pcl_ros/point_cloud.h>

namespace cat_laser_scan_qa
{

struct TorchCutQAParameters
{
  double surface_tolerance; // (m) Max acceptable distance (+/-) to the ideal top surface
};


struct TorchCutQAResult
{

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
