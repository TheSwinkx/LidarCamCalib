#include <iostream>
#include <opencv2/core.hpp>

#include "boxsize.hpp"
#include "utils.h"
#include "findboxcorners.h"
#include "select_image_corners.h"


int main() {

    const std::string LIDAR_FN = "/home/swinkx/Programming/CalibrationCPP/data/Velo-64.xyz";
    const std::string IMAGE_FN = "/home/swinkx/Programming/CalibrationCPP/data/Cam 2 - Frame 00465.jpg";
    const BoxSize boxSize(0.31, 0.39, 0.263); //meters

    const cv::Mat cam_mat = (cv::Mat_<double>(3, 3) << 774.3316790629223, 0, 644.4686880279397,
                                                        0, 781.0520626286016, 473.6598129715586,
                                                        0, 0, 1);
    const cv::Mat distcoeffs = (cv::Mat_<double>(1, 5) << -0.3079287463017346, 0.07606051754199189, 0, 0, 0);

    auto point_cloud = utils::read_xyz(LIDAR_FN);

    FindBoxCorners corner_finder(point_cloud, boxSize);
    auto corners = corner_finder.get_corners();

    std::cout << "Please select the image corners one by one, by the colored corners in final_cube.ply."
                 "Then press ESC!" << std::endl;

    auto image_corners = SelectImageCorners::select_image_points(IMAGE_FN);

    std::vector<cv::Point3d> object_corners(corners.begin(), corners.end());
    cv::Mat rvec, tvec;
    cv::solvePnP(object_corners, image_corners, cam_mat, distcoeffs, rvec, tvec, false, cv::SOLVEPNP_EPNP);

    cv::Mat r_mat;
    cv::Rodrigues(rvec, r_mat);
    std::cout << "LiDAR - camera transformation: \nR:" << r_mat << "\nt:" << tvec << std::endl;


    return 0;
}