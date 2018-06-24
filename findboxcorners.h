//
// Created by swinkx on 15/02/18.
//

#ifndef CALIBRATIONCPP_FINDBOXCORNERS_H
#define CALIBRATIONCPP_FINDBOXCORNERS_H

#include <vector>

#include <opencv2/core.hpp>

#include "boxsize.hpp"
#include "findboxplanes.h"
#include "utils.h"

class FindBoxCorners {
public:
    explicit FindBoxCorners(const std::vector<cv::Point3d> &points, const BoxSize &box_size);

    const std::array<cv::Point3d, 7>& get_corners() { return final_refined_corners;};
private:
    std::array<int, 3> calc_plane_ordering(const std::array<std::vector<cv::Point3d>, 3> &plane_points,
                                           const std::array<Plane, 3> &planes,
                                           std::array<std::vector<cv::Point3d>, 3> &outlier_free_points,
                                           std::array<cv::Point3d, 7> &corners);

    std::array<cv::Point3d, 7> get_corners(const std::array<std::vector<cv::Point3d>, 3> &array,
                                           const std::array<Plane, 3> &ordering);

    bool flip_edge(const cv::Point3d &mid_corner,
                   const cv::Mat &edge,
                   const std::array<std::vector<cv::Point3d>, 2> &points);

    unsigned long calc_box_inliers(const std::array<std::vector<cv::Point3d>, 3> &plane_points,
                                   const std::array<Plane, 3> &planes,
                                   const std::array<cv::Point3d, 7> &corners,
                                   std::array<std::vector<cv::Point3d>, 3> &inliers,
                                   std::vector<cv::Point3d> &all_points, std::vector<cv::Point3d> &all_colors);

    std::vector<cv::Point3d> transform_to_origin(const Plane &plane,
                                                 const std::vector<cv::Point3d> &points);


    std::pair<cv::Mat, cv::Point3d> transform_to_match_Z2(const std::array<Plane, 3> &planes,
                                                          const std::array<std::vector<cv::Point3d>, 3> &points,
                                                          std::array<std::vector<cv::Point3d>, 3> &transformed);

    std::vector<cv::Point3d> project_to_plane(const cv::Mat &plane_normal,
                                              const std::vector<cv::Point3d> &points);

    std::array<Plane, 3> fit_cube_ransac(const std::array<std::vector<cv::Point3d>, 3> &points,
                                         std::array<std::vector<cv::Point3d>, 3> &inliers);

    std::array<std::vector<cv::Point3d>, 3> inliers_to_box(const std::array<std::vector<cv::Point3d>, 3> &points,
                                                           const std::array<Plane, 3> &planes);

    void refine_planes(const std::array<std::vector<cv::Point3d>, 3> &points, std::array<Plane, 3> &planes);

    double get_rotation_angle(const std::vector<cv::Point3d> &points1, const std::vector<cv::Point3d> &points2);

    Plane rotate_plane(const double angle, const std::pair<cv::Mat, cv::Point3d> &transform, const Plane &plane);

    cv::Mat
    get_optimal_translation(const std::array<Plane, 3> &planes, const std::array<std::vector<cv::Point3d>, 3> &points);

private:
    std::array<cv::Point3d, 7> final_refined_corners;
    const BoxSize BOX_SIZE;
    const std::array<cv::Point3d, 7> corner_colors{cv::Point3d(0, 0, 0),
                                                   cv::Point3d(255, 0, 0),
                                                   cv::Point3d(0, 255, 0),
                                                   cv::Point3d(0, 0, 255),
                                                   cv::Point3d(255, 255, 0),
                                                   cv::Point3d(0, 255, 255),
                                                   cv::Point3d(255, 0, 255)};

    const float R = 0.01;

    const int RANSAC_FIRST_RUN = 50;
    const int RANSAC_SECOND_RUN = 30;
    const int RANSAC_THIRD_RUN = 70;

//    const float RANSAC_PLANE_THRESHOLD = 0.005; //lidar - 16
    const float RANSAC_PLANE_THRESHOLD = 0.015; //lidar - 64

    const float REFINE_ITERATION_THRESHOLD = 0.000001;

    cv::Point3d get_intersection_point(const std::array<Plane, 3> &planes) const;
};


#endif //CALIBRATIONCPP_FINDBOXCORNERS_H
