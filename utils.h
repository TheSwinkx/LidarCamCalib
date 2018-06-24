#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <fstream>
#include <opencv2/core.hpp>
#include <array>
#include <random>

#include "plane.hpp"

namespace utils {

    std::vector<cv::Point3d> read_xyz(const std::string &fn);
    void write_ply(const std::string &fn,
                   const std::vector<cv::Point3d> &points,
                   const std::vector<cv::Point3d> &colors = {},
                   const std::vector<cv::Point3d> &normals = {});

    std::array<int, 3> gen_randoms(const unsigned int max_rand);

    Plane fit_plane_to_points(const std::array<cv::Point3d, 3> &plane_points);

    int add_sphere(std::vector<cv::Point3d> &points, const cv::Point3d center, const double r = 1.0d);
};

#endif