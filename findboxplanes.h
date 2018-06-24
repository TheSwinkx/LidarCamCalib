#ifndef FINDBOXPLANES_H
#define FINDBOXPLANES_H

#include <vector>
#include <array>
#include <random>

#include <opencv2/core.hpp>

#include "utils.h"


class FindBoxPlanes {
public:
    explicit FindBoxPlanes(const std::vector<cv::Point3d> &points);

    const std::array<std::vector<cv::Point3d>, 3> &get_final_plane_points() const;

    const std::array<Plane, 3> &get_final_planes() const;

private:
    std::array<Plane, 3> select_orthogonals(const std::vector<Plane> &planes);

    std::vector<Plane> find_planes_ransac(const std::vector<cv::Point3d> &points);

    Plane ransac_plane(const std::vector<cv::Point3d> &points,
                       size_t &remaining_points,
                       const std::vector<bool> &not_used,
                       std::vector<std::size_t> &inliers);

    void calc_inliers(const std::vector<cv::Point3d> &points,
                      const std::vector<bool> &not_used,
                      const Plane &plane,
                      std::vector<std::size_t> &inliers);

    std::array<cv::Point3d, 3> get_random_points(const std::vector<cv::Point3d> &points,
                                                 const size_t remaining_points,
                                                 const std::vector<bool> &not_used);
private:
    const int MAX_RANSAC_ITERATIONS = 500;
    const unsigned int MAX_REMAINING_POINTS = 10;
    const double MAX_PLANE_DIST = 0.03; //0.015

    std::array<std::vector<cv::Point3d>, 3> final_plane_points;
    std::array<Plane, 3> final_planes;
};

#endif