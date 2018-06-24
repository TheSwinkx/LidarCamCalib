//
// Created by swinkx on 21/02/18.
//

#ifndef CALIBRATIONCPP_PLANE_HPP
#define CALIBRATIONCPP_PLANE_HPP

struct Plane {

    Plane() {
        plane = cv::Mat::zeros(1, 4, CV_64FC1);
    }

    Plane(const Plane &p) {
        plane = p.plane.clone();
    }

    Plane(const cv::Mat &normal, const cv::Point3d &plane_point) {
        double w = -1 * normal.dot(cv::Mat(plane_point));
        plane = (cv::Mat_<double>(1, 4) << normal.at<double>(0), normal.at<double>(1), normal.at<double>(2), w);
    }

    Plane(double nx, double ny, double nz, double w) {
        plane = (cv::Mat_<double>(1, 4) << nx, ny, nz, w);
    }

    Plane(const cv::Mat &normal, double w) {
        plane = (cv::Mat_<double>(1, 4) << normal.at<double>(0), normal.at<double>(1), normal.at<double>(2), w);
    }

    Plane& operator= (const Plane &other) {
        plane = other.plane.clone();
        return *this;
    }

    double dot(const cv::Mat &m) const {
        return plane.dot(m);
    }

    cv::Mat get_normal() const {
        return plane.colRange(0, 3).t();
    }

    double get_w() const {
        return plane.at<double>(0, 3);
    }

    cv::Point3d get_plane_point() const {
        double d = -get_w();
        return {plane.at<double>(0, 0) * d, plane.at<double>(0, 1) * d, plane.at<double>(0, 2) * d};
    }

    void flip_normal() {
        plane = -1 * plane;
    }

    void translate(const cv::Mat &t) {
        cv::Mat translated_plane_point = cv::Mat(get_plane_point()) + t;
        plane.at<double>(0, 3) = -translated_plane_point.dot(get_normal());
    }

private:
    cv::Mat plane{};
};

#endif //CALIBRATIONCPP_PLANE_HPP
