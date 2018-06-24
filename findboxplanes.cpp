#include <iostream>
#include "findboxplanes.h"

using namespace std;
using namespace cv;


FindBoxPlanes::FindBoxPlanes(const vector<Point3d> &points) {
    auto planes = find_planes_ransac(points);

    final_planes = select_orthogonals(planes);

    vector<Point3d> outliers;

    for (const auto &point : points) {
        Mat hom_point = (Mat_<double>(1, 4) << point.x, point.y, point.z, 1.0d);

        double dists[3];
        double min = 10e7;
        for (int i = 0; i < 3; i++) {
            dists[i] = abs(final_planes[i].dot(hom_point));
            if (dists[i] < min)
                min = dists[i];
        }

        if (min > MAX_PLANE_DIST) {
            outliers.emplace_back(point);
        } else {
            auto min_dist_place = min_element(begin(dists), end(dists));
            auto min_dist_index = static_cast<size_t>(distance(begin(dists), min_dist_place));

            final_plane_points[min_dist_index].emplace_back(point);
        }
    }

    vector<Point3d> all_planes;
    vector<Point3d> all_colors;
    for (size_t i = 0; i < final_plane_points.size(); i++) {
        Point3d color;

        if (i < 3)
            color = Point3d(i == 0 ? 255 : 0, i == 1 ? 255 : 0, i == 2 ? 255 : 0);
        else {
            color = Point3d(0, 0, 0);
        }

        for (const auto &p : final_plane_points[i]) {
            all_planes.emplace_back(p);
            all_colors.emplace_back(color);
        }
    }

    for (const auto &p : outliers) {
        all_planes.emplace_back(p);
        all_colors.emplace_back(Point3d(0, 0, 0));
    }

    ::utils::write_ply("planes.ply", all_planes, all_colors);
}

vector<Plane> FindBoxPlanes::find_planes_ransac(const vector<Point3d> &points) {

    vector<Plane> planes;

    size_t remaining_points = points.size();
    vector<bool> not_used(remaining_points, true);

    while (true) {

        vector<size_t> inliers;
        Plane plane = ransac_plane(points, remaining_points, not_used, inliers);

        std::cout << "\tfound plane: " << inliers.size() << " of " << remaining_points << std::endl;

        for (const auto &inlier_ind : inliers)
            not_used[inlier_ind] = false;

        remaining_points -= inliers.size();

        planes.emplace_back(plane);

        if (remaining_points < MAX_REMAINING_POINTS)
            break;
    }

    return planes;
}

Plane FindBoxPlanes::ransac_plane(const vector<Point3d> &points,
                                  size_t &remaining_points,
                                  const vector<bool> &not_used,
                                  vector<size_t> &inliers) {

    size_t max_inliers = 0;
    Plane best_plane;
    vector<size_t> best_plane_inliers;

    for (int i = 0; i < MAX_RANSAC_ITERATIONS; i++) {

        auto plane_points = get_random_points(points, remaining_points, not_used);

        Plane plane = ::utils::fit_plane_to_points(plane_points);

        vector<size_t> p_inliers;
        p_inliers.reserve(remaining_points);

        calc_inliers(points, not_used, plane, p_inliers);

        if (p_inliers.size() > max_inliers) {
            max_inliers = p_inliers.size();
            best_plane = plane;
            best_plane_inliers = p_inliers;
        }
    }

    inliers = best_plane_inliers;

    return best_plane;
}

void FindBoxPlanes::calc_inliers(const vector<Point3d> &points,
                                 const vector<bool> &not_used,
                                 const Plane &plane,
                                 vector<size_t> &inliers) {

    inliers.clear();

    for (size_t i = 0; i < points.size(); i++) {

        if (not_used[i]) {
            Mat hom_p = (Mat_<double>(1, 4) << points[i].x, points[i].y, points[i].z, 1.0d);

            double d = plane.dot(hom_p);
            if (abs(d) < MAX_PLANE_DIST) {
                inliers.push_back(i);
            }
        }

    }
}

array<Point3d, 3> FindBoxPlanes::get_random_points(const vector<Point3d> &points,
                                                   const size_t remaining_points,
                                                   const vector<bool> &not_used) {

    auto randoms = ::utils::gen_randoms(static_cast<unsigned int>(remaining_points));

    if (randoms[0] == randoms[1] || randoms[1] == randoms[2]) {
        std::cerr << "Randoms are equal! " << __FILE__ << " : " << __LINE__ << std::endl;
    }

    array<Point3d, 3> random_points;

    int counter = 0;
    size_t found = 0;

    for (size_t i = 0; i < not_used.size(); i++) {
        if (!not_used[i])
            continue;

        for (size_t j = found; j < randoms.size(); j++) {
            if (counter == randoms[j]) {
                random_points[j] = points[i];
                ++found;

                break;
            }
        }

        if (found >= randoms.size()) {
            break;
        }

        ++counter;
    }

    return random_points;
}


array<Plane, 3> FindBoxPlanes::select_orthogonals(const vector<Plane> &planes) {
    array<Plane, 3> orthogonal_planes;

    double lowest_value = 10e7;
    for (size_t i = 0; i < planes.size(); i++) {
        Mat plane1_normal = planes[i].get_normal();

        for (size_t j = i + 1; j < planes.size(); j++) {
            Mat plane2_normal = planes[j].get_normal();

            for (size_t k = j + 1; k < planes.size(); k++) {
                Mat plane3_normal = planes[k].get_normal();

                double angle1 = plane1_normal.dot(plane2_normal);
                double angle2 = plane2_normal.dot(plane3_normal);
                double angle3 = plane3_normal.dot(plane1_normal);

                double sum = abs(angle1) + abs(angle2) + abs(angle3);

                if (sum < lowest_value) {
                    lowest_value = sum;
                    orthogonal_planes[0] = planes[i];
                    orthogonal_planes[1] = planes[j];
                    orthogonal_planes[2] = planes[k];
                }
            }
        }
    }

    return orthogonal_planes;
}

const array<vector<Point3d>, 3> &FindBoxPlanes::get_final_plane_points() const {
    return final_plane_points;
}

const array<Plane, 3> &FindBoxPlanes::get_final_planes() const {
    return final_planes;
}
