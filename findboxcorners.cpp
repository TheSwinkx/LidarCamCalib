//
// Created by swinkx on 15/02/18.
//

#include <opencv2/imgproc.hpp>
#include <iostream>
#include "findboxcorners.h"

using namespace std;
using namespace cv;

FindBoxCorners::FindBoxCorners(const vector<Point3d> &points, const BoxSize &box_size) : BOX_SIZE(box_size) {
    FindBoxPlanes plane_finder(points);

    const auto &planes = plane_finder.get_final_planes();
    const auto &plane_points = plane_finder.get_final_plane_points();

    array<vector<Point3d>, 3> inlier_plane_points;
    array<Point3d, 7> corners;
    auto ordering = calc_plane_ordering(plane_points, planes, inlier_plane_points, corners);

    vector<Point3d> all_points;
    vector<Point3d> all_colors;

    for (const auto &plane : inlier_plane_points) {
        for (const auto &p : plane) {
            all_points.push_back(p);
            all_colors.emplace_back();
        }
    }

    ::utils::write_ply("inliers.ply", all_points, all_colors);

    array<Plane, 3> act_plane_ordering = {planes[ordering[0]], planes[ordering[1]], planes[ordering[2]]};

    array<vector<Point3d>, 3> cube_ransac_inliers;
    auto cube_planes = fit_cube_ransac(inlier_plane_points, cube_ransac_inliers);

    all_points.clear();
    all_colors.clear();

    auto intersection_points = get_intersection_point(cube_planes);

    all_points.push_back(intersection_points);
    all_colors.emplace_back();
    for (int i = 0; i < 3; i++) {
        const auto &plane = cube_ransac_inliers[i];
        Point3d color{i == 0 ? 255 : 0, i == 1 ? 255 : 0, i == 2 ? 255 : 0};

        for (const auto &p : plane) {
            all_points.push_back(p);
            all_colors.emplace_back(color);
        }

        all_points.push_back(Point3d(cube_planes[i].get_normal()) + intersection_points);
        all_colors.emplace_back(color);
    }

    ::utils::write_ply("inliers_cube.ply", all_points, all_colors);


    refine_planes(cube_ransac_inliers, cube_planes);

//    all_points.clear();
//    all_colors.clear();
    for (int i = 0; i < points.size(); i++) {
        const auto &p = points[i];
        Point3d color{0, 0, 0};

        all_points.push_back(p);
        all_colors.emplace_back(color);
    }

    auto refined_corners = get_corners(cube_ransac_inliers, cube_planes);
    for (int j = 0; j < refined_corners.size(); j++) {
        const auto &p = refined_corners[j];
        auto num_points = ::utils::add_sphere(all_points, p, 0.01);

        for (int i = 0; i < num_points; i++) {
            all_colors.emplace_back(corner_colors[j]);
        }
    }

    ::utils::write_ply("final_cube.ply", all_points, all_colors);
    final_refined_corners = refined_corners;
}

array<int, 3> FindBoxCorners::calc_plane_ordering(const array<vector<Point3d>, 3> &plane_points,
                                                  const array<Plane, 3> &planes,
                                                  array<vector<Point3d>, 3> &outlier_free_points,
                                                  array<Point3d, 7> &corners) {
    array<int, 3> ordering = {0, 1, 2};
    sort(begin(ordering), end(ordering)); //insanity ordering

    array<int, 3> best_ordering{};
    unsigned long max_inliers = 0;

    int counter = 0;
    do {
        array<Plane, 3> act_plane_ordering = {planes[ordering[0]], planes[ordering[1]], planes[ordering[2]]};
        array<vector<Point3d>, 3> act_point_ordering = {plane_points[ordering[0]], plane_points[ordering[1]],
                                                        plane_points[ordering[2]]};

        auto act_corners = get_corners(act_point_ordering, act_plane_ordering);

        vector<Point3d> all_points;
        vector<Point3d> all_colors;

        for (size_t i = 0; i < act_corners.size(); i++) {
            for (double alpha = 0; alpha < M_PI; alpha += 0.1) {
                for (double beta = 0; beta < 2 * M_PI; beta += 0.1) {
                    Point3d p =
                            act_corners[i] + R * Point3d(sin(alpha) * cos(beta), sin(alpha) * sin(beta), cos(alpha));
                    all_points.push_back(p);
                    all_colors.push_back(corner_colors[i]);
                }
            }
        }

        array<vector<Point3d>, 3> act_inliers;
        auto inliers_num = calc_box_inliers(act_point_ordering, act_plane_ordering, act_corners, act_inliers,
                                            all_points, all_colors);

        ::utils::write_ply("corners_" + to_string(counter) + ".ply", all_points, all_colors);

        if (inliers_num > max_inliers) {
            max_inliers = inliers_num;
            best_ordering = ordering;
            corners = act_corners;
            outlier_free_points = act_inliers;

        }

        ++counter;
    } while (next_permutation(begin(ordering), end(ordering)));

    return best_ordering;
}

array<Point3d, 7> FindBoxCorners::get_corners(const array<vector<Point3d>, 3> &points, const array<Plane, 3> &planes) {
    Mat A;
    A.push_back(planes[0].get_normal().t());
    A.push_back(planes[1].get_normal().t());
    A.push_back(planes[2].get_normal().t());

    Mat b;
    b = (Mat_<double>(3, 1) << -planes[0].get_w(), -planes[1].get_w(), -planes[2].get_w());

    Point3d corner = Point3d(Mat(A.inv() * b));


    Mat edge1 = BOX_SIZE._x * planes[0].get_normal().cross(planes[1].get_normal());
    Mat edge2 = BOX_SIZE._y * planes[1].get_normal().cross(planes[2].get_normal());
    Mat edge3 = BOX_SIZE._z * planes[2].get_normal().cross(planes[0].get_normal());

    if (flip_edge(corner, edge1, {points[0], points[1]})) {
        edge1 *= -1;
    }

    if (flip_edge(corner, edge2, {points[1], points[2]})) {
        edge2 *= -1;
    }

    if (flip_edge(corner, edge3, {points[2], points[0]})) {
        edge3 *= -1;
    }

    return {corner,
            corner + Point3d(edge1),
            corner + Point3d(edge2),
            corner + Point3d(edge3),
            corner + Point3d(edge1) + Point3d(edge2),
            corner + Point3d(edge2) + Point3d(edge3),
            corner + Point3d(edge3) + Point3d(edge1)};
}


bool FindBoxCorners::flip_edge(const Point3d &mid_corner,
                               const Mat &edge,
                               const array<vector<Point3d>, 2> &points) {

    double min1 = 10e7;
    double min2 = 10e7;

    Point3d corner1 = mid_corner + Point3d(edge);
    Point3d corner2 = mid_corner - Point3d(edge);

    for (const auto &plane_point : points) {
        for (const auto &point : plane_point) {

            double d1 = norm(point - corner1);
            double d2 = norm(point - corner2);

            if (d1 < min1) {
                min1 = d1;
            }

            if (d2 < min2) {
                min2 = d2;
            }
        }
    }

    return min1 > min2;
}

unsigned long FindBoxCorners::calc_box_inliers(const array<vector<Point3d>, 3> &plane_points,
                                               const array<Plane, 3> &planes,
                                               const array<Point3d, 7> &corners,
                                               std::array<std::vector<cv::Point3d>, 3> &inliers,
                                               std::vector<cv::Point3d> &all_points,
                                               std::vector<cv::Point3d> &all_colors) {
    for (auto &inlier : inliers)
        inlier.clear();

    unsigned long inlier_num = 0;

    for (size_t i = 0; i < plane_points.size(); i++) {

        const auto &plane = planes[i];
        const auto &points = plane_points[i];

        vector<Point3d> corners_to_plane;
        for (const auto &corner : corners) {
            double d = plane.dot((Mat_<double>(1, 4) << corner.x, corner.y, corner.z, 1.0d));
            if (abs(d) < 10e-5) {
                corners_to_plane.push_back(corner);
            }
        }

        if (corners_to_plane.size() != 4) {
            std::cerr << "Error in corners to plane calculation!" << __FILE__ << " : " << __LINE__ << std::endl;
        }

        auto orig_points = transform_to_origin(plane, points);
        auto orig_planes = transform_to_origin(plane, corners_to_plane);

        auto transformed_points = project_to_plane((Mat_<double>(3, 1) << 0, 0, 1.0d), orig_points);
        auto transformed_corners = project_to_plane((Mat_<double>(3, 1) << 0, 0, 1.0d), orig_planes);

        vector<Point2f> transformed_corners_2d;
        transformed_corners_2d.reserve(transformed_corners.size());

        for (const auto &p : transformed_corners) {
            transformed_corners_2d.push_back(Point2f(p.x, p.y));
        }

        vector<Point2f> conv_hull;
        convexHull(Mat(transformed_corners_2d), conv_hull);

        for (const auto &t_corner : transformed_corners) {
            all_points.push_back(t_corner);
            all_colors.push_back(Point3d(255, 0, 0));
        }

        for (size_t j = 0; j < transformed_points.size(); j++) {
            const auto &orig_point = points[j];
            const auto &point = transformed_points[j];

            const auto &point_2d = Point2f(point.x, point.y);

            all_points.push_back(orig_point);
            all_points.push_back(point);
            if (pointPolygonTest(conv_hull, point_2d, false) >= 0) {
                inliers[i].emplace_back(orig_point);
                ++inlier_num;
                all_colors.push_back(Point3d(0, 255, 0));
                all_colors.push_back(Point3d(0, 255, 0));
            } else {
                all_colors.push_back(Point3d(255, 0, 0));
                all_colors.push_back(Point3d(0, 0, 0));
            }
        }
    }

    return inlier_num;
}

vector<Point3d>
FindBoxCorners::transform_to_origin(const Plane &plane, const vector<Point3d> &points) {

    auto normal = plane.get_normal();
    double u = normal.at<double>(0, 0);
    double v = normal.at<double>(1, 0);
    double w = normal.at<double>(2, 0);

    double sqrtuv = sqrt(u * u + v * v);
    Mat Rxy = (Mat_<double>(3, 3) << u / sqrtuv, v / sqrtuv, 0,
            -v / sqrtuv, u / sqrtuv, 0,
            0, 0, 1.0d);

    double sqrtuvw = sqrt(u * u + v * v + w * w);

    Mat Rz;
    Rz = (Mat_<double>(3, 3) << w / sqrtuvw, 0, -sqrtuv / sqrtuvw,
            0, 1, 0,
            sqrtuv / sqrtuvw, 0, w / sqrtuvw);

    Mat R = Rz * Rxy;
    Point3d plane_point = plane.get_plane_point();

    vector<Point3d> transformed_points;
    transformed_points.reserve(points.size());

    for (const auto &p : points) {
        Mat transformed = R * Mat(p - plane_point);
        transformed_points.emplace_back(Point3d(transformed));
    }

    return transformed_points;
}

vector<Point3d> FindBoxCorners::project_to_plane(const Mat &plane_normal, const vector<Point3d> &points) {

    vector<Point3d> projected_points;
    projected_points.reserve(points.size());

    for (const auto &p : points) {
        double dist_to_plane = plane_normal.dot(Mat(p));
        Mat projected = Mat(p) - plane_normal * dist_to_plane;
        projected_points.emplace_back(Point3d(projected));
    }

    return projected_points;
}

array<Plane, 3> FindBoxCorners::fit_cube_ransac(const array<vector<Point3d>, 3> &points,
                                                array<vector<Point3d>, 3> &inliers) {
    array<Plane, 3> best_planes;
    unsigned long max_inliers = 0;

    const auto &first_plane_points = points[0];
    const auto &second_plane_points = points[1];
    const auto &third_plane_points = points[2];

    for (int i = 0; i < RANSAC_FIRST_RUN; ++i) {
        auto randoms1 = ::utils::gen_randoms(static_cast<const unsigned int>(first_plane_points.size()));

        array<Point3d, 3> plane1_points{first_plane_points[randoms1[0]],
                                        first_plane_points[randoms1[1]],
                                        first_plane_points[randoms1[2]]};

        const auto plane1 = ::utils::fit_plane_to_points(plane1_points);
        std::cout << "counter: " << i << " of " << RANSAC_FIRST_RUN <<  std::endl;

        for (int j = 0; j < RANSAC_SECOND_RUN; ++j) {
            auto randoms2 = ::utils::gen_randoms(static_cast<const unsigned int>(second_plane_points.size()));

            Mat v2 = Mat(second_plane_points[randoms2[0]] - second_plane_points[randoms2[1]]);
            Mat second_plane_normal = plane1.get_normal().cross(v2);
            second_plane_normal /= norm(second_plane_normal);

            double w2 = -1 * (second_plane_normal.dot(Mat(second_plane_points[randoms2[0]])));

            Plane plane2(second_plane_normal, static_cast<double>(w2));

            for (int k = 0; k < RANSAC_THIRD_RUN; ++k) {
                auto randoms3 = ::utils::gen_randoms(static_cast<const unsigned int>(third_plane_points.size()));

                Mat third_plane_normal = second_plane_normal.cross(plane1.get_normal());
                third_plane_normal /= norm(third_plane_normal);

                double w3 = -1 * (third_plane_normal.dot(Mat(third_plane_points[randoms3[0]])));

                Plane plane3(third_plane_normal, static_cast<double>(w3));

                array<Plane, 3> planes{plane1, plane2, plane3};
                auto act_inliers = inliers_to_box(points, planes);

                auto inliers_num = act_inliers[0].size() + act_inliers[1].size() + act_inliers[2].size();
                if (inliers_num > max_inliers) {
                    max_inliers = inliers_num;
                    best_planes = planes;
                    inliers = act_inliers;
                }
            } //end third plane search
        } //end of second plane search
    }


    //make sure normals point outwards of the box
    auto intersection_points = get_intersection_point(best_planes);

    for (int i = 0; i < 3; i++) {
        auto &plane = best_planes[i];

        auto p1 = intersection_points + Point3d(plane.get_normal());
        double err1 = 0;
        for (int j = 1; j < 3; ++j) {
            for (const auto &p : inliers[(i + j) % 3]) {
                err1 += norm(p - p1);
            }
        }

        auto p2 = intersection_points - Point3d(plane.get_normal());
        double err2 = 0;
        for (int j = 1; j < 3; ++j) {
            for (const auto &p : inliers[(i + j) % 3]) {
                err2 += norm(p - p2);
            }
        }

        if (err1 < err2) {
            plane.flip_normal();
        }
    }

    return best_planes;
}

array<vector<Point3d>, 3> FindBoxCorners::inliers_to_box(const array<vector<Point3d>, 3> &points,
                                                         const array<Plane, 3> &planes) {
    array<vector<Point3d>, 3> inliers;

    for (int i = 0; i < 3; ++i) {
        const auto &plane = planes[i];

        for (const auto &point : points[i]) {
            double d = plane.dot((Mat_<double>(1, 4) << point.x, point.y, point.z, 1.0d));
            if (abs(d) < RANSAC_PLANE_THRESHOLD) {
                inliers[i].emplace_back(point);
            }
        }
    }

    return inliers;
}

array<vector<Point3d>, 2> rotate_points(const double angle, const array<vector<Point3d>, 2> &points) {

    double cos_angle = cos(angle);
    double sin_angle = sin(angle);

    Mat rotation_matrix_Z = (Mat_<double>(3, 3) << cos_angle, -sin_angle, 0,
            sin_angle, cos_angle, 0,
            0, 0, 1.0d);

    array<vector<Point3d>, 2> rotated_points;
    for (int i = 0; i < 2; i++) {
        for (const auto &p : points[i]) {
            Mat rotated = rotation_matrix_Z * Mat(p);
            rotated_points[i].push_back(Point3d(rotated));
        }
    }

    return rotated_points;
}

double calc_plane_error(const array<vector<Point3d>, 3> &points, const array<Plane, 3> &planes) {
    double error = 0.0d;

    for (int i = 0; i < 3; ++i) {
        const auto &plane = planes[i];

        for (const auto &point : points[i]) {
            double d = plane.dot((Mat_<double>(1, 4) << point.x, point.y, point.z, 1.0d));
            error += abs(d);
        }
    }

    return error;
}

void FindBoxCorners::refine_planes(const array<vector<Point3d>, 3> &points, array<Plane, 3> &planes) {

    double last_error, error = 10e6;

    double min_err = 10e6;

    int counter = 0;
    do {
        last_error = error;

        for (int i = 0; i < 3; i++) {
            auto &plane1 = planes[i];
            auto &plane2 = planes[(i + 1) % 3];
            auto &plane3 = planes[(i + 2) % 3];

            const auto &points1 = points[i];
            const auto &points2 = points[(i + 1) % 3];
            const auto &points3 = points[(i + 2) % 3];

            array<vector<Point3d>, 3> transformed;
            auto p = transform_to_match_Z2({plane1, plane2, plane3}, {points1, points2, points3}, transformed);

            auto angle = get_rotation_angle(transformed[0], transformed[1]);

            auto rotated_plane1 = rotate_plane(angle, p, plane1);
            auto rotated_plane2 = rotate_plane(angle, p, plane2);

            plane1 = rotated_plane1;
            plane2 = rotated_plane2;


            error = calc_plane_error(points, planes);
        }

        auto translation = get_optimal_translation(planes, points);
        for (auto &plane : planes) {
            plane.translate(translation);
        }

        error = calc_plane_error(points, planes);

        if (error < min_err)
            min_err = error;

        ++counter;
    } while (abs(last_error - error) > REFINE_ITERATION_THRESHOLD);

}

double FindBoxCorners::get_rotation_angle(const std::vector<cv::Point3d> &points1, const vector<Point3d> &points2) {
    cv::Mat A(static_cast<int>(points1.size() + points2.size()), 2, CV_64FC1);
    int row_index = 0;
    for (const auto &p : points1) {
        A.at<double>(row_index, 0) = p.x;
        A.at<double>(row_index, 1) = -p.y;
        ++row_index;
    }

    for (const auto &p : points2) {
        A.at<double>(row_index, 0) = p.y;
        A.at<double>(row_index, 1) = p.x;
        ++row_index;
    }

    cv::Mat m = A.t() * A;
    cv::Mat eigen_values, eigen_vectors;
    cv::eigen(m, eigen_values, eigen_vectors);

    int min_eigen_value_row_index = 0;
    if (eigen_values.at<double>(0, 0) > eigen_values.at<double>(1, 0)) {
        min_eigen_value_row_index = 1;
    }

    double c = eigen_vectors.at<double>(min_eigen_value_row_index, 0);
    double s = eigen_vectors.at<double>(min_eigen_value_row_index, 1);

    return atan2(c, s);
}

Plane FindBoxCorners::rotate_plane(const double angle, const pair<Mat, Point3d> &transform, const Plane &plane) {
    auto cos_angle = cos(angle);
    auto sin_angle = sin(angle);


    Mat normal_orig = plane.get_normal();
    Mat transformed_normal = transform.first * plane.get_normal();
    Mat transformed_point = transform.first * Mat(plane.get_plane_point() - transform.second);


    Mat rotation_matrix_Z = (Mat_<double>(3, 3) << cos_angle, -sin_angle, 0,
            sin_angle, cos_angle, 0,
            0, 0, 1.0d);

    Mat transformed2_normal = rotation_matrix_Z * transformed_normal;
    Mat transformed2_point = rotation_matrix_Z * transformed_point;

    Mat inverted_r = transform.first.inv(0);
    Mat transformed3_normal = inverted_r * transformed2_normal;
    transformed3_normal /= norm(transformed3_normal);
    Mat transformed3_point = (inverted_r * transformed2_point) + Mat(transform.second);

    double d = -1 * transformed3_normal.dot(transformed3_point);

    Plane p(transformed3_normal, d);
    return p;
}

pair<Mat, Point3d> FindBoxCorners::transform_to_match_Z2(const array<Plane, 3> &planes,
                                                         const array<vector<Point3d>, 3> &points,
                                                         array<vector<Point3d>, 3> &transformed) {
    transformed[0].clear();
    transformed[1].clear();
    transformed[2].clear();

    Mat R;
    for (int i = 0; i < 3; ++i) {
        R.push_back(-1 * planes[i].get_normal().t());
    }

    bool fliped_det = false;
    auto det1 = determinant(R);
    if (determinant(R) < 0) { //determinant == -1
        fliped_det = true;
        auto n0 = planes[0].get_normal();
        auto n1 = planes[1].get_normal();
        auto n2 = planes[2].get_normal();

        for (int i = 0; i < 3; ++i)
            R.at<double>(0, i) = -n1.at<double>(i, 0);

        for (int i = 0; i < 3; ++i)
            R.at<double>(1, i) = -n0.at<double>(i, 0);

        for (int i = 0; i < 3; ++i)
            R.at<double>(2, i) = -n2.at<double>(i, 0);
    }
    auto det2 = determinant(R);

    Point3d intersection_point = get_intersection_point(planes);

    for (int i = 0; i < 3; i++) {
        const auto &plane_points = points[i];

        for (const auto &p : plane_points) {
            Mat t_p = R * Mat(p - intersection_point);
            transformed[i].emplace_back(Point3d(t_p));
        }
    }

    return {R.clone(), intersection_point};
}

Point3d FindBoxCorners::get_intersection_point(const array<Plane, 3> &planes) const {

    auto n0 = planes[0].get_normal();
    auto n1 = planes[1].get_normal();
    auto n2 = planes[2].get_normal();

    double denom = n0.cross(n1).dot(n2);

    Mat intersect = (n1.cross(n2) * -planes[0].get_w() +
                     n2.cross(n0) * -planes[1].get_w() +
                     n0.cross(n1) * -planes[2].get_w()) / denom;

    return Point3d(intersect);
}

Mat FindBoxCorners::get_optimal_translation(const array<Plane, 3> &planes, const array<vector<Point3d>, 3> &points) {

    const auto total_points = static_cast<const int>(points[0].size() + points[1].size() + points[2].size());

    Mat B(total_points, 3, CV_64FC1);
    Mat c(total_points, 1, CV_64FC1);

    const array plane_points = {planes[0].get_plane_point(),
                                planes[1].get_plane_point(),
                                planes[2].get_plane_point()};

    int current_row = 0;
    for (int i = 0; i < 3; ++i) {
        const Mat plane_normal = planes[i].get_normal();

        for (size_t j = 0; j < points[i].size(); ++j) {
            B.at<double>(current_row, 0) = plane_normal.at<double>(0, 0);
            B.at<double>(current_row, 1) = plane_normal.at<double>(1, 0);
            B.at<double>(current_row, 2) = plane_normal.at<double>(2, 0);

            c.at<double>(current_row, 0) = plane_normal.dot(Mat(points[i][j] - plane_points[i]));

            ++current_row;
        }
    }

    return (B.t() * B).inv() * B.t() * c;
}
