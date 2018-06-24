#include "findboxplanes.h"
#include <iostream>
#include <math.h>

using namespace std;

vector<cv::Point3d> utils::read_xyz(const string &fn) {
    ifstream file(fn);

    double x, y, z;
    vector<cv::Point3d> points;
    while (file >> x >> y >> z) {
        points.emplace_back(x, y, z);
    }

    file.close();
    return points;
}

void utils::write_ply(const string &fn,
                      const vector<cv::Point3d> &points,
                      const vector<cv::Point3d> &colors,
                      const vector<cv::Point3d> &normals) {

    if (!colors.empty() && points.size() != colors.size()) {
        std::cerr << "error (colors) writing PLY file: " << fn << std::endl;
    }

    if (!normals.empty() && points.size() != normals.size()) {
        std::cerr << "error (normals) writing PLY file: " << fn << std::endl;
    }

    ofstream file(fn);
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " <<  to_string(points.size()) << "\n";

    file << "property float x\nproperty float y\nproperty float z\n";

    if (!colors.empty())
        file << "property uchar red\nproperty uchar green\nproperty uchar blue\n";

    if (!normals.empty())
        file << "property float nx\nproperty float ny\nproperty float nz\n";

    file << "end_header\n";

    for (size_t i = 0; i < points.size(); i++) {
        const auto &pos = points[i];
        file << pos.x << " " << pos.y << " " << pos.z;

        if (!colors.empty()) {
            const auto &color = colors[i];
            file << " " << to_string((int)color.x) << " " << to_string((int)color.y) << " " << to_string((int)color.z);
        }


        if (!normals.empty()) {
            const auto &normal = normals[i];
            file << " " << to_string((int)normal.x) << " " << to_string((int)normal.y) << " " << to_string((int)normal.z);
        }

        file << "\n";
    }

    file.close();
}

array<int, 3> utils::gen_randoms(const unsigned int max_rand) {

    random_device rd;
    mt19937 gen(rd());

    uniform_int_distribution<> dis1(0, max_rand);
    uniform_int_distribution<> dis2(0, max_rand - 1);
    uniform_int_distribution<> dis3(0, max_rand - 2);

    int a = dis1(gen);
    int b = dis2(gen);
    int c = dis3(gen);

    if (b >= a) b++;
    if (c >= min(a, b)) c++;
    if (c >= max(a, b)) c++;

    array<int, 3> randoms{a, b, c};
    sort(begin(randoms), end(randoms));

    return randoms;
}

Plane utils::fit_plane_to_points(const array<cv::Point3d, 3> &plane_points) {

    cv::Mat v1 = cv::Mat(plane_points[1]) - cv::Mat(plane_points[0]);
    cv::Mat v2 = cv::Mat(plane_points[2]) - cv::Mat(plane_points[0]);

    cv::Mat normal = v1.cross(v2);
    normal /= norm(normal);

    return {normal, plane_points[0]};
}

int utils::add_sphere(std::vector<cv::Point3d> &points, const cv::Point3d center, const double r) {

    int num_points = 0;
    for (float alpha = 0; alpha < 2 * M_PI; alpha += 0.1f) {

        for (float beta = 0; beta < M_PI; beta += 0.1f) {

            points.emplace_back(center.x + r * cos(alpha) * sin(beta),
                                center.y + r * sin(alpha) * sin(beta),
                                center.z + r * cos(beta));
            ++num_points;
        }
    }

    return num_points;
}