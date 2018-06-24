//
// Created by swinkx on 09/06/17.
//

#ifndef POLYGONALBOARD_SELECT_IMAGE_CORNERS_H
#define POLYGONALBOARD_SELECT_IMAGE_CORNERS_H

#include <vector>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc.hpp>

class SelectImageCorners {
public:
    static std::vector<cv::Point2d> select_image_points(const std::string &IMG_FILE_NAME);

    static const cv::Scalar colors[7];
private:
    static void onMouse(int evt, int x, int y, int flags, void *param);

private:
    static const std::string WINDOW_NAME;
    static std::vector<cv::Point2d> image_points;
    static cv::Mat img2;
};


#endif //POLYGONALBOARD_SELECT_IMAGE_CORNERS_H
