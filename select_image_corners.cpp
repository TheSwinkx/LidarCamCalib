//
// Created by swinkx on 09/06/17.
//

#include "select_image_corners.h"

using namespace std;
using namespace cv;

//init static members;
const cv::Scalar SelectImageCorners::colors[7] = {cv::Scalar(0, 0, 0),
                                                  cv::Scalar(255, 0, 0),
                                                  cv::Scalar(0, 255, 0),
                                                  cv::Scalar(0, 0, 255),
                                                  cv::Scalar(255, 255, 0),
                                                  cv::Scalar(0, 255, 255),
                                                  cv::Scalar(255, 0, 255)};

const std::string SelectImageCorners::WINDOW_NAME = "asd";
std::vector<cv::Point2d> SelectImageCorners::image_points;
cv::Mat SelectImageCorners::img2;

void SelectImageCorners::onMouse(int evt, int x, int y, int flags, void *param) {

    if (evt == CV_EVENT_LBUTTONDOWN) {
        std::cout << "Selecting " << x << " : " << y << std::endl;
        Point2d click = Point2d(x, y);

        image_points.push_back(click);
        Scalar color = colors[(image_points.size() - 1) % 4];
        circle(img2, click, 10, color, 5);

        cv::imshow(WINDOW_NAME, img2);
    }
}

std::vector<cv::Point2d> SelectImageCorners::select_image_points(const std::string &IMG_FILE_NAME) {

    image_points.clear();

    img2 = cv::imread(IMG_FILE_NAME, cv::IMREAD_GRAYSCALE);

    cv::namedWindow(WINDOW_NAME, WINDOW_NORMAL);

    imshow(WINDOW_NAME, img2);
    cv::setMouseCallback(WINDOW_NAME, onMouse);
    int key = 0; // Wait for a keystroke in the window
    while (key != 27) {
        key = waitKey(0);
        if (key == 'z' || key == 'Z') {
            image_points.pop_back();
        }
    }

    return image_points;
}