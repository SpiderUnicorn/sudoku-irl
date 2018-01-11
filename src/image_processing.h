#ifndef SUDOKU_IMAGE_PROCESSING
#define SUDOKU_IMAGE_PROCESSING

#include <stdio.h>
#include <opencv2/opencv.hpp>

namespace sudoku {
    enum LineType {
        HORIZONTAL,
        VERTICAL
    };

    cv::Mat deskew(cv::Mat &img, const int SIDE_LENGTH);
    void deskew_all(std::vector<cv::Mat> &deskewedDigits, std::vector<cv::Mat> &digits);
    void extract_lines(const cv::Mat& img, cv::Mat& dst, LineType lineType);
    std::vector<cv::Point> get_largest_contour(cv::InputOutputArray src);
    void center_digit(cv::Mat &src, cv::Mat &outbound);
    // cv::Point2f* get_largest_contour_corners(cv::Mat img);
    // cv::Mat extract_straightened_board(cv::Mat img, int size);
}

#endif