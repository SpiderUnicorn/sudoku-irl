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

}

#endif