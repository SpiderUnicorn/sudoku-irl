#ifndef SUDOKU_HOG_PROCESSING
#define SUDOKU_HOG_PROCESSING

#include <stdio.h>
#include <opencv2/opencv.hpp>

namespace sudoku {
    cv::Ptr<cv::ml::TrainData> get_train_data(cv::HOGDescriptor hog, cv::Mat img, const int SIDE_LENGTH);
}

#endif