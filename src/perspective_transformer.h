#ifndef SUDOKU_PERSPECTIVE_TRANSFORMER
#define SUDOKU_PERSPECTIVE_TRANSFORMER

#include <stdio.h>
#include <opencv2/opencv.hpp>

namespace sudoku
{

class PerspectiveTransformer
{
    public:
        PerspectiveTransformer(cv::Mat img)
        : original(img)
        {}

        cv::Mat extract_straightened_board(int size);
        cv::Mat project_onto_unstraightened(cv::Mat projection);
        // unextract();

    private:
        cv::Mat original;
        cv::Point2f corners[4];
        cv::Point2f destinationCorners[4];
};

}


#endif