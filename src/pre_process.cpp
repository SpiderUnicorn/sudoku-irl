#include "pre_process.h"
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Illuminate pixels above threshold to max
const double THRESHOLD_MAX_VALUE = 255;
// A smaller value preserves less detail
const int THRESHOLD_BLOCK_SIZE = 7;
// Constant subtracted from the mean or weighted mean
// Removes noise from the threshold. A high value removes detail
const int THRESHOLD_CONSTANT = 2;

/**
 * Grayscaling, smoothing, thresholding
 */
void pre_process(InputArray src, OutputArray dest) {
    // remove rubbish color information
    cvtColor(src, dest, COLOR_BGR2GRAY);

    // smooth out noise in the image before thresholding
    GaussianBlur(dest, dest, Size(11,11), 0);

    // apply threshold
    adaptiveThreshold(
        dest,
        dest,
        THRESHOLD_MAX_VALUE,
        ADAPTIVE_THRESH_MEAN_C,
        // invert image
        THRESH_BINARY_INV,
        THRESHOLD_BLOCK_SIZE,
        THRESHOLD_CONSTANT
    );
}
