#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "image_processing.h"
#include "pre_process.h"
#include "constants.h"

using namespace cv;
using namespace std;
using namespace sudoku;

const int AFFINE_FLAGS = WARP_INVERSE_MAP | INTER_LINEAR;

namespace sudoku
{

// understand this better
Mat deskew(Mat &img)
{
    const int WIDTH = img.size().width;
    Moments m = moments(img);

    // if what?
    if (abs(m.mu02) < 0.01)
    {
        return img.clone();
    }

    double skew = m.mu11 / m.mu02;
    Mat warpMat = (Mat_<float>(2, 3) << 1, skew, -0.5 * WIDTH * skew, 0, 1, 0);
    Mat imgOut = Mat::zeros(img.rows, img.cols, img.type());

    warpAffine(img, imgOut, warpMat, imgOut.size(), AFFINE_FLAGS);

    return imgOut;
}

void deskew_all(vector<Mat> &deskewedDigits, vector<Mat> &digits)
{
    for (auto digit : digits)
    {
        deskewedDigits.push_back(deskew(digit));
    }
}

using LineTestFn = function<bool(Rect &, Mat &)>;
using ExpandRectFn = function<Rect(Rect &, Mat &)>;

/**
* Detect horizontal/vertical lines of a sudoku grid in img Mat and copy expanded lines to dst Mat
*/
void extract_lines(const Mat &img, Mat &dst, LineType lineType)
{
    // Clone the source image
    Mat clone = img.clone();

    LineTestFn lineTest;
    ExpandRectFn expandRect;
    Size size;

    // Setup correct structure size, line test and rect expansion for horizontal vs. vertical
    if (lineType == HORIZONTAL)
    {
        size = Size(img.cols / 9, 1);
        lineTest = [](Rect &rect, Mat &mat) { return rect.height / double(mat.rows) < 0.05 && rect.width / double(mat.cols) > 0.111; };
        expandRect = [](Rect &rect, Mat &mat) {
            Rect expanded = rect;
            if (expanded.y > 1)
            {
                expanded.y -= 2;
            }

            if (expanded.y + expanded.height < mat.rows)
            {
                expanded.height += min(4, mat.rows - expanded.y - expanded.height);
            }
            expanded.x = 0;
            expanded.width = mat.cols;
            return expanded;
        };
    }
    else
    {
        size = Size(1, img.rows / 9);
        lineTest = [](Rect &rect, Mat &mat) { return rect.width / double(mat.cols) < 0.05 && rect.height / double(mat.rows) > 0.111; };
        expandRect = [](Rect &rect, Mat &mat) {
            Rect expanded = rect;
            if (expanded.x > 1)
            {
                expanded.x -= 2;
            }

            if (expanded.x + expanded.width < mat.cols)
            {
                expanded.width += min(4, mat.cols - expanded.x - expanded.width);
            }
            expanded.y = 0;
            expanded.height = mat.rows;
            return expanded;
        };
    }
    // Create structure element for extracting lines through morphology operations
    Mat structure = getStructuringElement(MORPH_RECT, size);

    // Apply morphology operations
    erode(clone, clone, structure, Point(-1, -1));
    dilate(clone, clone, structure, Point(-1, -1));

    // Find all contours
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(clone, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
    vector<vector<Point>> contours_poly(contours.size());
    vector<Rect> boundRect(contours.size());

    // Mark contours which pass line test in the destination image
    for (size_t i = 0; i < contours.size(); i++)
    {
        approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
        boundRect[i] = boundingRect(Mat(contours_poly[i]));
        if (lineTest(boundRect[i], clone))
        {
            Rect expanded = expandRect(boundRect[i], clone);
            dst(expanded) |= 255; // set the expanded rect to white
        }
    }
}

vector<Point> get_largest_contour(InputOutputArray src)
{
    vector<vector<Point>> contours;
    findContours(src, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

    // Find the largest rectangle in the picture
    int largest_contour_index;
    double maxArea;

    for (uint i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        if (area > maxArea)
        {
            maxArea = area;
            largest_contour_index = i;
        }
    }

    return contours[largest_contour_index];
}
int i = 0;
void center_digit(Mat &src, Mat &outbound)
{
    // imshow("before center" + to_string(++i), src);
    // paint frame black
    // rectangle(src, Point(0, 0), Point(src.rows - 1, src.cols - 1), Scalar(0), 5);

    // get the contours
    vector<vector<Point>> contours;
    findContours(src, contours, RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    // return if there is nothing to center
    if (contours.empty())
    {
        src.copyTo(outbound);
        // cout << "image with no digit in center digit" << endl;
        return;
    }

    // get the largest contour
    vector<Point> max_contour = contours[0];
    double max_area = 0;
    for (auto contour : contours)
    {
        double area = contourArea(contour);
        if (area > max_area)
        {
            max_area = area;
            max_contour = contour;
        }
    }

    vector<vector<Point>> contours_poly(1);
    approxPolyDP(Mat(max_contour), contours_poly[0], 5, true);
    Rect boundRect = boundingRect(Mat(contours_poly[0]));

    Point image_center(src.cols / 2, src.rows / 2);

    // center the largest contour
    Mat roi = Mat(src, boundRect);
    Mat centered = Mat::zeros(src.cols, src.rows, src.type());
    Rect centerBox(image_center.x - boundRect.width / 2, image_center.y - boundRect.height / 2, boundRect.width, boundRect.height);

    // paste the cropped contour into a new image
    roi.copyTo(centered(Rect(centerBox)));
    // imshow("centered" + to_string(++i), centered);

    centered.copyTo(outbound);
}

void get_largest_contour_corners(Mat img, Point2f corners[4])
{
    Mat thresholded;
    pre_process(img, thresholded);

    vector<Point> contour = get_largest_contour(thresholded);

    // approximate lines around the countour to get the corner coordinates
    vector<vector<Point>> contours_poly(1);
    approxPolyDP(Mat(contour), contours_poly[0], 5, true);

    // corners of the game board
    for (int i = 0; i < 4; i++)
    {
        corners[i] = Point2f(contours_poly[0][i].x, contours_poly[0][i].y);
    }
}

int* order_corners(Point2f corners[4])
{
    // get the index of corner theclosest to the upper left corner
    // of the image by finding the smallest hypotenuse (cSquared)
    int min = 99999999;
    int minIndex = 0;
    for (int i = 0; i < 4; i++) {
        auto corner = corners[i];
        auto cSquared = pow(corner.x, 2) + pow(corner.y, 2);
        if (cSquared < min) {
            min = cSquared;
            minIndex = i;
        }
    }

    int *indexArray = new int[4];
    for (int i = minIndex; i < minIndex + 4; i++) {
        *(indexArray++) = i % 4;
    }
    // rewind pointer
    indexArray -= 4;

    return indexArray;
}

Mat extract_straightened_board(Mat img, int size = BOARD_SIZE)
{
    Point2f corners[4];
    get_largest_contour_corners(img, corners);

    Point2f destinationCorners[4];
    int * orderedIndex = order_corners(corners);

    destinationCorners[0[orderedIndex]] = Point2f(0, 0);
    destinationCorners[1[orderedIndex]] = Point2f(0, size);
    destinationCorners[2[orderedIndex]] = Point2f(size, size);
    destinationCorners[3[orderedIndex]] = Point2f(size, 0);

    Mat straightened = Mat(Size(size, size), CV_8UC1);
    warpPerspective(
        img,
        straightened,
        getPerspectiveTransform(corners, destinationCorners),
        Size(size, size));

    return straightened.clone();
}

}
