#include "image_processing.h"
#include "pre_process.h"
#include "perspective_transformer.h"

using namespace cv;
using namespace std;

namespace sudoku
{

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

    /*
    Mat clone = Mat::zeros(src.size(), src.type());
    drawContours(clone, contours, largest_contour_index, Scalar(255, 255, 255));
    imshow("contours", clone);
    */

    return contours[largest_contour_index];
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

Mat PerspectiveTransformer::extract_straightened_board(int size)
{
    Point2f corners[4];
    get_largest_contour_corners(original, corners);


    Point2f destinationCorners[4];
    int * orderedIndex = order_corners(corners);

    destinationCorners[0[orderedIndex]] = Point2f(0, 0);
    destinationCorners[1[orderedIndex]] = Point2f(0, size);
    destinationCorners[2[orderedIndex]] = Point2f(size, size);
    destinationCorners[3[orderedIndex]] = Point2f(size, 0);

    delete orderedIndex;

    for (int i = 0; i < 4; i++) {
        cout << destinationCorners[i] << endl;
    }

    cout << "-----" << endl;

    for (int i = 0; i < 4; i++) {
        cout << corners[i].x << " " << corners[i].y << endl;
    }

    Mat transform = getPerspectiveTransform(corners, destinationCorners);

    Mat straightened = Mat(Size(size, size), CV_8UC1);
    warpPerspective(
        original,
        straightened,
        transform,
        straightened.size());

    return straightened.clone();
}


Mat PerspectiveTransformer::project_onto_unstraightened(Mat projection)
{
    Mat bSubmat = submat(
        projection.rows(),
        projection.rows()*2,
        projection.cols(),
        projection.cols()*2);
    projection.copyTo(bSubmat);

    imshow("Overlay Image", original);

    return original;
}

}