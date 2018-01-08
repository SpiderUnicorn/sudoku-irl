#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "image_processing.h"

using namespace cv;
using namespace std;
using namespace sudoku;

const int AFFINE_FLAGS = WARP_INVERSE_MAP | INTER_LINEAR;

namespace sudoku {

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
	for (auto digit: digits)
	{
        deskewedDigits.push_back(deskew(digit));
	}
}

using LineTestFn = function<bool(Rect&, Mat&)>;
using ExpandRectFn = function<Rect(Rect&, Mat&)>;

/**
* Detect horizontal/vertical lines of a sudoku grid in img Mat and copy expanded lines to dst Mat
*/
void extract_lines(const Mat& img, Mat& dst, LineType lineType)
{
    // Clone the source image
    Mat clone = img.clone();

    LineTestFn lineTest;
    ExpandRectFn expandRect;
    Size size;

    // Setup correct structure size, line test and rect expansion for horizontal vs. vertical
    if (lineType == HORIZONTAL) {
        size = Size(img.cols / 9, 1);
        lineTest = [](Rect& rect, Mat& mat) { return rect.height / double(mat.rows) < 0.05 && rect.width / double(mat.cols) > 0.111; };
        expandRect = [](Rect& rect, Mat& mat) {
            Rect expanded = rect;
            if (expanded.y > 1) { expanded.y -= 2; }

            if (expanded.y + expanded.height < mat.rows) {
                expanded.height += min(4, mat.rows - expanded.y - expanded.height);
            }
            expanded.x = 0;
            expanded.width = mat.cols;
            return expanded;
        };
    } else {
        size = Size(1, img.rows / 9);
        lineTest = [](Rect& rect, Mat& mat) { return rect.width / double(mat.cols) < 0.05 && rect.height / double(mat.rows) > 0.111; };
        expandRect = [](Rect& rect, Mat& mat) {
            Rect expanded = rect;
            if (expanded.x > 1) { expanded.x -= 2; }

            if (expanded.x + expanded.width < mat.cols) {
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
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours( clone, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect( contours.size() );

    // Mark contours which pass line test in the destination image
    for( size_t i = 0; i < contours.size(); i++ )
    {
        approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );
        if (lineTest(boundRect[i], clone)) {
            Rect expanded = expandRect(boundRect[i], clone);
            dst(expanded) |= 255; // set the expanded rect to white
        }
    }
}

}
