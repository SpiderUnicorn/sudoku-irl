#include "pre_process.h"
#include "digit_recognizer.h"
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// do better when awake
int SZ = 28; // was 20
float affineFlags = WARP_INVERSE_MAP|INTER_LINEAR;

vector<Point> get_largest_contour(InputOutputArray src) {
    vector<vector<Point>> contours;
    findContours(src, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

    // Find the largest rectangle in the picture
    int largest_contour_index;
    double maxArea;

    for (uint i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            largest_contour_index = i;
        }
    }

    return contours[largest_contour_index];
}

Mat deskew2(Mat& img){
    Moments m = moments(img);
    if(abs(m.mu02) < 1e-2){
        return img.clone();
    }
    float skew = m.mu11/m.mu02;
    Mat warpMat = (Mat_<float>(2,3) << 1, skew, -0.5*SZ*skew, 0, 1, 0);
    Mat imgOut = Mat::zeros(img.rows, img.cols, img.type());
    warpAffine(img, imgOut, warpMat, imgOut.size(),affineFlags);

    return imgOut;
}


int main(int argc, char** argv) {
    const String image_path = argv[1];

    Mat src = imread(image_path, CV_LOAD_IMAGE_UNCHANGED);
    imshow("Source", src);

    Mat thresholded;
    pre_process(src, thresholded);

    imshow("Thresholded", thresholded);

    vector<Point> contour = get_largest_contour(thresholded);

    vector<vector<Point> > contours_poly(1);
    approxPolyDP(Mat(contour), contours_poly[0],5, true);

    // corners of the game area
    Point2f corners[4];
    // corners of the image
    Point2f cornersDest[4];
    cornersDest[0] = Point2f(500, 0);
    cornersDest[1] = Point2f(0, 0);
    cornersDest[2] = Point2f(0, 500);
    cornersDest[3] = Point2f(500, 500);

    for (int i = 0; i < 4; i++) {
        // cout << contours_poly[0][i] << "," << contours_poly[0][(i+1)%4] << endl;
        corners[i] = Point2f(contours_poly[0][i].x, contours_poly[0][i].y);
    }

    // Mat drawing = Mat::zeros( thresholded.size(), CV_8UC3 );
    /*
    for( int j = 0; j < 4; j++ )
        line(drawing, contours_poly[0][j], contours_poly[0][(j+1)%4], Scalar(255, 255, 255), 1, 8 );
    */

    Mat undistorted = Mat(Size(500, 500), CV_8UC1);
    warpPerspective(
        src,
        undistorted,
        getPerspectiveTransform(corners, cornersDest),
        Size(500, 500)
    );

    Mat undistortedThreshed = undistorted.clone();

    pre_process(undistorted, undistortedThreshed);

    imshow("undistortedThreshed", undistortedThreshed);

    DigitRecognizer *dr = new DigitRecognizer();

    // into the unknown...
    dr->train();

	std::cout << "done training... phew" << endl;

    int dist = ceil((double)500/9);
    Mat currentCell = Mat(dist, dist, CV_8UC1);

    for(int j=0;j<9;j++)
    {
        for(int i=0;i<9;i++)
        {
            for(int y=0;y<dist && j*dist+y<undistortedThreshed.cols;y++)
            {

                uchar* ptr = currentCell.ptr(y);

                for(int x=0;x<dist && i*dist+x<undistortedThreshed.rows;x++)
                {
                    ptr[x] = undistortedThreshed.at<uchar>(j*dist+y, i*dist+x);
                }
            }

            Moments m = cv::moments(currentCell, true);
            int area = m.m00;
            if(area > currentCell.rows*currentCell.cols/5)
            {
                int number = dr->classify(currentCell);
                String hest = to_string(j) + to_string(i) + to_string(number);
                imshow(hest, currentCell);
                cout << number << " ";
                waitKey(3000);
            }
            else
            {
            }
        }
    }
    cout << endl;

    waitKey(0);
    cout << "done" << endl;
    return 0;
}