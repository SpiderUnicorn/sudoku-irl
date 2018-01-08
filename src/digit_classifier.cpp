#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "digit_classifier.h"
#include "image_processing.h"

using namespace cv;
using namespace cv::ml;
using namespace std;

#define PIXELS_IN_SIDE 20
#define PIXELS_IN_IMAGE PIXELS_IN_SIDE *PIXELS_IN_SIDE
#define ENABLE_TRAIN 1

string pathName = "digits.png";
float affineFlags_d = WARP_INVERSE_MAP | INTER_LINEAR;

namespace sudoku {

void center_digit(Mat &src, Mat &outbound)
{
	// paint frame black
	rectangle(src, Point(0, 0), Point(src.rows - 1, src.cols - 1), Scalar(0), 5);

	// get the contours
	vector<vector<Point>> contours;
	findContours(src, contours, RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	// return if there is nothing to center
	if (contours.empty())
	{
		src.copyTo(outbound);
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

	centered.copyTo(outbound);
}

Mat DigitClassifier::preprocessImage(Mat img)
{
	Mat centeredDigit;
	center_digit(img, centeredDigit);

	Mat cloneImg = Mat(PIXELS_IN_SIDE, PIXELS_IN_SIDE, CV_32FC1);
	resize(centeredDigit, cloneImg, Size(PIXELS_IN_SIDE, PIXELS_IN_SIDE));

	return cloneImg;
}

int DigitClassifier::classify(Mat img, HOGDescriptor hog)
{
	vector<float> descriptors;
	hog.compute(img, descriptors);

	Mat descriptorMat(1, descriptors.size(), CV_32FC1);

	// convert descriptors to a 1 x cols mat
	for (int col = 0; col < descriptors.size(); col++)
	{
		descriptorMat.at<float>(0, col) = descriptors[col];
	}

	/*
	Mat p = Mat(PIXELS_IN_SIDE * 4, PIXELS_IN_SIDE * 4, CV_8UC1);
	String s = "to predict" + to_string(rand() * 100);
	resize(img, p, Size(PIXELS_IN_SIDE * 4, PIXELS_IN_SIDE * 4));
	imshow(s, p);
	*/
	cout << descriptorMat.size().width << endl;
	int prediction = svm->predict(descriptorMat);

	return prediction;
}

}
