#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "digit_classifier.h"
#include "image_processing.h"

using namespace cv;
using namespace cv::ml;
using namespace std;

const int PIXELS_IN_SIDE = 20;

namespace sudoku {

Mat DigitClassifier::preprocessImage(Mat img)
{
	imshow("noroded", img);
	Mat kernel = getStructuringElement(MORPH_CROSS, Size(2, 2));

	erode(img, img, kernel, Point(-1, -1));
	imshow("eroded", img);

	Mat centeredDigit;
	center_digit(img, centeredDigit);

	Mat cloneImg = Mat(PIXELS_IN_SIDE, PIXELS_IN_SIDE, CV_32FC1);
	resize(centeredDigit, cloneImg, Size(PIXELS_IN_SIDE, PIXELS_IN_SIDE));

	return cloneImg;
}

int DigitClassifier::classify(Mat img, HOGDescriptor hog)
{
	// img.convertTo(img, CV_8UC1);
	vector<float> descriptors;
	hog.compute(img, descriptors);

	Mat descriptorMat(1, descriptors.size(), CV_32FC1);

	// convert descriptors to a 1 x cols mat
	for (int col = 0; col < descriptors.size(); col++)
	{
		descriptorMat.at<float>(0, col) = descriptors[col];
	}

	Mat1f test(descriptors.size(), CV_32FC1, descriptors.data());

	int prediction = svm->predict(descriptorMat);

	return prediction;
}

}
