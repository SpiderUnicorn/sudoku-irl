#include <cv.h>
#include <opencv2/opencv.hpp>
#include <highgui.h>
#include <ml.h>

using namespace cv;

#define MAX_NUM_IMAGES	60000

namespace sudoku {

class DigitClassifier
{
public:
	DigitClassifier(cv::Ptr<cv::ml::SVM> svmArg)
	: svm(svmArg)
	{}

	int classify(Mat img, HOGDescriptor hog);
	cv::Mat preprocessImage(Mat img);

private:
	cv::Ptr<cv::ml::SVM> svm;
};

}