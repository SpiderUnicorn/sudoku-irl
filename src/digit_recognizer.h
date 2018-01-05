#include <cv.h>
#include <opencv2/opencv.hpp>
#include <highgui.h>
#include <ml.h>

using namespace cv;

#define MAX_NUM_IMAGES	60000

class DigitRecognizer
{
public:
	DigitRecognizer();
	~DigitRecognizer();

	bool train(const char* trainPath, const char* labelsPath);
	int classify(Mat img);

private:
	cv::Mat preprocessImage(Mat img);
	int readFlippedInteger(FILE *fp);

private:
	Ptr<cv::ml::SVM> svm;
	int numRows, numCols, numImages;
};