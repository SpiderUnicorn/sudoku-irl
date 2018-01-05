#include <stdio.h>
#include "digit_recognizer.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace cv::ml;
using namespace std;

typedef unsigned char BYTE;

#define PIXELS_IN_SIDE 28
#define PIXELS_IN_IMAGE PIXELS_IN_SIDE * PIXELS_IN_SIDE
#define ENABLE_TRAIN 0

// do better when awake
int SZ_d = 28; // was 20
float affineFlags_d = WARP_INVERSE_MAP | INTER_LINEAR;

HOGDescriptor initHog() {
	auto windowSize = Size(PIXELS_IN_SIDE, PIXELS_IN_SIDE);
	auto blockSize = Size(PIXELS_IN_SIDE / 2, PIXELS_IN_SIDE / 2);
	auto blockStride = Size(7, 7);
	auto cellSize = Size(14, 14);

	HOGDescriptor hog(
		windowSize,
		blockSize,
		blockStride,
		cellSize,
		9,			  //nbins,
		1,			  //derivAper,
		-1,			  //winSigma,
		0,			  //histogramNormType,
		0.2,		  //L2HysThresh,
		1,			  //gammal correction,
		64,			  //nlevels=64
		1);			  //Use signed gradients

	return hog;
}

HOGDescriptor hog = initHog();

Mat deskew(Mat &img)
{
	Moments m = moments(img);
	if (abs(m.mu02) < 1e-2)
	{
		return img.clone();
	}
	float skew = m.mu11 / m.mu02;
	Mat warpMat = (Mat_<float>(2, 3) << 1, skew, -0.5 * SZ_d * skew, 0, 1, 0);
	Mat imgOut = Mat::zeros(img.rows, img.cols, img.type());
	warpAffine(img, imgOut, warpMat, imgOut.size(), affineFlags_d);

	return imgOut;
}

DigitRecognizer::DigitRecognizer()
{
	svm = cv::ml::SVM::create();
	// svm->setKernel(SVM::LINEAR);
	svm->setType(SVM::C_SVC);
	// svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 50, 1e-6));

	// Set SVM Kernel to Radial Basis Function (RBF)
	svm->setKernel(SVM::RBF);
	// Set parameter C
	svm->setC(5);
	// Set parameter Gamma
	svm->setGamma(0.50625);
}

DigitRecognizer::~DigitRecognizer()
{
	delete svm;
}

void digit_bounding_box(Mat &img, int &rowTop, int &rowBottom, int &colLeft, int &colRight) {
	int thresholdBottom = 50;
	int thresholdTop = 50;
	int thresholdLeft = 50;
	int thresholdRight = 50;
	int center = img.rows / 2;

	for (int i = center; i < img.rows; i++)
	{
		if (rowBottom == -1)
		{
			IplImage stub = img.row(i);
			if (cvSum(&stub).val[0] < thresholdBottom || i == img.rows - 1)
				rowBottom = i;
		}

		if (rowTop == -1)
		{
			IplImage stub = img.row(img.rows - i);
			if (cvSum(&stub).val[0] < thresholdTop || i == img.rows - 1)
				rowTop = img.rows - i;
		}

		if (colRight == -1)
		{
			IplImage stub = img.col(i);
			if (cvSum(&stub).val[0] < thresholdRight || i == img.cols - 1)
				colRight = i;
		}

		if (colLeft == -1)
		{
			IplImage stub = img.col(img.cols - i);
			if (cvSum(&stub).val[0] < thresholdLeft || i == img.cols - 1)
				colLeft = img.cols - i;
		}
	}
}

void center_digit(Mat &inbound, Mat &outbound) {
	int rowTop = -1, rowBottom = -1, colLeft = -1, colRight = -1;

	digit_bounding_box(inbound, rowTop, rowBottom, colLeft, colRight);

	outbound = outbound.zeros(inbound.rows, inbound.cols, CV_8UC1);

	int startAtX = (outbound.cols / 2) - (colRight - colLeft) / 2;

	int startAtY = (outbound.rows / 2) - (rowBottom - rowTop) / 2;

	for (int y = startAtY; y < (outbound.rows / 2) + (rowBottom - rowTop) / 2; y++)
	{
		uchar *ptr = outbound.ptr<uchar>(y);
		for (int x = startAtX; x < (outbound.cols / 2) + (colRight - colLeft) / 2; x++)
		{
			ptr[x] = inbound.at<uchar>(rowTop + (y - startAtY), colLeft + (x - startAtX));
		}
	}
}

Mat DigitRecognizer::preprocessImage(Mat img)
{
	Mat centeredDigit;
	center_digit(img, centeredDigit);

	Mat cloneImg = Mat(numRows, numCols, CV_8UC1);

	resize(img, cloneImg, Size(numCols, numRows));

	// remove any lines near the image borders
	rectangle(cloneImg, Point(0, 0), Point(27, 27), Scalar(0, 0, 0), 5);

	cloneImg = cloneImg.reshape(1, 1);

	return cloneImg;
}

int DigitRecognizer::classify(Mat img)
{
	Mat cloneImg = preprocessImage(img);

	// im is of type Mat

	vector<float> descriptors;
	// Mat deskewedHog = img.reshape(1, 28);

	hog.compute(img, descriptors);

	int prediction = svm->predict(descriptors);

	/*
	Mat foo = cloneImg.reshape(1, 28);
	imshow(to_string(prediction), foo);
	waitKey(2000);
	*/

	return prediction;
}

// Fix for Intel processors big endian
int reverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;

	return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

bool DigitRecognizer::train(const char *trainPath, const char *labelsPath)
{
	std::cout << "trying to open files" << endl;

	std::ifstream pic_file(trainPath, std::ios::binary);
	std::ifstream label_file(labelsPath, std::ios::binary);

	std::cout << "opened files, yay" << endl;

	if (!pic_file.is_open() || !label_file.is_open())
		return false;

	int magic_number = 0;
	int number_of_images = 0;
	int n_rows = 0;
	int n_cols = 0;

	label_file.read((char *)&magic_number, sizeof(magic_number));
	pic_file.read((char *)&magic_number, sizeof(magic_number));
	magic_number = reverseInt(magic_number);

	label_file.read((char *)&number_of_images, sizeof(number_of_images));
	pic_file.read((char *)&number_of_images, sizeof(number_of_images));
	number_of_images = reverseInt(number_of_images) / 10;

	pic_file.read((char *)&n_rows, sizeof(n_rows));
	n_rows = reverseInt(n_rows);
	pic_file.read((char *)&n_cols, sizeof(n_cols));
	n_cols = reverseInt(n_cols);

	numRows = n_rows;
	numCols = n_cols;

	int n_stride = n_cols * n_rows;
	Mat training_data = Mat(number_of_images, n_stride, CV_8U);
	Mat label_data = Mat(number_of_images, 1, CV_8U);

	vector<vector<float>> trainHOG;

	for (int i = 0; i < number_of_images; ++i)
	{
		unsigned char data_tmp[PIXELS_IN_IMAGE];
		pic_file.read((char *)data_tmp, sizeof(unsigned char) * n_stride);
		Mat row_image(1, n_stride, CV_8U, data_tmp);
		row_image.row(0).copyTo(training_data.row(i));

		// maybe this goes here??
		vector<float> descriptors;
		auto cell = training_data.row(i);
		Mat deskewedHog = cell.reshape(1, 28);
		deskewedHog = deskew(deskewedHog);

		hog.compute(deskewedHog, descriptors);
		trainHOG.push_back(descriptors);

		char label = 0;
		label_file.read((char *)&label, sizeof(label));
		label_data.at<uchar>(i, 0) = label;
	}

	std::cout << "creating train data" << endl;

	training_data.convertTo(training_data, CV_32FC1);
	label_data.convertTo(label_data, CV_32SC1);

	// Ptr<ml::SVM> svm;
	std::cout << "at the station" << endl;

	int descriptor_size = trainHOG[0].size();

	for (int i = 0; i < trainHOG.size(); i++)
	{
		for (int j = 0; j < descriptor_size; j++)
		{
			training_data.at<float>(i, j) = trainHOG[i][j];
		}
	}

	Ptr<TrainData> td = TrainData::create(training_data, ROW_SAMPLE, label_data);
	if (ENABLE_TRAIN)
	{
		svm->train(td);
		svm->save("SVM_MNIST.xml");
	}
	else
	{
		svm = Algorithm::load<SVM>("SVM_MNIST.xml");
	}
	
	cout << "accuracy: " << endl;
	int correct_count = 0;
	for (int idx = 0; idx < label_data.rows; idx++)
	{
		// float response = svm->predict(training_data.row(idx));
		/*
		Mat foo = training_data.row(idx);
		foo = foo.reshape(1, 28);
		String hest = to_string(label_data.at<uchar>(idx, 0)) + " " + to_string(response);
		imshow(hest, foo);
		waitKey(2000);
		*/
		/*
		if (label_data.at<uchar>(idx, 0) == (uchar)response)
		{
			correct_count++;
		}
		*/
	}

	double correct_ratio = (double)correct_count / (double)label_data.rows;
	cout << 1 - correct_ratio << endl;

	return true;
}