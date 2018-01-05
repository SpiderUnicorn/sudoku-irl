#include <stdio.h>
#include "digit_recognizer.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace cv::ml;
using namespace std;

typedef unsigned char BYTE;

#define PIXELS_IN_SIDE 20
#define PIXELS_IN_IMAGE PIXELS_IN_SIDE * PIXELS_IN_SIDE
#define ENABLE_TRAIN 1

string pathName = "digits.png";
float affineFlags_d = WARP_INVERSE_MAP | INTER_LINEAR;

HOGDescriptor initHog() {
	auto windowSize = Size(PIXELS_IN_SIDE, PIXELS_IN_SIDE);
	auto blockSize = Size(8, 8);
	auto blockStride = Size(4, 4);
	auto cellSize = Size(8, 8);

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
	Mat warpMat = (Mat_<float>(2, 3) << 1, skew, -0.5 * PIXELS_IN_SIDE * skew, 0, 1, 0);
	Mat imgOut = Mat::zeros(img.rows, img.cols, img.type());
	warpAffine(img, imgOut, warpMat, imgOut.size(), affineFlags_d);

	return imgOut;
}

void load_train_data(string &pathName, vector<Mat> &trainCells, vector<Mat> &testCells,vector<int> &trainLabels, vector<int> &testLabels)
{
    Mat img = imread(pathName,CV_LOAD_IMAGE_GRAYSCALE);
    int ImgCount = 0;
    for(int i = 0; i < img.rows; i = i + PIXELS_IN_SIDE)
    {
        for(int j = 0; j < img.cols; j = j + PIXELS_IN_SIDE)
        {
            Mat digitImg = (img.colRange(j,j+PIXELS_IN_SIDE).rowRange(i,i+PIXELS_IN_SIDE)).clone();
            if(j < int(0.9*img.cols))
            {
                trainCells.push_back(digitImg);
            }
            else
            {
                testCells.push_back(digitImg);
            }
            ImgCount++;
        }
    }

    cout << "Image Count : " << ImgCount << endl;
    float digitClassNumber = 0;

    for(int z=0;z<int(0.9*ImgCount);z++){
        if(z % 450 == 0 && z != 0){
            digitClassNumber = digitClassNumber + 1;
            }
        trainLabels.push_back(digitClassNumber);
    }
    digitClassNumber = 0;
    for(int z=0;z<int(0.1*ImgCount);z++){
        if(z % 50 == 0 && z != 0){
            digitClassNumber = digitClassNumber + 1;
            }
        testLabels.push_back(digitClassNumber);
    }
}

void CreateDeskewedTrainTest(vector<Mat> &deskewedTrainCells,vector<Mat> &deskewedTestCells, vector<Mat> &trainCells, vector<Mat> &testCells){
    for(int i=0;i<trainCells.size();i++){

        Mat deskewedImg = deskew(trainCells[i]);
        deskewedTrainCells.push_back(deskewedImg);
    }

    for(int i=0;i<testCells.size();i++){

        Mat deskewedImg = deskew(testCells[i]);
        deskewedTestCells.push_back(deskewedImg);
    }
}

void CreateTrainTestHOG(vector<vector<float> > &trainHOG, vector<vector<float> > &testHOG, vector<Mat> &deskewedtrainCells, vector<Mat> &deskewedtestCells){

    for(int y=0;y<deskewedtrainCells.size();y++){
        vector<float> descriptors;
        hog.compute(deskewedtrainCells[y],descriptors);
        trainHOG.push_back(descriptors);
    }

    for(int y=0;y<deskewedtestCells.size();y++){

        vector<float> descriptors;
        hog.compute(deskewedtestCells[y],descriptors);
        testHOG.push_back(descriptors);
    }
}
void ConvertVectortoMatrix(vector<vector<float> > &trainHOG, vector<vector<float> > &testHOG, Mat &trainMat, Mat &testMat)
{

    int descriptor_size = trainHOG[0].size();

    for(int i = 0;i<trainHOG.size();i++){
        for(int j = 0;j<descriptor_size;j++){
           trainMat.at<float>(i,j) = trainHOG[i][j];
        }
    }
    for(int i = 0;i<testHOG.size();i++){
        for(int j = 0;j<descriptor_size;j++){
            testMat.at<float>(i,j) = testHOG[i][j];
        }
    }
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

	// remove any lines near the image borders
	rectangle(inbound, Point(0, 0), Point(ceil((double)500/9) - 1, ceil((double)500/9) - 1), Scalar(0, 0, 0), 15);
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

	Mat cloneImg = Mat(PIXELS_IN_SIDE, PIXELS_IN_SIDE, CV_8UC1);

	resize(img, cloneImg, Size(PIXELS_IN_SIDE, PIXELS_IN_SIDE));

 	// cloneImg = cloneImg.reshape(1, 1);

	return cloneImg;
}

int DigitRecognizer::classify(Mat img)
{
	cout <<"checkpoint -1" << endl;
	Mat cloneImg = preprocessImage(img);

	// im is of type Mat


	// Mat deskewedHog = img.reshape(1, 28);
 /*
	Mat hogger(cloneImg.size(), 1, CV_32FC1);

	for (int i = 0; i < trainHOG.size(); i++)
	{
		for (int j = 0; j < descriptor_size; j++)
		{
			trainMat.at<float>(i, j) = trainHOG[i][j];
		}
	}

	cout << descriptors[0].cols << endl;

	*/

	vector<float> descriptors;
	cout <<"checkpoint 0" << endl;
	hog.compute(cloneImg, descriptors);

	Mat hest(1, descriptors.size(), CV_32FC1);

	int i = 0;
	for(int j = 0; j < descriptors.size(); j++){
		hest.at<float>(i,j) = descriptors[j];
	}

	cout <<"checkpoint" << endl;

	imshow("meh", cloneImg);

	int prediction = svm->predict(hest);

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

bool DigitRecognizer::train()
{
	std::cout << "trying to open files" << endl;
	vector<Mat> trainCells;
	vector<Mat> testCells;
	vector<int> trainLabels;
	vector<int> testLabels;
	load_train_data(pathName, trainCells, testCells, trainLabels, testLabels);

	vector<Mat> deskewedTrainCells;
    vector<Mat> deskewedTestCells;
    CreateDeskewedTrainTest(deskewedTrainCells,deskewedTestCells,trainCells,testCells);

    std::vector<std::vector<float> > trainHOG;
    std::vector<std::vector<float> > testHOG;
	// hog->compute every cell in the test data set and get descriptors
    CreateTrainTestHOG(trainHOG,testHOG,deskewedTrainCells,deskewedTestCells);

    int descriptor_size = trainHOG[0].size();
    cout << "Descriptor Size : " << descriptor_size << endl;

    Mat trainMat(trainHOG.size(),descriptor_size,CV_32FC1);
    Mat testMat(testHOG.size(),descriptor_size,CV_32FC1);

    ConvertVectortoMatrix(trainHOG,testHOG,trainMat,testMat);

	// ....
	Ptr<TrainData> td = TrainData::create(trainMat, ROW_SAMPLE, trainLabels);

	if (ENABLE_TRAIN)
	{
		svm->train(td);
		svm->save("svm.xml");
	}
	else
	{
		svm = Algorithm::load<SVM>("svm.xml");
	}

	Mat testResponse;
	float count = 0;
    float accuracy = 0 ;
	svm->predict(testMat, testResponse);

	for(int i = 0; i < testResponse.rows; i++)
	{
		if(testResponse.at<float>(i,0) == testLabels[i])
			count = count + 1;
	}
	accuracy = (count/testResponse.rows)*100;
	cout << "accuracy " << accuracy << endl;


	return true;
}