#include <stdio.h>
#include "digit_recognizer.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace cv::ml;
using namespace std;

typedef unsigned char BYTE;

#define PIXELS_IN_IMAGE 28*28
#define ENABLE_TRAIN 1

HOGDescriptor hog(
	Size(28,28), //winSize
	Size(14,14), //blocksize
	Size(7,7), //blockStride,
	Size(14,14), //cellSize,
				9, //nbins,
				1, //derivAper,
				-1, //winSigma,
				0, //histogramNormType,
			0.2, //L2HysThresh,
				1,//gammal correction,
				64,//nlevels=64
				1);//Use signed gradients

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

Mat DigitRecognizer::preprocessImage(Mat img)
{
	int rowTop=-1, rowBottom=-1, colLeft=-1, colRight=-1;

    Mat temp;
    int thresholdBottom = 50;
    int thresholdTop = 50;
    int thresholdLeft = 50;
    int thresholdRight = 50;
    int center = img.rows/2;
    for(int i=center;i<img.rows;i++)
    {
        if(rowBottom==-1)
        {
            temp = img.row(i);
            IplImage stub = temp;
            if(cvSum(&stub).val[0] < thresholdBottom || i==img.rows-1)
                rowBottom = i;

        }

        if(rowTop==-1)
        {
            temp = img.row(img.rows-i);
            IplImage stub = temp;
            if(cvSum(&stub).val[0] < thresholdTop || i==img.rows-1)
                rowTop = img.rows-i;

        }

        if(colRight==-1)
        {
            temp = img.col(i);
            IplImage stub = temp;
            if(cvSum(&stub).val[0] < thresholdRight|| i==img.cols-1)
                colRight = i;

        }

        if(colLeft==-1)
        {
            temp = img.col(img.cols-i);
            IplImage stub = temp;
            if(cvSum(&stub).val[0] < thresholdLeft|| i==img.cols-1)
                colLeft = img.cols-i;
        }
    }


	Mat newImg;
    newImg = newImg.zeros(img.rows, img.cols, CV_8UC1);

    int startAtX = (newImg.cols/2)-(colRight-colLeft)/2;

    int startAtY = (newImg.rows/2)-(rowBottom-rowTop)/2;

    for(int y=startAtY;y<(newImg.rows/2)+(rowBottom-rowTop)/2;y++)
    {
        uchar *ptr = newImg.ptr<uchar>(y);
        for(int x=startAtX;x<(newImg.cols/2)+(colRight-colLeft)/2;x++)
        {
            ptr[x] = img.at<uchar>(rowTop+(y-startAtY),colLeft+(x-startAtX));
        }
    }
	Mat cloneImg = Mat(numRows, numCols, CV_8UC1);

	resize(newImg, cloneImg, Size(numCols, numRows));

	rectangle(cloneImg, Point(0, 0), Point(27, 27), Scalar(0, 0, 0), 5);

    // Now don't fill along the borders
	/*
    for(int i=0;i<cloneImg.rows;i++)
    {
        floodFill(cloneImg, cvPoint(0, i), cvScalar(0,0,0));

        floodFill(cloneImg, cvPoint(cloneImg.cols-1, i), cvScalar(0,0,0));

        floodFill(cloneImg, cvPoint(i, 0), cvScalar(0));
        floodFill(cloneImg, cvPoint(i, cloneImg.rows-1), cvScalar(0));
    }
	*/

	cloneImg = cloneImg.reshape(1, 1);

	return cloneImg;
}

int DigitRecognizer::classify(Mat img)
{
	Mat cloneImg = preprocessImage(img);

	// im is of type Mat


	int prediction = svm->predict(Mat_<float>(cloneImg));

	/*
	Mat foo = cloneImg.reshape(1, 28);
	imshow(to_string(prediction), foo);
	waitKey(2000);
	*/

	return prediction;
}

// Fix for Intel processors big endian
int reverseInt(int i) {
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

	label_file.read((char*)&magic_number, sizeof(magic_number));
	pic_file.read((char*)&magic_number, sizeof(magic_number));
	magic_number = reverseInt(magic_number);

	label_file.read((char*)&number_of_images, sizeof(number_of_images));
	pic_file.read((char*)&number_of_images, sizeof(number_of_images));
	number_of_images = reverseInt(number_of_images) / 10;

	pic_file.read((char*)&n_rows, sizeof(n_rows));
	n_rows = reverseInt(n_rows);
	pic_file.read((char*)&n_cols, sizeof(n_cols));
	n_cols = reverseInt(n_cols);

	numRows = n_rows;
	numCols = n_cols;

	int n_stride = n_cols * n_rows;
	Mat training_data = Mat(number_of_images, n_stride, CV_8U);
	Mat label_data = Mat(number_of_images, 1, CV_8U);

	vector<vector<float>> trainHOG;

	for (int i = 0; i < number_of_images; ++i) {
		unsigned char data_tmp[PIXELS_IN_IMAGE];
		pic_file.read((char*)data_tmp, sizeof(unsigned char) * n_stride);
		Mat row_image(1, n_stride, CV_8U, data_tmp);
		row_image.row(0).copyTo(training_data.row(i));

		// maybe this goes here??
		vector<float> descriptors;
        hog.compute(training_data.row(i), descriptors);
        trainHOG.push_back(descriptors);

		char label = 0;
		label_file.read((char*)&label, sizeof(label));
		label_data.at<uchar>(i, 0) = label;
	}


	std::cout << "creating train data" << endl;

	training_data.convertTo(training_data, CV_32FC1);
	label_data.convertTo(label_data, CV_32SC1);

	// Ptr<ml::SVM> svm;
	std::cout << "at the station" << endl;

	int descriptor_size = trainHOG[0].size();

	for(int i = 0;i<trainHOG.size();i++){
        for(int j = 0;j<descriptor_size;j++){
           training_data.at<float>(i,j) = trainHOG[i][j];
        }
    }

	Ptr<TrainData> td = TrainData::create(training_data, ROW_SAMPLE, label_data);
	if(ENABLE_TRAIN) {
	 	svm->train(td);
		svm->save("SVM_MNIST.xml");
	} else {
		svm = Algorithm::load<SVM>("SVM_MNIST.xml");
	}

	cout << "accuracy: " << endl;
	int correct_count = 0;
	for (int idx = 0; idx < label_data.rows; idx++) {
		float response = svm->predict(training_data.row(idx));
		/*
		Mat foo = training_data.row(idx);
		foo = foo.reshape(1, 28);
		String hest = to_string(label_data.at<uchar>(idx, 0)) + " " + to_string(response);
		imshow(hest, foo);
		waitKey(2000);
		*/
		if (label_data.at<uchar>(idx, 0) == (uchar)response) {
			correct_count++;
		}
	}


	double correct_ratio = (double)correct_count / (double)label_data.rows;
	cout << 1-correct_ratio << endl;

	return true;
}