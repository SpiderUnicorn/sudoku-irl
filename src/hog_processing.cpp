#include "image_processing.h"
#include "hog_processing.h"

using namespace cv;
using namespace cv::ml;
using namespace std;

namespace sudoku {

void parse_training_data(Mat img, vector<Mat> &digits, vector<int> &labels, const int SIDE_LENGTH)
{
    const int DIGITS_IN_BASE_TEN = 10;
    const int DIGITS_PER_ROW = img.size().width / SIDE_LENGTH;
    const int LINES_PER_DIGIT = img.size().height / DIGITS_IN_BASE_TEN / SIDE_LENGTH;
    const int IMAGES_PER_DIGIT = DIGITS_PER_ROW * LINES_PER_DIGIT;

    // Extract digits from img
	for (int row = 0; row < img.rows; row += SIDE_LENGTH)
	{
		for (int col = 0; col < img.cols; col += SIDE_LENGTH)
		{
            Mat digit = img.colRange(col, col + SIDE_LENGTH).rowRange(row, row + SIDE_LENGTH);
			// center_digit(digit, digit);
			digits.push_back(digit.clone());
		}
	}

    // -1 since the loop will increment to 0 on first number check
	int currentDigit = -1;

    // Extract digit labels from image
	for (int digitIndex = 0; digitIndex < digits.size(); digitIndex++)
	{
		if (digitIndex % IMAGES_PER_DIGIT == 0)
		{
			currentDigit += 1;
		}

		labels.push_back(currentDigit);
	}
}

void compute_hogs(vector<vector<float>> &hogs, vector<Mat> &digits, HOGDescriptor hogDescriptor)
{
	for (auto digit: digits)
	{
		vector<float> descriptors;
		hogDescriptor.compute(digit, descriptors);
		hogs.push_back(descriptors);
	}
}

void vector_to_mat(vector<vector<float>> &vect, Mat &mat)
{
	int vector_size = vect[0].size();

	for (int row = 0; row < vect.size(); row++)
	{
		for (int col = 0; col < vector_size; col++)
		{
			mat.at<float>(row, col) = vect[row][col];
		}
	}
}

Ptr<TrainData> get_train_data(HOGDescriptor hog, Mat img, const int SIDE_LENGTH)
{
	vector<Mat> trainCells;
	vector<int> trainLabels;
	parse_training_data(img, trainCells, trainLabels, SIDE_LENGTH);

	vector<Mat> deskewedTrainCells;
    deskew_all(deskewedTrainCells, trainCells);

	vector<vector<float>> trainHOG;
	compute_hogs(trainHOG, deskewedTrainCells, hog);

	int descriptor_size = trainHOG[0].size();
	Mat trainMat(trainHOG.size(), descriptor_size, CV_32FC1);

	vector_to_mat(trainHOG, trainMat);
	cout << trainHOG[0].size() << endl;

	return TrainData::create(trainMat, ROW_SAMPLE, trainLabels);
}

}
