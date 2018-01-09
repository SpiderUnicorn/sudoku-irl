#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "pre_process.h"
#include "digit_classifier.h"
#include "image_processing.h"
#include "hog_processing.h"

using namespace cv;
using namespace cv::ml;
using namespace std;
using namespace sudoku;

#define BOARD_SIZE 500

const int SIDE_LENGTH = 20;

HOGDescriptor initHog()
{
	auto windowSize = Size(SIDE_LENGTH, SIDE_LENGTH);
	auto blockSize = Size(8, 8);
	auto blockStride = Size(4, 4);
	auto cellSize = Size(8, 8);

	HOGDescriptor hog(
		windowSize,
		blockSize,
		blockStride,
		cellSize,
		9,   //nbins,
		1,   //derivAper,
		-1,  //winSigma,
		0,   //histogramNormType,
		0.2, //L2HysThresh,
		1,   //gammal correction,
		64,  //nlevels=64
		1);  //Use signed gradients

	return hog;
}

HOGDescriptor hog = initHog();

void test_remove_lines(Mat &img, Mat &out) {
    // Find contours
    vector<vector<Point>> contours;
    findContours(img.clone(), contours, RETR_LIST, CHAIN_APPROX_NONE);

    for (int i = 0; i < contours.size(); i++) {
        const auto& contour = contours[i];
        // Find minimum area rectangle
        RotatedRect rr = minAreaRect(contour);

        // Compute aspect ratio
        float aspect_ratio = min(rr.size.width, rr.size.height) / max(rr.size.width, rr.size.height);

        // Define a threshold on the aspect ratio in [0, 1]
        float thresh = 0.3f;

        Vec3b color = Vec3b(255, 255, 255);
        if (aspect_ratio < thresh) {

            drawContours(out, contours, i, color, CV_FILLED);
            // Almost straight line
        }
    }
}

int main(int argc, char **argv)
{
    const String image_path = argv[1];

    Mat raw_img = imread(image_path, CV_LOAD_IMAGE_UNCHANGED);
    imshow("Source", raw_img);

    Mat board = extract_straightened_board(raw_img, BOARD_SIZE);
    imshow("bored", board);

    pre_process(board, board);
    fastNlMeansDenoising(board, board, 10);

    Mat lines = Mat::zeros( board.size(),board.type() );
    extract_lines(board, lines, HORIZONTAL);
    extract_lines(board, lines, VERTICAL);

    //imshow("lines: " + filename, lines);

    // subtract grid lines from the black/white image
    // so they don't interfere with digit detection
    board = board - lines;


    Mat restLines = Mat::zeros( board.size(),board.type() );
    test_remove_lines(board, restLines);
    board = board - restLines;

    blur(board, board, Size(1, 1));

    imshow("board", board);
    imwrite("no_lines.png", board);

    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    // svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 50, 1e-6));
    svm->setKernel(SVM::RBF);
    svm->setC(5);
    svm->setGamma(0.50625);

    // Mat handWritten = imread("digits.png", CV_LOAD_IMAGE_GRAYSCALE);
    Mat handWritten = imread("digits_computer.png", CV_LOAD_IMAGE_GRAYSCALE);
    Ptr<TrainData> trainData = get_train_data(hog, handWritten, 20);

    // svm->train(trainData);
    // svm->save("svm.xml");
	svm = Algorithm::load<SVM>("svm.xml");

    std::cout << "done training... phew" << endl;

    int dist = ceil((double)BOARD_SIZE / 9);
    Mat currentCell = Mat(dist, dist, CV_8UC1);

    int _ = -1; //typ
    /*
    vector<vector<int>> facitArr = {
			{ 8, _, _,   _, 1, _,   _, _, 9 },
			{ _, 5, _,   8, _, 7,   _, 1, _ },
			{ _, _, 4,   _, 9, _,   7, _, _ },

			{ _, 6, _,   7, _, 1,   _, 2, _ },
			{ 5, _, 8,   _, 6, _,   1, _, 7 },
			{ _, 1, _,   5, _, 2,   _, 9, _ },

			{ _, _, 7,   _, 4, _,   6, _, _ },
			{ _, 8, _,   3, _, 9,   _, 4, _ },
			{ 3, _, _,   _, 5, _,   _, _, 8 },
		};
    */
    vector<vector<int>> facitArr = {
			{ 5, 3, _,   _, 7, _,   _, _, _ },
			{ 6, _, _,   1, 9, 5,   _, _, _ },
			{ _, 9, 8,   _, _, _,   _, 6, _ },

			{ 8, _, _,   _, 6, _,   _, _, 3 },
			{ 4, _, _,   8, _, 3,   _, _, 1 },
			{ 7, _, _,   _, 2, _,   _, _, 6 },

			{ _, 6, _,   _, _, _,   2, 8, _ },
			{ _, _, _,   4, 1, 9,   _, _, 5 },
			{ _, _, _,   _, 8, _,   _, 7, 9 },
		};


    DigitClassifier *classifier = new DigitClassifier(svm);
    for (int row = 0; row < 9; row++)
    {
        for (int col = 0; col < 9; col++)
        {
            for (int y = 0; y < dist && row * dist + y < board.cols; y++)
            {

                uchar *ptr = currentCell.ptr(y);

                for (int x = 0; x < dist && col * dist + x < board.rows; x++)
                {
                    ptr[x] = board.at<uchar>(row * dist + y, col * dist + x);
                }
            }

            Mat processed = classifier->preprocessImage(currentCell);

            Moments m = cv::moments(processed, true);
            int area = m.m00;

            if (area > processed.rows * processed.cols / 25)
            {
                int number = classifier->classify(processed, hog);
                String answer = to_string(facitArr[row][col]);
                String res = number == facitArr[row][col] ? "true" : "false";
                String hest = "(" + to_string(row) + ", " + to_string(col) + ") -> " + to_string(number) + " = " + answer + " " + res;
                if (number != facitArr[row][col]) {
                    imshow(hest, processed);
                }

                cout << hest << endl;

                // waitKey(3000);
            }
            else
            {
                if (facitArr[row][col] != -1) {
                    String hest = "(" + to_string(row) + ", " + to_string(col) + ") != " + to_string(facitArr[row][col]);
                    cout << hest << endl;

                }
            }
            //cout << area << " " << col << "//" << row << endl;
        }
    }
    cout << endl;

    waitKey(0);
    cout << "done" << endl;
    return 0;
}
