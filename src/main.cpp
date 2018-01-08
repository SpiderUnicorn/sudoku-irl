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

    Mat src = imread(image_path, CV_LOAD_IMAGE_UNCHANGED);
    imshow("Source", src);

    Mat thresholded;
    pre_process(src, thresholded);

    imshow("Thresholded", thresholded);

    vector<Point> contour = get_largest_contour(thresholded);

    vector<vector<Point>> contours_poly(1);
    approxPolyDP(Mat(contour), contours_poly[0], 5, true);

    // corners of the game area
    Point2f corners[4];
    // corners of the image
    Point2f cornersDest[4];
    cornersDest[0] = Point2f(500, 0);
    cornersDest[1] = Point2f(0, 0);
    cornersDest[2] = Point2f(0, 500);
    cornersDest[3] = Point2f(500, 500);

    for (int i = 0; i < 4; i++)
    {
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
        Size(500, 500));

    Mat undistortedThreshed = undistorted.clone();

    pre_process(undistorted, undistortedThreshed);

    fastNlMeansDenoising(undistortedThreshed, undistortedThreshed, 10);

    Mat lines = Mat::zeros( undistortedThreshed.size(),undistortedThreshed.type() );
    extract_lines(undistortedThreshed, lines, HORIZONTAL);
    extract_lines(undistortedThreshed, lines, VERTICAL);

    //imshow("lines: " + filename, lines);

    // subtract grid lines from the black/white image
    // so they don't interfere with digit detection
    undistortedThreshed = undistortedThreshed - lines;


    Mat restLines = Mat::zeros( undistortedThreshed.size(),undistortedThreshed.type() );
    test_remove_lines(undistortedThreshed, restLines);
    undistortedThreshed = undistortedThreshed - restLines;

    blur(undistortedThreshed, undistortedThreshed, Size(1, 1));

    imshow("undistortedThreshed", undistortedThreshed);
    imwrite("no_lines.png", undistortedThreshed);

    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    // svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 50, 1e-6));
    svm->setKernel(SVM::RBF);
    svm->setC(5);
    svm->setGamma(0.50625);

    Mat handWritten = imread("digits.png", CV_LOAD_IMAGE_GRAYSCALE);
    Ptr<TrainData> trainData = get_train_data(hog, handWritten, 20);

    svm->train(trainData);
    svm->save("svm.xml");
	// svm = Algorithm::load<SVM>("svm.xml");

    std::cout << "done training... phew" << endl;

    int dist = ceil((double)500 / 9);
    Mat currentCell = Mat(dist, dist, CV_8UC1);

    int _ = -1; //typ
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

    DigitClassifier *classifier = new DigitClassifier(svm);
    for (int row = 0; row < 9; row++)
    {
        for (int col = 0; col < 9; col++)
        {
            for (int y = 0; y < dist && row * dist + y < undistortedThreshed.cols; y++)
            {

                uchar *ptr = currentCell.ptr(y);

                for (int x = 0; x < dist && col * dist + x < undistortedThreshed.rows; x++)
                {
                    ptr[x] = undistortedThreshed.at<uchar>(row * dist + y, col * dist + x);
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
