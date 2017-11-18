#include <iostream>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <string>
#include <math.h>
#include <vector>

using namespace cv;
using namespace std;

#define PI 3.14159265

/** Global variables */
String cascade_path = "cascade.xml";
CascadeClassifier cascade;

class Classifier {
    Mat mag, grey;
    vector<Point> centers;
    vector<Point> circle_centers;
    vector<Rect> viola;
    public:
    Mat image, output_image;
    int read_image (char*);
    int load_cascade (String);
    void convert_grey();
    Mat convolve(Mat);
    void sobel();
    void write_image_mag(String);
    void houghTransformLine();
    void houghTransformCircle();
    void violaJones();
    void write_output_image(String);
};

Mat Classifier::convolve (Mat kernel) {
    Mat output;
    output.create(grey.size(), grey.type());

    double temp[grey.rows][grey.cols];
    int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
    int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;

    cv::Mat paddedInput;
    cv::copyMakeBorder(grey, paddedInput, kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY, BORDER_REPLICATE);

    // now we can do the convoltion
    for ( int i = 0; i < grey.rows; i++ ) {
        for( int j = 0; j < grey.cols; j++ ) {
            double sum = 0.0;

            for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ ) {
                for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ ) {
                    // find the correct indices we are using
                    int imagex = i + m + kernelRadiusX;
                    int imagey = j + n + kernelRadiusY;
                    int kernelx = m + kernelRadiusX;
                    int kernely = n + kernelRadiusY;

                    // get the values from the padded image and the kernel
                    int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
                    double kernalval = kernel.at<double>( kernelx, kernely );

                    // do the multiplication
                    sum += imageval * kernalval;
                }
            }
            // set the output value as the sum of the convolution
            temp[i][j]= sum;
        }
    }

    float min, max = 0;
    int tempMin = 999999;
    int tempMax = 0;
    for ( int i = 0; i < grey.rows; i++ ) {
        for( int j = 0; j < grey.cols; j++ ) {
            if(temp[i][j] > tempMax) {
                tempMax = temp[i][j];
            }
            if(temp[i][j] < tempMin) {
                tempMin = temp[i][j];
            }
        }
    }

    min = tempMin;
    max = tempMax;

    for ( int i = 0; i < grey.rows; i++ ) {
        for( int j = 0; j < grey.cols; j++ ) {
            output.at<uchar>(i, j) = (255-0)/(max-min)*(temp[i][j]-max)+255;
        }
    }

    return output;
}

void Classifier::sobel() {
    cv::Mat outputX, outputY;
    outputX.create(grey.size(), grey.type());
    outputY.create(grey.size(), grey.type());

    //input -> grey
    mag.create(grey.size(), grey.type());

    float dataX[] = {1,0,-1,2,0,-2,1,0,-1};
    cv::Mat kernelX(3,3, CV_32F, dataX);

    float dataY[] = {1,2,1,0,0,0,-1,-2,-1};
    cv::Mat kernelY(3,3, CV_32F, dataY);

    //find dx and dy
    outputX = convolve(kernelX);
    outputY = convolve(kernelY);

    double tempMag[grey.rows][grey.cols];

    //find mag
    for ( int i = 0; i < grey.rows; i++ ) {
        for( int j = 0; j < grey.cols; j++ ) {
            float temp = outputX.at<uchar>(i,j);
            tempMag[i][j] = sqrt((outputX.at<uchar>(i,j) * outputX.at<uchar>(i,j)) + (outputY.at<uchar>(i,j) * outputY.at<uchar>(i,j)));
        }
    }

    float min, max = 0;
    int tempMin = 999999;
    int tempMax = 0;
    for ( int i = 0; i < grey.rows; i++ ) {
        for( int j = 0; j < grey.cols; j++ ) {
            if(tempMag[i][j] > tempMax) {
                tempMax = tempMag[i][j];
            }
            if(tempMag[i][j] < tempMin) {
                tempMin = tempMag[i][j];
            }
        }
    }
    min = tempMin;
    max = tempMax;

    for ( int i = 0; i < grey.rows; i++ ) {
        for( int j = 0; j < grey.cols; j++ ) {
            mag.at<uchar>(i, j) = (255-0)/(max-min)*(tempMag[i][j]-max)+255;
        }
    }

    return;
}


// Finds the intersection of two lines, or returns false.
// The lines are defined by (o1, p1) and (o2, p2).
bool intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2, Point2f &r) {
    Point2f x = o2 - o1;
    Point2f d1 = p1 - o1;
    Point2f d2 = p2 - o2;

    float cross = d1.x*d2.y - d1.y*d2.x;
    if (abs(cross) < /*EPS*/1e-8) {
        return false;
    }

    double t1 = (x.x * d2.y - x.y * d2.x)/cross;
    r = o1 + d1 * t1;

    return true;
}


void Classifier::houghTransformLine() {
    Mat workImage;
    vector<Vec2f> lines;

    Canny(mag, workImage, 50, 200, 3);

    //dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
    //lines: A vector that will store the parameters (r,\theta) of the detected lines
    //rho : The resolution of the parameter r in pixels. We use 1 pixel.
    //theta: The resolution of the parameter \theta in radians. We use 1 degree (CV_PI/180)
    //threshold: The minimum number of intersections to “detect” a line
    //srn and stn: Default parameters to zero. Check OpenCV reference for more info.
    HoughLines(workImage, lines, 1, CV_PI/180, 80, 0, 0 );

    const int x = mag.rows;
    const int y = mag.cols;

    int crosses[x][y];
    float sum = 0;

    for ( int i = 0; i < mag.rows; i++ ) {
        for( int j = 0; j < mag.cols; j++ ) {
            crosses[i][j] = 0;
        }
    }

    for( size_t i = 0; i < lines.size(); i++ ) {
        for( size_t j = i+1; j < lines.size(); j++ ) {
            float rho = lines[i][0], theta = lines[i][1];
            Point o1, p1;
            double a = cos(theta), b = sin(theta);
            double x0 = a*rho, y0 = b*rho;
            o1.x = cvRound(x0 + 1000*(-b));
            o1.y = cvRound(y0 + 1000*(a));
            p1.x = cvRound(x0 - 1000*(-b));
            p1.y = cvRound(y0 - 1000*(a));

            rho = lines[j][0];
            theta = lines[j][1];
            Point o2, p2;
            a = cos(theta), b = sin(theta);
            x0 = a*rho, y0 = b*rho;
            o2.x = cvRound(x0 + 1000*(-b));
            o2.y = cvRound(y0 + 1000*(a));
            p2.x = cvRound(x0 - 1000*(-b));
            p2.y = cvRound(y0 - 1000*(a));

            Point2f r;

            if (intersection(o1, p1, o2, p2, r)) {
                if (r.x < mag.rows && r.y < mag.cols && r.x > 0 && r.y > 0) {
                    int x = r.x;
                    int y = r.y;
                    crosses[x][y] += 1;
                    sum ++;
                }
            }
        }
    }

    for ( int i = 0; i < mag.rows; i++ ) {
        for( int j = 0; j < mag.cols; j++ ) {
            // TODO: maybe change this to == 8 or some other
            if (crosses[i][j] > (sum / (5 * lines.size()))) {
                Point center;
                center.x = i;
                center.y = j;
                circle( output_image, center, 7, Scalar(30,255,30), -1, 8, 0 );
                centers.push_back(center);
            }
        }
    }

    return;
}


void Classifier::houghTransformCircle() {
    Mat blur_image;
    vector<Vec3f> circles;

    GaussianBlur(mag, blur_image, Size(9, 9), 1.5, 1.5 );

    //src_gray: Input image (grayscale)
    //circles: A vector that stores sets of 3 values: x_{c}, y_{c}, r for each detected circle.
    //CV_HOUGH_GRADIENT: Define the detection method. Currently this is the only one available in OpenCV
    //dp = 1: The inverse ratio of resolution
    //min_dist = src_gray.rows/8: Minimum distance between detected centers
    //param_1 = 200: Upper threshold for the internal Canny edge detector
    //param_2 = 100*: Threshold for center detection.
    //min_radius = 0: Minimum radio to be detected. If unknown, put zero as default.
    //max_radius = 0: Maximum radius to be detected. If unknown, put zero as default
    HoughCircles(blur_image, circles, CV_HOUGH_GRADIENT, 1, output_image.rows/16, 100, 35, 10, 120);

    for( size_t i = 0; i < circles.size(); i++ ) {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        circle( output_image, center, 3, Scalar(0,0,255), -1, 8, 0 );
        circle( output_image, center, radius, Scalar(0,0,255), 3, 8, 0 );
        circle_centers.push_back(center);
    }

    return;
}

void Classifier::violaJones() {
    cascade.detectMultiScale(grey, viola, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

    for( int i = 0; i < viola.size(); i++ ) {
        rectangle(output_image, Point(viola[i].x, viola[i].y), Point(viola[i].x + viola[i].width, viola[i].y + viola[i].height), Scalar( 255, 0, 0 ), 2);
    }

    return;
}

int Classifier::read_image (char* path) {
    image = imread(path, CV_LOAD_IMAGE_COLOR);
    if(!image.data){
        printf( " No image data \n " );
        return -1;
    }

    return 0;
}

int Classifier::load_cascade(String path) {
    if(!cascade.load( path )) {
        printf("Could no load cascade xml\n");
        return -1;
    }

    return 0;
}

void Classifier::write_image_mag(String path) {
    imwrite(path, mag);
}

void Classifier::convert_grey() {
    cvtColor(image, grey, CV_BGR2GRAY);
    //equalizeHist(grey, grey);
}

void Classifier::write_output_image(String path) {
    imwrite(path, output_image);
}

int main( int argc, char** argv ) {
    Classifier darts;

    if (argc != 3) {
        printf("enter input and output image only");
        return -1;
    }

    if (darts.read_image(argv[1])) {
        return -1;
    }
    if (darts.load_cascade(cascade_path)) {
        return -1;
    }

    darts.output_image = darts.image;
    darts.convert_grey();

    darts.sobel();
    darts.houghTransformLine();
    darts.houghTransformCircle();
    darts.violaJones();

    darts.write_output_image(argv[2]);

    //vector<Point>::iterator center;
    //for (center = centers.begin(); center != centers.end(); ++center) {
    //        printf("%d-%d\n", center->x, center->y);
    //}


    return 0;
}
