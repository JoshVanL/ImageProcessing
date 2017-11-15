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

#define PI 3.14159265

/** Global variables */
String cascade_name = "cascade.xml";
CascadeClassifier cascade;

void convolve (cv::Mat input, cv::Mat kernel, cv::Mat output) {

    double temp[input.rows][input.cols];
    int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
    int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;

    cv::Mat paddedInput;
    cv::copyMakeBorder( input, paddedInput,
            kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
            cv::BORDER_REPLICATE );

    // now we can do the convoltion
    for ( int i = 0; i < input.rows; i++ )
    {
        for( int j = 0; j < input.cols; j++ )
        {
            double sum = 0.0;
            for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ )
            {
                for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ )
                {
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
    for ( int i = 0; i < input.rows; i++ ) {
        for( int j = 0; j < input.cols; j++ ) {
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

    for ( int i = 0; i < input.rows; i++ ) {
        for( int j = 0; j < input.cols; j++ ) {
            output.at<uchar>(i, j) = (255-0)/(max-min)*(temp[i][j]-max)+255;
        }
    }
}

void sobel(cv::Mat &input, cv::Mat &mag) {
    cv::Mat outputX, outputY;

    outputX.create(input.size(), input.type());
    outputY.create(input.size(), input.type());
    mag.create(input.size(), input.type());

    float dataX[] = {1,0,-1,2,0,-2,1,0,-1};
    cv::Mat kernelX(3,3, CV_32F, dataX);

    float dataY[] = {1,2,1,0,0,0,-1,-2,-1};
    cv::Mat kernelY(3,3, CV_32F, dataY);

    //find dx and dy
    convolve(input, kernelX, outputX);
    convolve(input, kernelY, outputY);

    double tempMag[input.rows][input.cols];

    //find mag
    for ( int i = 0; i < input.rows; i++ ) {
        for( int j = 0; j < input.cols; j++ ) {
            float temp = outputX.at<uchar>(i,j);
            tempMag[i][j] = sqrt((outputX.at<uchar>(i,j) * outputX.at<uchar>(i,j)) + (outputY.at<uchar>(i,j) * outputY.at<uchar>(i,j)));
        }
    }

    float min, max = 0;
    int tempMin = 999999;
    int tempMax = 0;
    for ( int i = 0; i < input.rows; i++ ) {
        for( int j = 0; j < input.cols; j++ ) {
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

    for ( int i = 0; i < input.rows; i++ ) {
        for( int j = 0; j < input.cols; j++ ) {
            mag.at<uchar>(i, j) = (255-0)/(max-min)*(tempMag[i][j]-max)+255;
        }
    }

}

// Finds the intersection of two lines, or returns false.
// The lines are defined by (o1, p1) and (o2, p2).
bool intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2, Point2f &r)
{
    Point2f x = o2 - o1;
    Point2f d1 = p1 - o1;
    Point2f d2 = p2 - o2;

    float cross = d1.x*d2.y - d1.y*d2.x;
    if (abs(cross) < /*EPS*/1e-8)
        return false;

    double t1 = (x.x * d2.y - x.y * d2.x)/cross;
    r = o1 + d1 * t1;
    return true;
}


void HoughTransformLine(cv::Mat &input, cv::Mat &output)
{

    Mat workImage;

    Canny(input, workImage, 50, 200, 3);
    vector<Vec2f> lines;

    //dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
    //lines: A vector that will store the parameters (r,\theta) of the detected lines
    //rho : The resolution of the parameter r in pixels. We use 1 pixel.
    //theta: The resolution of the parameter \theta in radians. We use 1 degree (CV_PI/180)
    //threshold: The minimum number of intersections to “detect” a line
    //srn and stn: Default parameters to zero. Check OpenCV reference for more info.
    HoughLines(workImage, lines, 1, CV_PI/180, 80, 0, 0 );

    const int x = input.rows;
    const int y = input.cols;

    int crosses[x][y];
    float sum = 0;

    for ( int i = 0; i < input.rows; i++ )
    {
        for( int j = 0; j < input.cols; j++ )
        {
            crosses[i][j] = 0;
        }
    }

    for( size_t i = 0; i < lines.size(); i++ )
    {
        for( size_t j = i+1; j < lines.size(); j++ )
        {
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
                if (r.x < input.rows && r.y < input.cols && r.x > 0 && r.y > 0) {
                    int x = r.x;
                    int y = r.y;
                    crosses[x][y] += 1;
                    sum ++;
                }
            }

        }
    }

    for ( int i = 0; i < input.rows; i++ )
    {
        for( int j = 0; j < input.cols; j++ )
        {
            if (crosses[i][j] > (sum / (5 * lines.size()))) {
                Point center;
                center.x = i;
                center.y = j;
                circle( output, center, 7, Scalar(30,255,30), -1, 8, 0 );
            }
        }
    }

}


void HoughTransformCircle(cv::Mat &input, cv::Mat &output)
{

    GaussianBlur( input, input, Size(9, 9), 1.5, 1.5 );
    vector<Vec3f> circles;
    //src_gray: Input image (grayscale)
    //circles: A vector that stores sets of 3 values: x_{c}, y_{c}, r for each detected circle.
    //CV_HOUGH_GRADIENT: Define the detection method. Currently this is the only one available in OpenCV
    //dp = 1: The inverse ratio of resolution
    //min_dist = src_gray.rows/8: Minimum distance between detected centers
    //param_1 = 200: Upper threshold for the internal Canny edge detector
    //param_2 = 100*: Threshold for center detection.
    //min_radius = 0: Minimum radio to be detected. If unknown, put zero as default.
    //max_radius = 0: Maximum radius to be detected. If unknown, put zero as default
    HoughCircles( input, circles, CV_HOUGH_GRADIENT, 1, output.rows/16, 100, 35, 10, 120);

    for( size_t i = 0; i < circles.size(); i++ )
    {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        circle( output, center, 3, Scalar(0,0,255), -1, 8, 0 );
        circle( output, center, radius, Scalar(0,0,255), 3, 8, 0 );
    }
}


void ViolaJones(Mat frame, cv::Mat &output)
{
    cascade.load( cascade_name );

    std::vector<Rect> faces;
    Mat frame_gray;

    // 1. Prepare Image by turning it into Grayscale and normalising lighting
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    // 2. Perform Viola-Jones Object Detection
    cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

    // 3. Draw box around faces found
    for( int i = 0; i < faces.size(); i++ )
    {
        rectangle(output, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 255, 0, 0 ), 2);
    }

}


int main( int argc, const char** argv )
{

    Mat image, mag;
    Mat output_image;
    Mat grey_image;

    image = imread( argv[1], CV_LOAD_IMAGE_COLOR);
    if(!image.data )
    {
        printf( " No image data \n " );
        return -1;
    }

    output_image = image;

    cvtColor(image, grey_image, CV_BGR2GRAY);

    sobel(grey_image, mag);
    HoughTransformLine(mag, output_image);
    HoughTransformCircle(mag, output_image);
    ViolaJones(image, output_image);
    imwrite(argv[2], output_image);

    return 0;
}
