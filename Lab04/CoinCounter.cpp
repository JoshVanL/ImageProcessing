// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <math.h>       /* atan */

#define PI 3.14159265

using namespace cv;

void Sobel(
	cv::Mat &input,
	cv::Mat &OutputX,
	cv::Mat &OutputY,
	cv::Mat &Magnitude,
	cv::Mat &Direction
    );

int main( int argc, char** argv )
{

 // LOADING THE IMAGE
 char* imageName = argv[1];

 Mat image;
 image = imread( imageName, 1 );

 if( argc != 2 || !image.data )
 {
   printf( " No image data \n " );
   return -1;
 }

 // CONVERT COLOUR, BLUR AND SAVE
 Mat gray_image;
 Mat imageX;
 Mat imageY;
 Mat Magnitude;
 Mat Direction;
 cvtColor( image, gray_image, CV_BGR2GRAY );
 cvtColor( image, imageX, CV_BGR2GRAY );
 cvtColor( image, imageY, CV_BGR2GRAY );
 cvtColor( image, Magnitude, CV_BGR2GRAY );
 cvtColor( image, Direction, CV_BGR2GRAY );

 Sobel(gray_image, imageX, imageY, Magnitude, Direction);

 imwrite( "imgs/gradX.png", imageX );
 imwrite( "imgs/gradY.png", imageY );
 imwrite( "imgs/magnitude.png", Magnitude );
 imwrite( "imgs/direction.png", Direction );

 return 0;
}

void Sobel(cv::Mat &input, cv::Mat &OutputX, cv::Mat &OutputY, cv::Mat &Magnitude, cv::Mat &Direction)
{
	// intialise the output using the input
	OutputX.create(input.size(), input.type());
	OutputY.create(input.size(), input.type());
	Magnitude.create(input.size(), input.type());
	Direction.create(input.size(), input.type());

    double WorkX[input.rows][input.cols];
    double WorkY[input.rows][input.cols];

	// Derivative in x direction ketnal
    float dataX[] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
    cv::Mat kernelX(3, 3, CV_32F, dataX);

	// Derivative in y direction ketnal
    float dataY[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
    cv::Mat kernelY(3, 3, CV_32F, dataY);


	int kernelRadiusX = ( kernelX.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kernelX.size[1] - 1 ) / 2;

	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput, kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY, cv::BORDER_REPLICATE );

	for ( int i = 0; i < input.rows; i++ )
	{
		for( int j = 0; j < input.cols; j++ )
		{
            double sumX, sumY;
            sumX = 0.0;
            sumY = 0.0;
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
					double kernalXval = kernelX.at<double>( kernelx, kernely );
					double kernalYval = kernelY.at<double>( kernelx, kernely );

					// do the multiplication
					sumX += imageval * kernalXval;
					sumY += imageval * kernalYval;
                }
            }
            WorkX[i][j] = sumX;
            WorkY[i][j] = sumY;
        }
    }

    double minX, maxX, minY, maxY;
    minX = minY = 999999999999;
    maxX = maxY = 0;
	for ( int i = 0; i < input.rows; i++ )
	{
		for( int j = 0; j < input.cols; j++ )
		{
            if (WorkX[i][j] > maxX) {
                maxX = WorkX[i][j];
            }
            if (WorkX[i][j] < minX) {
                minX = WorkX[i][j];
            }
            if (WorkY[i][j] > maxY) {
                maxY = WorkY[i][j];
            }
            if (WorkY[i][j] < minY) {
                minY = WorkY[i][j];
            }

        }
    }


    double x, y;
    double Mag[input.rows][input.cols];

	for ( int i = 0; i < input.rows; i++ )
	{
		for( int j = 0; j < input.cols; j++ )
		{
            x  = (255 * (WorkX[i][j] - minX) / (maxX - minX));
            y  = (255 * (WorkY[i][j] - minY) / (maxY - minY));
            OutputX.at<uchar>(i, j) = (uchar) x;
            OutputY.at<uchar>(i, j) = (uchar) y;

            Mag[i][j] = sqrt((x * x) + (y * y));

            //result = sqrt((WorkX[i][j] * WorkX[i][j]) + (WorkY[i][j] * WorkY[i][j]));
            //y  = (255 * (WorkY[i][j] - minY) / (maxY - minY));
            //Magnitude.at<uchar>(i, j) = (uchar) sqrt((x * x) + (y * y));
        }
    }

    minX = 999999999999;
    maxX = 0;
	for ( int i = 0; i < input.rows; i++ )
	{
		for( int j = 0; j < input.cols; j++ )
		{
            if (Mag[i][j] > maxX) {
                maxX = Mag[i][j];
            }
            if (Mag[i][j] < minX) {
                minX = Mag[i][j];
            }
        }
    }
	for ( int i = 0; i < input.rows; i++ )
	{
		for( int j = 0; j < input.cols; j++ )
		{
            x  = (255 * (Mag[i][j] - minX) / (maxX - minX));
            Magnitude.at<uchar>(i, j) = (uchar) x;
        }
    }

    minX = 0;
    maxX = 2*PI;

    double result;
    for ( int i = 0; i < input.rows; i++ )
    {
        for( int j = 0; j < input.cols; j++ )
        {
            printf("%f - ", WorkX[i][j]);
            result = atan(WorkY[i][j] / WorkX[i][j]);
            if (result < 0) {
                result = -result;
            }
            //x  = (255 * (result - minX) / (maxX - minX));
            //Direction.at<uchar>(i, j) = (uchar) x;
            Direction.at<uchar>(i, j) = (uchar)(180.0 + result / M_PI * 180.0);
        }
    }
}
