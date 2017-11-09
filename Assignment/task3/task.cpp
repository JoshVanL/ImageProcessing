#include <iostream>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <string>
#include <math.h>

using namespace cv;

#define PI 3.14159265

void convolve (cv::Mat input, cv::Mat kernel, cv::Mat output) {

  double temp[input.rows][input.cols];

  //CREATING A DIFFERENT IMAGE kernel WILL BE NEEDED
	//TO PERFORM OPERATIONS OTHER THAN GUASSIAN BLUR!!!

	// we need to create a padded version of the input
	// or there will be border effects
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

void sobel (cv::Mat &input, cv::Mat &outputX, cv::Mat &outputY, cv::Mat &mag, cv::Mat &dir) {

  float dataX[] = {1,0,-1,2,0,-2,1,0,-1};
  cv::Mat kernelX(3,3, CV_32F, dataX);

  float dataY[] = {1,2,1,0,0,0,-1,-2,-1};
  cv::Mat kernelY(3,3, CV_32F, dataY);

  //find dx and dy
  convolve(input, kernelX, outputX);
  convolve(input, kernelY, outputY);

  double tempMag[input.rows][input.cols];
  //find mag and dir
  for ( int i = 0; i < input.rows; i++ ) {
		for( int j = 0; j < input.cols; j++ ) {
      float temp = outputX.at<uchar>(i,j);
      tempMag[i][j] = sqrt((outputX.at<uchar>(i,j) * outputX.at<uchar>(i,j)) + (outputY.at<uchar>(i,j) * outputY.at<uchar>(i,j)));
      if (temp != 0) {
        dir.at<uchar>(i,j) = atan(outputY.at<uchar>(i,j) / outputX.at<uchar>(i,j)) * 180 / PI;
      }
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

int main( int argc, const char** argv )
{

 Mat image, outputX, outputY, mag, dir;
 Mat grey_image;
 image = imread( argv[1], 1 );
 cvtColor(image, grey_image, CV_BGR2GRAY);

 outputX.create(grey_image.size(), grey_image.type());
 outputY.create(grey_image.size(), grey_image.type());
 mag.create(grey_image.size(), grey_image.type());
 dir.create(grey_image.size(), grey_image.type());

 if(!image.data )
 {
   printf( " No image data \n " );
   return -1;
 }

 sobel(grey_image, outputX, outputY, mag, dir);
 //imwrite("dx.png", outputX);
 //imwrite("dy.png", outputY);
 imwrite(argv[2], mag);
 //imwrite("dir.png", dir);

 return 0;
}
