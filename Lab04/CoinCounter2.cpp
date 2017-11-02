// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <math.h>       /* atan */

#define PI 3.14159265

using namespace cv;

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
 Mat out;
 cvtColor( image, gray_image, CV_BGR2GRAY );
 cvtColor( image, out, CV_BGR2GRAY );


 for ( int i = 0; i < gray_image.rows; i++ )
 {
    for( int j = 0; j < gray_image.cols; j++ )
    {
        if (gray_image.at<uchar>(i, j) < 200) {
            out.at<uchar>(i, j) = 0;
        } else {
            out.at<uchar>(i, j) = 155;
        }

    }
 }

 imwrite( "imgs/out.png", out );

 return 0;
}
