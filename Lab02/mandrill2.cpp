#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

int main( int argc, char** argv ) {

    char* imageName = argv[1];

    Mat inImage;
    inImage = imread( imageName, 1 );

    if( argc != 2 || !inImage.data ) {
        printf( " No image data \n " );
        return -1;
    }

    cvtColor( inImage, inImage, CV_HSV2BGR );

    imwrite( "reverse.jpg", inImage );

    return 0;

}
