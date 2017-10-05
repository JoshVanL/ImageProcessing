#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

int main( int argc, char** argv ) {

    char* imageName = argv[1];

    Mat image;
    image = imread( imageName, 1 );

    if( argc != 2 || !image.data ) {
        printf( " No image data \n " );
        return -1;
    }

    for(int y=0; y<image.rows; y++) {
        for(int x=0; x<image.cols; x++) {
            uchar pixelBlue = image.at<Vec3b>(y, x)[0];
            uchar pixelGreen = image.at<Vec3b>(y, x)[1];
            uchar pixelRed = image.at<Vec3b>(y, x)[2];
            image.at<Vec3b>(y, x)[2] = pixelGreen;
            image.at<Vec3b>(y, x)[0] = pixelRed;
            image.at<Vec3b>(y, x)[1] = pixelBlue;
        }
    }

    imwrite( "rev.jpg", image );

    return 0;

}
