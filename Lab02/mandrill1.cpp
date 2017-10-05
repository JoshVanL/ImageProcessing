#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

int main( int argc, char** argv ) {

    char* imageName = argv[1];

    Mat inImage;
    Mat outImage(512, 512, CV_8UC3, Scalar(0, 0, 0));
    inImage = imread( imageName, 1 );
    int size = inImage.size().width;

    if( argc != 2 || !inImage.data ) {
        printf( " No image data \n " );
        return -1;
    }

    for(int y=0; y<inImage.rows; y++) {
        for(int x=0; x<inImage.cols; x++) {
            outImage.at<Vec3b>(((y+32)%size), ((x+32)%size))[2] = inImage.at<Vec3b>(y, x)[2];
            outImage.at<Vec3b>(y, x)[1] = inImage.at<Vec3b>(y, x)[1];
            outImage.at<Vec3b>(y, x)[0] = inImage.at<Vec3b>(y, x)[0];
        }
    }

    imwrite( "redMove.jpg", outImage );

    return 0;

}
