// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

const int size = 48;

int median(int[]);

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


    Mat gray_image;
    Mat out_image;
    cvtColor( image, gray_image, CV_BGR2GRAY );
    cvtColor( image, out_image, CV_BGR2GRAY );

    int values[size];
    int count;
    int edge = (size/8);
    int limitC = gray_image.cols-edge;
    int limitR = gray_image.rows-edge;
    int med;

    for ( int i = edge; i < limitR; i++ )
    {
        for( int j = edge; j < limitC; j++ )
        {
            count = 0;

            for(int x = i-edge; x <= i+edge; x++){
                for(int y = j-edge; y <= j+edge ; y++){
                    values[count] = (int) gray_image.at<uchar>(x, y);
                    count++;
                    //printf("%d", count);
                    //]printf("%f - ", (float) gray_image.at<uchar>(x, y));
                }
            }
            //printf("%d-", median(values));

            //printf("%d,%d-", i, j);
            med = median(values);
            out_image.at<uchar>(i, j) = (uchar) med;
        }
    }


    imwrite( "imgs/car2FIX.jpg", out_image );

    return 0;
}

int median(int x[]) {
    float temp;
    int i, j;
    for(i=0; i<size; i++) {
        //printf("%f", x[i]);
    }
    //printf("\n");
    // the following two loops sort the array x in ascending order
    for(i=0; i<size; i++) {
        for(j=0; j<size; j++) {
            if(x[j] < x[i]) {
                // swap elements
                temp = x[i];
                x[i] = x[j];
                x[j] = temp;
            }
        }
    }

    if(size%2==0) {
        // if there is an even number of elements, return mean of the two elements in the middle
        return((x[size/2] + x[size/2 - 1]) / 2.0);
    } else {
        // else return the element in the middle
        return x[size/2];
    }
}
