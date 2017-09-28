#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

int main() {

  // Read image from file
  Mat image = imread("mandrill.jpg", 1);

  // Convert to grey scale
  Mat gray_image1, gray_image2;
  cvtColor(image, gray_image1, CV_BGR2GRAY);
  cvtColor(image, gray_image2, CV_BGR2GRAY);

  // Threshold by looping through all pixels
    for (int y = 0; y < gray_image1.rows; y++) {
        for (int x = 0; x < gray_image1.cols; x++) {

            uchar pixel = gray_image1.at<uchar>(y, x);
            if (pixel>100) {
                gray_image1.at<uchar>(y, x) = 0;
            } else  {
                gray_image1.at<uchar>(y, x) = 250;
            }

        }
    }

    threshold(gray_image2, gray_image2, 100, 255, THRESH_BINARY_INV);


  //Save thresholded image
  imwrite("thr1.jpg", gray_image1);
  imwrite("thr2.jpg", gray_image2);

  return 0;
}
