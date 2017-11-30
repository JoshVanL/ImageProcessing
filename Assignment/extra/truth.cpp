#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>
#include <string>
#include <iostream>  //depending on your machine setup

using namespace cv;

int catchInt(string s) {
  char *p = new char[s.length()+1];
  strcpy(p,s.c_str());
  while (*p) {
    if (isdigit(*p)) {
      int val = strtol(p,&p,10);
      return val;
    }
    else{
      p++;
    }
  }
  return 0;
}

std::vector<int> findXCentre(int n) {
  std::vector<int> x;
  switch (n) {
    case 0: x.push_back(515); break;
    case 1: x.push_back(290); break;
    case 2: x.push_back(147); break;
    case 3: x.push_back(356); break;
    case 4: x.push_back(285); break;
    case 5: x.push_back(482); break;
    case 6: x.push_back(241); break;
    case 7: x.push_back(321); break;
    case 8: x.push_back(894); x.push_back(90); break;
    case 9: x.push_back(318); break;
    case 10: x.push_back(136); x.push_back(610); x.push_back(933);break;
    case 11: x.push_back(205); break;
    case 12: x.push_back(184); break;
    case 13: x.push_back(335); break;
    case 14: x.push_back(177); x.push_back(1048); break;
    case 15: x.push_back(219); break;
    default: return x;
  }
  return x;
}

std::vector<int> findYCentre(int n) {
  std::vector<int> y;
  switch (n) {
    case 0: y.push_back(100); break;
    case 1: y.push_back(225); break;
    case 2: y.push_back(141); break;
    case 3: y.push_back(183); break;
    case 4: y.push_back(193); break;
    case 5: y.push_back(193); break;
    case 6: y.push_back(149); break;
    case 7: y.push_back(241); break;
    case 8: y.push_back(276); y.push_back(294); break;
    case 9: y.push_back(163); break;
    case 10: y.push_back(157); y.push_back(170); y.push_back(184);break;
    case 11: y.push_back(142); break;
    case 12: y.push_back(145); break;
    case 13: y.push_back(187); break;
    case 14: y.push_back(164); y.push_back(155); break;
    case 15: y.push_back(121); break;
    default: return y;
  }
  return y;
}

Mat groundTruth(string in) {
  Mat image = imread(in,1);
  int num = catchInt(in);
  std::vector<int> x ,y;
  int w, h;
  x = findXCentre(num);
  y = findYCentre(num);
  w = 100;
  h = 100;

  for (int i = 0; i < x.size(); i++) {
    cv::rectangle(image, cvPoint(x.at(i)-w/2,y.at(i)-h/2) , cvPoint(x.at(i)+w/2,y.at(i)+h/2), CV_RGB(255,0,0) ,3);
  }

  return image;
}

int main( int argc, char** argv ) {
  if (argc != 3) {
      printf("enter [input image] [output image]\n");
      return -1;
  }
  string in = argv[1];
  string out = argv[2];

  Mat outImage = groundTruth(in);

  imwrite(out, outImage);
    return 0;
}
