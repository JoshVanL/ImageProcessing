/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );

/** Global variables */
String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;

vector<Rect> groundTruthBoxes;
vector<Rect> dartboards;

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

vector<int> findXCentre(int n) {
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
    case 11: x.push_back(205); x.push_back(464); break;
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
    case 11: y.push_back(142); y.push_back(149); break;
    case 12: y.push_back(145); break;
    case 13: y.push_back(187); break;
    case 14: y.push_back(164); y.push_back(155); break;
    case 15: y.push_back(121); break;
    default: return y;
  }
  return y;
}

Mat groundTruth(char* in, Mat frame) {
  Mat image = imread(in,1);
  int num = catchInt(in);
  std::vector<int> x ,y;
  int w, h;
  x = findXCentre(num);
  y = findYCentre(num);
  w = 200;
  h = 200;

  for (int i = 0; i < x.size(); i++) {
    Rect box;
    box.x = x.at(i);
    box.y = y.at(i);
    box.width = w;
    box.height = h;
    rectangle(frame, cvPoint(x.at(i)-w/2,y.at(i)-h/2) , cvPoint(x.at(i)+w/2,y.at(i)+h/2), CV_RGB(255,0,0) ,3);

    groundTruthBoxes.push_back(box);
  }

  return image;
}

float calculateF1Score() {
    vector<Rect>::iterator groundTruth;
    vector<Rect>::iterator dartboard;
    // F1 = (2 * detected hit * TPR) / ((TPR * Det) + Actual Hits)

    //F1 = 2 (precision * recall) / (precision + recall)
    //Precision = how many selected items are relevant (Actual hits / detections size)
    //Recall = how relevant items are selected  -- TPR
    // TPR = (Actual Hits / groundTruth size)

    float ActualHits = 0;
    float precision, TPR;

    for (groundTruth = groundTruthBoxes.begin(); groundTruth != groundTruthBoxes.end(); ++groundTruth) {
        for (dartboard = dartboards.begin(); dartboard != dartboards.end(); ++dartboard) {
            if (fabs(dartboard->x - groundTruth->x) < 100 && fabs(dartboard->y - groundTruth->y) < 100 && fabs((dartboard->x + dartboard->width) - (groundTruth->x + groundTruth->width)) < 100 && fabs((dartboard->y + dartboard->height) - (groundTruth->y + groundTruth->height)) < 100) {
                ActualHits++;
                break;
            }
        }
    }

    printf("Ground truth: %d\n", groundTruthBoxes.size());
    printf("Actual hits: %f\n", ActualHits);
    printf("DartBoards: %d\n", dartboards.size());
    //TPR = ActualHits / groundTruthBoxes.size();
    precision = ActualHits / dartboards.size();

    if (TPR == 0 && precision == 0 ) {
        return 0;
    }
    return (TPR * precision * 2) / (TPR + precision);
}


/** @function main */
int main( int argc, char** argv )
{
       // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Dart boards and Display Result
	detectAndDisplay( frame );

    groundTruth(argv[1], frame);
    float f1 = calculateF1Score();
    printf("%s - %f\n", argv[1], f1);

	// 4. Save Result Image
	imwrite( argv[2], frame );

	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
	std::vector<Rect> boards;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, boards, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

       // 3. Print number of Faces found
	std::cout << boards.size() << std::endl;

       // 4. Draw box around boards found
	for( int i = 0; i < boards.size(); i++ )
	{
        dartboards.push_back(boards[i]);
		rectangle(frame, Point(boards[i].x, boards[i].y), Point(boards[i].x + boards[i].width, boards[i].y + boards[i].height), Scalar( 0, 255, 0 ), 2);
	}

}
