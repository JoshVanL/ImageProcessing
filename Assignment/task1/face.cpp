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
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;

vector<Rect> groundTruthBoxes;
vector<Rect> faces;


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
    case 4: x.push_back(415); break;
    case 5: x.push_back(92); x.push_back(86); x.push_back(227); x.push_back(271); x.push_back(328); x.push_back(416); x.push_back(456); x.push_back(540); x.push_back(587); x.push_back(675); x.push_back(698); break;
    case 13: x.push_back(469); break;
    case 14: x.push_back(514); x.push_back(778); break;
    case 15: x.push_back(100); x.push_back(408); x.push_back(575); break;
    default: return x;
  }
  return x;
}

std::vector<int> findYCentre(int n) {
  std::vector<int> y;
  switch (n) {
    case 4: y.push_back(195); break;
    case 5: y.push_back(171); y.push_back(285); y.push_back(249); y.push_back(198); y.push_back(279); y.push_back(219); y.push_back(267); y.push_back(206); y.push_back(285); y.push_back(222); y.push_back(285); break;
    case 13: y.push_back(194); break;
    case 14: y.push_back(279); y.push_back(250); break;
    case 15: y.push_back(170); y.push_back(150); y.push_back(168); break;
    default: return y;
  }
  return y;
}

void groundTruth(char* in, Mat frame) {
  Mat image = imread(in,1);
  int num = catchInt(in);
  std::vector<int> x ,y;
  int w, h;
  x = findXCentre(num);
  y = findYCentre(num);
  w = 115;
  h = 115;

  for (int i = 0; i < x.size(); i++) {
    Rect box;
    box.x = x.at(i) - 70;
    box.y = y.at(i) - 70;
    box.width = w;
    box.height = h;
    //rectangle(frame, cvPoint(x.at(i)-w/2,y.at(i)-h/2) , cvPoint(x.at(i)+w/2,y.at(i)+h/2), CV_RGB(255,0,0) ,3);
    rectangle(frame, box, CV_RGB(255,0,0) ,3);

    groundTruthBoxes.push_back(box);
  }

  return;
}

float calculateTPR() {
    // F1 = (2 * detected hit * TPR) / ((TPR * Det) + Actual Hits)

    //F1 = 2 (precision * recall) / (precision + recall)
    //Precision = how many selected items are relevant (Actual hits / detections size)
    //Recall = how relevant items are selected  -- TPR
    // TPR = (Actual Hits / groundTruth size)

    vector<Rect>::iterator groundTruth;
    vector<Rect>::iterator face;
    // F1 = (2 * detected hit * TPR) / ((TPR * Det) + Actual Hits)

    //F1 = 2 (precision * recall) / (precision + recall)
    //Precision = how many selected items are relevant (Actual hits / detections size)
    //Recall = how relevant items are selected  -- TPR
    // TPR = (Actual Hits / groundTruth size)

    float ActualHits = 0;
    float TPR;

    for (face = faces.begin(); face != faces.end(); ++face) {
        for (groundTruth = groundTruthBoxes.begin(); groundTruth != groundTruthBoxes.end(); ++groundTruth) {
            if (abs(face->x - groundTruth->x) < 20 && abs(face->y - groundTruth->y) < 20) {
                ActualHits++;
                continue;
            }
        }
    }

    TPR = ActualHits / groundTruthBoxes.size();
    //precision = ActualHits / dartboards.size();
    //
    printf("Actual faces:%d\nActuat Hits: %f\nTPR: %f\n", faces.size(), ActualHits, TPR);

    return TPR;
}


/** @function main */
int main( int argc, char** argv )
{
       // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay( frame );

    groundTruth(argv[1], frame);

    //calculateTPR();

	// 4. Save Result Image
	imwrite( argv[2], frame );

	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

       // 3. Print number of Faces found
	std::cout << faces.size() << std::endl;

       // 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}

}
