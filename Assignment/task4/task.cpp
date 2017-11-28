#include <iostream>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <string>
#include <math.h>
#include <cmath>
#include <vector>

using namespace cv;
using namespace std;

#define PI 3.14159265

/** Global variables */
CascadeClassifier cascade;
RNG rng(12345);

struct DartBoard {
    Rect bounding_box;
};

struct CircleDetection {
    Point2f center;
    int radius;
};

struct Triangle {
    Point points[3];
};

class Detector {
    private:
        Mat mag, grey;
        vector<Point2f> line_hits;
        vector<CircleDetection> circle_hits;
        vector<Rect> viola_hits;
        vector<RotatedRect> ellipse_hits;
        vector<Triangle> triangle_hits;
        vector<DartBoard> dartboards;
        void deduplicate_hits();
        bool rectOverlap(Rect, Rect);
        bool valueInRange(float, float, float);
        float angle(Point2f, Point2f, Point2f);
        vector<double> otsuMethod(Mat);
        void drawLinesAndIntersections(vector<Vec2f>, int, int, int**);
        //void setLabel(Mat, const String, vector<Point>);

    public:
        Mat image, overlay_image, detections_image;
        int read_image (char*);
        int load_cascade (char*);
        void convert_grey();
        Mat convolve(Mat);
        void sobel();
        void houghTransformLine();
        void houghTransformCircle();
        void violaJones();
        void ellipses();
        void combineDectections();
        void write_image_mag(String);
        void write_overlay_image(String);
        void write_detections_image(String);
        void equalize_hist();
        void triangles();
};

Mat Detector::convolve (Mat kernel) {
    Mat output;
    output.create(this->grey.size(), grey.type());

    double temp[this->grey.rows][grey.cols];
    int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
    int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;

    cv::Mat paddedInput;
    cv::copyMakeBorder(this->grey, paddedInput, kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY, BORDER_REPLICATE);

    // now we can do the convoltion
    for ( int i = 0; i < this->grey.rows; i++ ) {
        for( int j = 0; j < this->grey.cols; j++ ) {
            double sum = 0.0;

            for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ ) {
                for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ ) {
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
    for ( int i = 0; i < this->grey.rows; i++ ) {
        for( int j = 0; j < this->grey.cols; j++ ) {
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

    for ( int i = 0; i < this->grey.rows; i++ ) {
        for( int j = 0; j < this->grey.cols; j++ ) {
            output.at<uchar>(i, j) = (255-0)/(max-min)*(temp[i][j]-max)+255;
        }
    }

    return output;
}

void Detector::sobel() {
    cv::Mat outputX, outputY;
    outputX.create(this->grey.size(), grey.type());
    outputY.create(this->grey.size(), grey.type());

    //input -> grey
    this->mag.create(this->grey.size(), grey.type());

    float dataX[] = {1,0,-1,2,0,-2,1,0,-1};
    cv::Mat kernelX(3,3, CV_32F, dataX);

    float dataY[] = {1,2,1,0,0,0,-1,-2,-1};
    cv::Mat kernelY(3,3, CV_32F, dataY);

    //find dx and dy
    outputX = convolve(kernelX);
    outputY = convolve(kernelY);

    double tempMag[this->grey.rows][grey.cols];

    //find this->mag
    for ( int i = 0; i < this->grey.rows; i++ ) {
        for( int j = 0; j < this->grey.cols; j++ ) {
            float temp = outputX.at<uchar>(i,j);
            tempMag[i][j] = sqrt((outputX.at<uchar>(i,j) * outputX.at<uchar>(i,j)) + (outputY.at<uchar>(i,j) * outputY.at<uchar>(i,j)));
        }
    }

    float min, max = 0;
    int tempMin = 999999;
    int tempMax = 0;
    for ( int i = 0; i < this->grey.rows; i++ ) {
        for( int j = 0; j < this->grey.cols; j++ ) {
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

    for ( int i = 0; i < this->grey.rows; i++ ) {
        for( int j = 0; j < this->grey.cols; j++ ) {
            this->mag.at<uchar>(i, j) = (255-0)/(max-min)*(tempMag[i][j]-max)+255;
        }
    }

    return;
}


// Finds the intersection of two lines, or returns false.
// The lines are defined by (o1, p1) and (o2, p2).
bool intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2, Point2f &r) {
    Point2f x = o2 - o1;
    Point2f d1 = p1 - o1;
    Point2f d2 = p2 - o2;

    float cross = d1.x*d2.y - d1.y*d2.x;
    if (abs(cross) < /*EPS*/1e-8) {
        return false;
    }

    double t1 = (x.x * d2.y - x.y * d2.x)/cross;
    r = o1 + d1 * t1;

    return true;
}


void Detector::houghTransformLine() {
    Mat workImage;
    vector<Vec2f> lines;

    vector<double> thresholds = otsuMethod(mag);

    Canny(mag, workImage, thresholds[0], thresholds[1], 3);

    //dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
    //lines: A vector that will store the parameters (r,\theta) of the detected lines
    //rho : The resolution of the parameter r in pixels. We use 1 pixel.
    //theta: The resolution of the parameter \theta in radians. We use 1 degree (CV_PI/180)
    //threshold: The minimum number of intersections to “detect” a line
    //srn and stn: Default parameters to zero. Check OpenCV reference for more info.
    HoughLines(workImage, lines, 1, CV_PI/180, 100, 0, 0 );

    const int x = this->mag.rows;
    const int y = this->mag.cols;

    //int crosses[x][y];
    int **crosses;
    float sum = 0;

    crosses = new int *[x];
    for(int i = 0; i <x; i++)
        crosses[i] = new int[y];

    for ( int i = 0; i < this->mag.rows; i++ ) {
        for( int j = 0; j < this->mag.cols; j++ ) {
            crosses[i][j] = 0;
        }
    }

    for( size_t i = 0; i < lines.size(); i++ ) {
        for( size_t j = i+1; j < lines.size(); j++ ) {
            float rho1 = lines[i][0], theta1 = lines[i][1];
            Point pt11, pt12;
            double a1 = cos(theta1), b1 = sin(theta1);
            double x10 = a1*rho1, y10 = b1*rho1;

            float rho2 = lines[j][0], theta2 = lines[j][1];
            Point pt21, pt22;
            double a2 = cos(theta2), b2 = sin(theta2);
            double x20 = a2*rho2, y20 = b2*rho2;

            pt11.x = cvRound(x10 + 1000*(-b1));
            pt11.y = cvRound(y10 + 1000*(a1));
            pt12.x = cvRound(x10 - 1000*(-b1));
            pt12.y = cvRound(y10 - 1000*(a1));

            pt21.x = cvRound(x20 + 1000*(-b2));
            pt21.y = cvRound(y20 + 1000*(a2));
            pt22.x = cvRound(x20 - 1000*(-b2));
            pt22.y = cvRound(y20 - 1000*(a2));

            Point2f r;

            if (intersection(pt11, pt12, pt21, pt22, r)) {
                if (r.x < workImage.cols && r.y < workImage.rows && r.x > 0 && r.y > 0) {
                    int x = r.y;
                    int y = r.x;
                    crosses[x][y] += 1;
                    sum ++;
                }
            }
        }
    }

    for ( int i = 0; i < workImage.rows; i++ ) {
        for( int j = 0; j < workImage.cols; j++ ) {
            // TODO: maybe change this to == 8 or some other
            if (crosses[i][j] > (sum / (4 * lines.size()))) {
            //if (crosses[i][j] > 0) {
                Point2f center;
                center.x = j;
                center.y = i;
                circle( this->overlay_image, center, 7, Scalar(30,255,30), -1, 8, 0 );
                line_hits.push_back(center);
            }
        }
    }

    //drawLinesAndIntersections(lines, x, y, crosses);

    return;
}

void Detector::drawLinesAndIntersections(vector<Vec2f> lines, int x, int y, int **crosses) {
    for( size_t i = 0; i < lines.size(); i++ ) {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;

        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));

        line( this->overlay_image, pt1, pt2, Scalar(0,0,255), 1, 8 );
    }

    for ( int i = 0; i < x; i++ ) {
        for( int j = 0; j < y; j++ ) {
            if (crosses[i][j] > 0) {
                Point2f center;
                center.x = j;
                center.y = i;
                circle( this->overlay_image, center, 1, Scalar(0,255, 0), -1, 8, 0 );
            }
        }
    }

}


void Detector::houghTransformCircle() {
    Mat blur_image;
    vector<Vec3f> circles;

    GaussianBlur(mag, blur_image, Size(9, 9), 1.5, 1.5 );

    //src_gray: Input image (grayscale)
    //circles: A vector that stores sets of 3 values: x_{c}, y_{c}, r for each detected circle.
    //CV_HOUGH_GRADIENT: Define the detection method. Currently this is the only one available in OpenCV
    //dp = 1: The inverse ratio of resolution
    //min_dist = src_gray.rows/8: Minimum distance between detected centers
    //param_1 = 200: Upper threshold for the internal Canny edge detector
    //param_2 = 100*: Threshold for center detection.
    //min_radius = 0: Minimum radio to be detected. If unknown, put zero as default.
    //max_radius = 0: Maximum radius to be detected. If unknown, put zero as default
    HoughCircles(blur_image, circles, CV_HOUGH_GRADIENT, 1, this->overlay_image.rows/16, 100, 35, 10, 120);

    for( size_t i = 0; i < circles.size(); i++ ) {
        Point2f center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        circle( this->overlay_image, center, 3, Scalar(0,0,255), -1, 8, 0 );
        circle( this->overlay_image, center, radius, Scalar(0,0,255), 3, 8, 0 );

        CircleDetection circle_hit = {center, radius};
        circle_hits.push_back(circle_hit);
    }

    return;
}

void Detector::ellipses() {
    Mat threshold_output;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    /// Detect edges using Threshold
    threshold( this->mag, threshold_output, 151, 255, THRESH_BINARY );
    /// Find contours
    findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    /// Find the rotated rectangles and ellipses for each contour
    vector<RotatedRect> minRect( contours.size() );
    vector<RotatedRect> ellipses( contours.size() );

    for( int i = 0; i < contours.size(); i++ ) {
        if( contours[i].size() > 100 ) {
            minRect[i] = minAreaRect( Mat(contours[i]) );
            ellipses[i] = fitEllipse( Mat(contours[i]) );
        }
    }

    /// Draw contours + rotated rects + ellipses
    Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
    for( int i = 0; i< contours.size(); i++ ) {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        // ellipse
        ellipse( this->overlay_image, ellipses[i], Scalar(226, 43 ,138 ), 6, 8 );

        // rotated rectangle
        //Point2f rect_points[4];
        //minRect[i].points( rect_points );
        //for( int j = 0; j < 4; j++ ) {
        //    line( this->overlay_image, rect_points[j], rect_points[(j+1)%4], Scalar(226,43,138), 2, 8 );
        //}
    }

    ellipse_hits = ellipses;

    return;
}

void setLabel(cv::Mat& im, const std::string label, std::vector<cv::Point>& contour) {
	int fontface = cv::FONT_HERSHEY_SIMPLEX;
	double scale = 0.4;
	int thickness = 1;
	int baseline = 0;

	cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
	cv::Rect r = cv::boundingRect(contour);

	cv::Point pt(r.x + ((r.width - text.width) / 2), r.y + ((r.height + text.height) / 2));
	cv::rectangle(im, pt + cv::Point(0, baseline), pt + cv::Point(text.width, -text.height), CV_RGB(255,255,255), CV_FILLED);
	cv::putText(im, label, pt, fontface, scale, CV_RGB(0,0,0), thickness, 8);
}


float triangle_area(Point p0, Point p1, Point p2) {
    float dArea = ((p1.x - p0.x)*(p2.y - p0.y) - (p2.x - p0.x)*(p1.y - p0.y))/2.0;
    return (dArea > 0.0) ? dArea : -dArea;
}

double angleABC(Point pt1, Point pt2, Point pt0)
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

bool acute_triangle(vector<Point> triangle) {
    for (int i = 0; i < 3; i++) {
        if (angleABC(triangle[i], triangle[(i+1)%3], triangle[(i+2)%3]) > 90) {
            return false;
        }
    }
    return true;
}

vector<double> Detector::otsuMethod(Mat input) {
    Mat output;
    double high_thresh, low_thresh;
    high_thresh = threshold(input, output, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    low_thresh = high_thresh * 0.5;

    return vector<double>{low_thresh, high_thresh};
}

void Detector::triangles() {
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    Mat canny_output, blur_image, bw;

    equalizeHist(this->grey, this->grey);
    blur( this->grey, grey, Size( 3, 3 ) );

    vector<double> thresholds;
    thresholds = this->otsuMethod(this->grey);
    Canny(this->grey, canny_output, thresholds[0], thresholds[1]);

    findContours(canny_output, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    vector<Point> approxTriangle;
    for(size_t i = 0; i < contours.size(); i++){
        approxPolyDP(contours[i], approxTriangle, 4, true);
        if(approxTriangle.size() == 3 && fabs(contourArea(contours[i])) > 80 && acute_triangle(approxTriangle)){

            Triangle triangle;
            for (int k = 0; k < 3; k ++) {
                triangle.points[k] = approxTriangle[k];
            }
            this->triangle_hits.push_back(triangle);

            drawContours(this->overlay_image, contours, i, Scalar(80, 80 ,255 ), CV_FILLED); // fill GREEN
        }
    }

    return;
}

float Detector::angle( Point2f a, Point2f b, Point2f c ) {
    Point2f ab = { b.x - a.x, b.y - a.y };
    Point2f cb = { b.x - c.x, b.y - c.y };

    float dot = (ab.x * cb.x + ab.y * cb.y);

    float abSqr = ab.x * ab.x + ab.y * ab.y;
    float cbSqr = cb.x * cb.x + cb.y * cb.y;

    float cosSqr = dot * dot / abSqr / cbSqr;

    float cos2 = 2 * cosSqr - 1;

    // Here's the only invocation of the heavy function.
    // It's a good idea to check explicitly if cos2 is within [-1 .. 1] range

    const float pi = 3.141592f;

    float alpha2 =
        (cos2 <= -1) ? pi :
        (cos2 >= 1) ? 0 :
        acosf(cos2);

    float rslt = alpha2 / 2;

    float rs = rslt * 180. / pi;


    // Now revolve the ambiguities.
    // 1. If dot product of two vectors is negative - the angle is definitely
    // above 90 degrees. Still we have no information regarding the sign of the angle.

    // NOTE: This ambiguity is the consequence of our method: calculating the cosine
    // of the double angle. This allows us to get rid of calling sqrt.

    if (dot < 0)
        rs = 180 - rs;

    // 2. Determine the sign. For this we'll use the Determinant of two vectors.

    float det = (ab.x * cb.y - ab.y * cb.y);
    if (det < 0)
        rs = -rs;

    return floor(rs + 0.5);
}


void Detector::violaJones() {
    cascade.detectMultiScale(this->grey, viola_hits, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

    for( int i = 0; i < viola_hits.size(); i++ ) {
        rectangle(this->overlay_image, Point2f(viola_hits[i].x, viola_hits[i].y), Point2f(viola_hits[i].x + viola_hits[i].width, viola_hits[i].y + viola_hits[i].height), Scalar( 255, 0, 0 ), 2);
    }

    return;
}

void Detector::combineDectections() {
    vector<Rect>::iterator viola;
    vector<Point2f>::iterator line_hit;
    vector<CircleDetection>::iterator circle_hit;
    vector<RotatedRect>::iterator ellipse_hit;
    vector<Triangle>::iterator triangle_hit;

    for (viola = viola_hits.begin(); viola!= viola_hits.end(); ++viola) {
        vector<Point2f> line_score;
        vector<CircleDetection> circle_score;
        vector<RotatedRect> ellipse_score;
        vector<Triangle> triangle_score;

        for (line_hit = line_hits.begin(); line_hit != line_hits.end(); ++line_hit) {
            if (line_hit->x >= viola->x && line_hit->y >= viola->y && line_hit->x <= (viola->x + viola->width) && line_hit->y <= (viola->y + viola->height)) {
                line_score.push_back(*line_hit);
            }
        }

        for (circle_hit = circle_hits.begin(); circle_hit != circle_hits.end(); ++circle_hit) {
            if (circle_hit->center.x >= viola->x && circle_hit->center.y >= viola->y && circle_hit->center.x <= (viola->x + viola->width) && circle_hit->center.y <= (viola->y + viola->height)) {
                circle_score.push_back(*circle_hit);
            }
        }

        for (ellipse_hit = ellipse_hits.begin(); ellipse_hit != ellipse_hits.end(); ++ellipse_hit) {
            if (ellipse_hit->center.x >= viola->x && ellipse_hit->center.y >= viola->y && ellipse_hit->center.x <= (viola->x + viola->width) && ellipse_hit->center.y <= (viola->y + viola->height)) {
                ellipse_score.push_back(*ellipse_hit);
            }
        }

        //TODO: Fix this, it's broken
        for (triangle_hit = triangle_hits.begin(); triangle_hit != triangle_hits.end(); ++triangle_hit) {
            bool inside = true;
            for (int i = 0; i < 3; i++) {
                if (triangle_hit->points[i].x < viola->x || triangle_hit->points[i].y < viola->y || triangle_hit->points[i].x > (viola->x + viola->width) || triangle_hit->points[i].y > (viola->y + viola->height)) {
                        inside = false;
                        break;
                }
            }
            if (inside) {
                triangle_score.push_back(*triangle_hit);
            }
        }

        if(line_score.empty() && circle_score.empty() && ellipse_score.empty() && triangle_score.empty()) {
            continue;
        }

        float avx, avy, avwidth, avheight;
        avx = avy = avwidth = avheight = 0;

        for (line_hit = line_score.begin(); line_hit != line_score.end(); ++line_hit) {
            avx += line_hit->x;
            avy += line_hit->y;
        }
        for (circle_hit = circle_score.begin(); circle_hit != circle_score.end(); ++circle_hit) {
            avx += circle_hit->center.x;
            avy += circle_hit->center.y;
            avwidth += (circle_hit->radius * 2);
            avheight += (circle_hit->radius * 2);
        }
        for (ellipse_hit = ellipse_score.begin(); ellipse_hit != ellipse_score.end(); ++ellipse_hit) {
            avx += ellipse_hit->center.x;
            avy += ellipse_hit->center.y;
            avwidth += (ellipse_hit->size.width / 2);
            avheight += (ellipse_hit->size.height / 2);
        }
        for (triangle_hit = triangle_score.begin(); triangle_hit != triangle_score.end(); ++triangle_hit) {
            avx += ((triangle_hit->points[0].x + triangle_hit->points[1].x + triangle_hit->points[2].x) / 3);
            avy += ((triangle_hit->points[0].y + triangle_hit->points[1].y + triangle_hit->points[2].y) / 3);
        }

        avx = avx / (line_score.size() + circle_score.size() + ellipse_score.size() + triangle_score.size());
        avy = avy / (line_score.size() + circle_score.size() + ellipse_score.size() + triangle_score.size());
        avwidth = (viola->width + avwidth) / (circle_score.size() + ellipse_score.size() + 1);
        avheight = (viola->height + avheight) / (circle_score.size() + ellipse_score.size() + 1);

        DartBoard dartboard;
        dartboard.bounding_box.x = avx - (avwidth / 2);
        dartboard.bounding_box.y = avy - (avheight / 2);
        dartboard.bounding_box.width = avwidth;
        dartboard.bounding_box.height = avheight;
        dartboards.push_back(dartboard);
    }

    deduplicate_hits();

    return;
}

void Detector::deduplicate_hits() {
    //Deep copy dartboards
    //Loop till clone is empty
        //Loop through clone, get next box
        //Loop through clone, find all that are overlapping, remove and average them -> add to new set
    //Use new deduplicated set

    vector<DartBoard>dartboards_clone = dartboards;
    vector<DartBoard>deduplicated_hits;
    DartBoard dartboard;
    DartBoard current;

    while (dartboards_clone.size() > 0) {
        float distance;
        float avx, avy, avwidth, avheight;
        int sum_duplicates = 1;

        current = dartboards_clone.back();
        avx = current.bounding_box.x;
        avy = current.bounding_box.y;
        avwidth = current.bounding_box.width;
        avheight = current.bounding_box.height;
        dartboards_clone.erase(dartboards_clone.end() -1);

        for(unsigned index = dartboards_clone.size(); index-- > 0;) {

            dartboard = dartboards_clone.at(index);

            if (rectOverlap(current.bounding_box, dartboard.bounding_box)) {
                sum_duplicates++;
                avx += dartboard.bounding_box.x;
                avy += dartboard.bounding_box.y;
                avwidth += dartboard.bounding_box.width;
                avheight += dartboard.bounding_box.height;

                //current = dartboard;
                dartboards_clone.erase(dartboards_clone.begin() + index);
                continue;
            }
        }

        dartboard.bounding_box.x = avx / sum_duplicates;
        dartboard.bounding_box.y = avy / sum_duplicates;
        dartboard.bounding_box.width = avwidth / sum_duplicates;
        dartboard.bounding_box.height = avheight / sum_duplicates;
        deduplicated_hits.push_back(dartboard);
    }

    dartboards = deduplicated_hits;

    return;
}

bool Detector::valueInRange(float value, float min, float max) {
    return (value >= min) && (value <= max);
}

bool Detector::rectOverlap(Rect A, Rect B) {
    bool xOverlap = valueInRange(A.x, B.x, B.x + B.width) || valueInRange(B.x, A.x, A.x + A.width);

    bool yOverlap = valueInRange(A.y, B.y, B.y + B.height) || valueInRange(B.y, A.y, A.y + A.height);

    return xOverlap && yOverlap;
}

int Detector::read_image (char* path) {
    image = imread(path, CV_LOAD_IMAGE_COLOR);
    if(!image.data){
        printf("No image data \n");
        return -1;
    }

    this->overlay_image = image.clone();
    detections_image = image.clone();

    return 0;
}

int Detector::load_cascade(char* path) {
    if(!cascade.load(path)) {
        printf("Could no load cascade xml\n");
        return -1;
    }

    return 0;
}

void Detector::write_image_mag(String path) {
    imwrite(path, this->mag);

    return;
}

void Detector::convert_grey() {
    cvtColor(image, this->grey, CV_BGR2GRAY);

    return;
}

void Detector::equalize_hist() {
    equalizeHist(this->grey, this->grey);

    return;
}

void Detector::write_overlay_image(String path) {
    imwrite(path, this->overlay_image);

    return;
}

void Detector::write_detections_image(String path) {
    vector<DartBoard>::iterator dartboard;
    for (dartboard = dartboards.begin(); dartboard != dartboards.end(); ++dartboard) {
	    rectangle(detections_image, Point(dartboard->bounding_box.x, dartboard->bounding_box.y), Point(dartboard->bounding_box.x + dartboard->bounding_box.width, dartboard->bounding_box.y + dartboard->bounding_box.height), Scalar( 0, 250, 0 ), 4);
    }

    imwrite(path, detections_image);

    return;
}

int main( int argc, char** argv ) {
    Detector detector;

    if (argc != 4) {
        printf("enter [input image] [output image] [cascade file]\n");
        return -1;
    }
    if (detector.read_image(argv[1])) {
        return -1;
    }
    if (detector.load_cascade(argv[3])) {
        return -1;
    }

    detector.convert_grey();
    //detector.equalize_hist();
    //detector.equalize_hist();
    detector.sobel();
    detector.houghTransformCircle();

    detector.houghTransformLine();

    detector.violaJones();

    detector.triangles();

    //detector.ellipses();

    detector.combineDectections();


    //detector.write_overlay_image(argv[2]);
    detector.write_detections_image(argv[2]);

    return 0;
}
