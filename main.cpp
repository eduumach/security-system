#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

void detectAndDisplay( Mat frame );

CascadeClassifier face_cascade;

int main(int argc, char* argv[]){

  //argv[1]
  VideoCapture cap(argv[1], cv::CAP_ANY); 
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }
  String face_cascade_name = samples::findFile("data/haarcascades/haarcascade_frontalface_default.xml");

  if( !face_cascade.load( face_cascade_name ) )
  {
      cout << "--(!)Error loading face cascade\n";
      return -1;
  };

  Mat frame;
  while(cap.read(frame)){

    if (frame.empty()){
      break;
    }

    detectAndDisplay(frame);
    

    char c=(char)waitKey(25);
    if(c==27){
      break;
    }
  }
 
  cap.release();

  destroyAllWindows();
  
  return 0;

}

void detectAndDisplay(Mat frame){
  Mat frame_gray;
  cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
  equalizeHist(frame_gray, frame_gray);
  vector<Rect> faces;
  face_cascade.detectMultiScale(frame_gray, faces);

  for ( size_t i = 0; i < faces.size(); i++ ){
    Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
    ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4 );
    Mat faceROI = frame_gray( faces[i] );
  }
  resize(frame,frame, Size(1280,720));
  imshow( "Frame", frame );
}