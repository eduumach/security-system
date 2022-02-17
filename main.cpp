#include "opencv2/opencv.hpp"
#include <iostream>


int main(int argc, char *argv[])
{

  cv::VideoCapture cap;

#ifdef WIN32
  cap.open(argv[1], cv::CAP_DSHOW);

#else

cap.open(argv[1], cv::CAP_V4L2);
#endif

  if (!cap.isOpened())
  {
    std::cout << "Error opening video stream or file" << std::endl;
    return -1;
  }

  while (cap.isOpened())
  {
    cv::Mat frame;
    // Capture frame-by-frame
    cap >> frame;

    // If the frame is empty, break immediately
    if (frame.empty())
    {
      break;
    }

    // Display the resulting frame
    cv::imshow("Frame", frame);

    // Press  ESC on keyboard to exit
    char c = (char)cv::waitKey(25);
    if (c == 27)
    {
      break;
    }
  }

  std::cerr << "CAMERA CAIU" << std::endl;

  // When everything done, release the video capture object
  cap.release();

  // Closes all the frames
  cv::destroyAllWindows();

  return 0;
}