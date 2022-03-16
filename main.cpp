#include "opencv2/opencv.hpp"
#include <iostream>

std::vector<cv::Rect> detectCar(cv::Mat frame);
std::vector<cv::Rect> detectPeople(cv::Mat frame);
void circleObjects(std::vector<cv::Rect> objects, cv::Mat &frame, cv::Scalar color);

cv::CascadeClassifier peopleCascade;
cv::CascadeClassifier carCascade;

int main(int argc, char *argv[]) {

    //argv[1]
    cv::VideoCapture cap(argv[1], cv::CAP_ANY);
    if (!cap.isOpened()) {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    cv::String people_cascade = cv::samples::findFile("./haarcascade_fullbody.xml");
    cv::String car_cascade = cv::samples::findFile("./cars.xml");

    if (!carCascade.load(car_cascade)) {
        std::cout << "--(!)Error loading face cascade\n";
    }
    if (!peopleCascade.load(people_cascade)) {
        std::cout << "--(!)Error loading face cascade\n";
    }

    cv::Mat frame;
    while (cap.read(frame)) {

        if (frame.empty()) {
            break;
        }

        std::vector<cv::Rect> cars = detectCar(frame);
        std::vector<cv::Rect> people = detectPeople(frame);

        circleObjects(cars, frame, cv::Scalar(0, 255, 0));
        circleObjects(people, frame, cv::Scalar(0, 0, 255));

        cv::resize(frame, frame, cv::Size(1280, 720));
        cv::Mat frame_gray;
        cv::imshow("Frame", frame);

        char c = (char) cv::waitKey(25);
        if (c == 27) {
            break;
        }
    }

    cap.release();

    cv::destroyAllWindows();

    return 0;

}

std::vector<cv::Rect> detectCar(cv::Mat frame) {
    cv::Mat frame_gray;
    cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);
    std::vector<cv::Rect> cars;
    carCascade.detectMultiScale(frame_gray, cars);
    return cars;
}

std::vector<cv::Rect> detectPeople(cv::Mat frame) {
    cv::Mat frame_gray;
    cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);
    std::vector<cv::Rect> people;
    peopleCascade.detectMultiScale(frame_gray, people, 1.1, 5);
    return people;
}

void circleObjects(std::vector<cv::Rect> objects, cv::Mat &frame, cv::Scalar color) {
    for (auto &object: objects) {
//        cv::Point center(object.x + object.width / 2, object.y + object.height / 2);
//        ellipse(frame, center, cv::Size(object.width / 2, object.height / 2), 0, 0, 360, color, 4);
        cv::rectangle(frame, object, color, 2,1,0);
    }
}