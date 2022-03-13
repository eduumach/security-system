#include "opencv2/opencv.hpp"
#include <iostream>

std::vector<cv::Rect> detectCar(cv::Mat frame);
std::vector<cv::Rect> detectPeople(cv::Mat frame);
void circleObjects(std::vector<cv::Rect> objects, cv::Mat frame);

cv::CascadeClassifier cascadeClassifier;

int main(int argc, char *argv[]) {

    //argv[1]
    cv::VideoCapture cap(argv[1], cv::CAP_ANY);
    if (!cap.isOpened()) {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (cap.read(frame)) {

        if (frame.empty()) {
            break;
        }

        std::vector<cv::Rect> objects;
        std::vector<cv::Rect> cars = detectCar(frame);
        std::vector<cv::Rect> peoples = detectPeople(frame);

        objects.insert(objects.end(), make_move_iterator(cars.begin()),
                       make_move_iterator(cars.end()));
        objects.insert(objects.end(), std::make_move_iterator(peoples.begin()),
                       std::make_move_iterator(peoples.end()));

        circleObjects(objects, frame);


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
    cv::String car_cascade = cv::samples::findFile("./cars.xml");
    if (!cascadeClassifier.load(car_cascade)) {
        std::cout << "--(!)Error loading face cascade\n";
    }
    cv::Mat frame_gray;
    cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);
    std::vector<cv::Rect> cars;
    cascadeClassifier.detectMultiScale(frame_gray, cars);
    return cars;
}

std::vector<cv::Rect> detectPeople(cv::Mat frame) {
    cv::String people_cascade = cv::samples::findFile("./haarcascade_fullbody.xml");
    if (!cascadeClassifier.load(people_cascade)) {
        std::cout << "--(!)Error loading face cascade\n";
    }
    cv::Mat frame_gray;
    cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);
    std::vector<cv::Rect> peoples;
    cascadeClassifier.detectMultiScale(frame_gray, peoples);
    return peoples;
}

void circleObjects(std::vector<cv::Rect> objects, cv::Mat frame) {
    for (auto &face: objects) {
        cv::Point center(face.x + face.width / 2, face.y + face.height / 2);
        ellipse(frame, center, cv::Size(face.width / 2, face.height / 2), 0, 0, 360, cv::Scalar(255, 0, 255), 4);
    }
    resize(frame, frame, cv::Size(1280, 720));
    imshow("Frame", frame);
}