#include <fstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

float confThreshold = 0.5;
float nmsThreshold = 0.4;
int inpWidth = 416;
int inpHeight = 416;
std::vector<std::string> classes;


void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& out);
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);
std::vector<cv::String> getOutputsNames(const cv::dnn::Net& net);

int main(int argc, char *argv[]){

    std::string classesFile = "coco.names";
    std::ifstream ifs(classesFile.c_str());
    std::string line;
    while (getline(ifs, line)) {
        classes.push_back(line);
    }

    cv::String modelConfiguration = "yolov3-tiny.cfg";
    cv::String modelWeights = "yolov3-tiny.weights";

    cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);

//  cpu
    std::cout << "Using CPU device" << std::endl;
    net.setPreferableBackend(cv::dnn::DNN_TARGET_CPU);

//    gpu
//    std::cout << "Using GPU device" << std::endl;
//    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
//    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    cv::VideoWriter video;
    cv::Mat frame, blob;

    cv::VideoCapture cap(argv[1], cv::CAP_ANY);
    if (!cap.isOpened()) {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    static const std::string kWinName = "Security System!";
    namedWindow(kWinName, cv::WINDOW_NORMAL);

    while (cv::waitKey(1) < 0) {

        cap >> frame;

        cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(inpWidth, inpHeight), cv::Scalar(0,0,0), true, false);

        net.setInput(blob);

        std::vector<cv::Mat> outs;
        net.forward(outs, getOutputsNames(net));

        postprocess(frame, outs);

        std::vector<double> layersTimes;
        double freq = cv::getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        std::string label = cv::format("Inference time for a frame : %.2f ms", t);
        putText(frame, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));


        cv::Mat detectedFrame;
        frame.convertTo(detectedFrame, CV_8U);
        video.write(detectedFrame);

        imshow(kWinName, frame);


    }

}

void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs){
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for(size_t i = 0; i < outs.size(); ++i){
        float* data = (float*)outs[i].data;
        for(int j = 0; j < outs[i].rows; ++j, data += outs[i].cols){
            cv::Mat scores = outs[i].row(j).colRange(5,outs[i].cols);
            cv::Point classIdPoint;
            double confidence;
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if(confidence > confThreshold){
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for(size_t i = 0; i < indices.size(); ++i){
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
}

void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame){

    rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 178, 50), 3);

    std::string label = cv::format("%.2f", conf);
    if (!classes.empty()){
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    int baseLine;
    cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = std::max(top, labelSize.height);
    rectangle(frame, cv::Point(left, top - round(1.5*labelSize.height)), cv::Point(left + round(1.5*labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
    putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,0),1);

}

std::vector<cv::String> getOutputsNames(const cv::dnn::Net& net){
    static std::vector<cv::String> names;
    if(names.empty()){
        std::vector<int> outLayers = net.getUnconnectedOutLayers();

        std::vector<cv::String> layersNames = net.getLayerNames();

        names.resize(outLayers.size());

        for(size_t i = 0; i < outLayers.size(); ++i){
            names[i] = layersNames[outLayers[i] - 1];
        }
    }
    return names;
}
