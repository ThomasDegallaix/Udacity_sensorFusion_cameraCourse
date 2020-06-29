/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"


// AKAZE descriptor extractor works only with key-points detected with KAZE/AKAZE detectors
// see https://docs.opencv.org/3.0-beta/modules/features2d/doc/feature_detection_and_description.html#akaze

// ORB descriptor extractor does not work with the SIFT detetor
// see https://answers.opencv.org/question/5542/sift-feature-descriptor-doesnt-work-with-orb-keypoinys/

using namespace std;

 
//See https://www.gormanalysis.com/blog/reading-and-writing-csv-files-with-cpp/ for the csv writing and reading
void write_csv(string filename, vector<pair<string, vector<double>>> data) {
    ofstream myFile(filename);

    for(int i = 0; i < data.size(); ++i) {
        myFile << data.at(i).first;
        if(i != data.size() - 1) myFile << ",";    
    }

    myFile << "\n";

    for(int i = 0; i < data.at(0).second.size(); ++i) {
        for(int j = 0; j < data.size(); ++j) {
            myFile << data.at(j).second.at(i);
            if( j!= data.size() - 1) myFile << ",";
        }
        myFile << "\n";
    }

    myFile.close();
}


std::vector<std::pair<std::string, std::vector<int>>> read_csv(std::string filename){
    // Reads a CSV file into a vector of <string, vector<int>> pairs where
    // each pair represents <column name, column values>

    // Create a vector of <string, int vector> pairs to store the result
    std::vector<std::pair<std::string, std::vector<int>>> result;

    // Create an input filestream
    std::ifstream myFile(filename);

    // Make sure the file is open
    if(!myFile.is_open()) throw std::runtime_error("Could not open file");

    // Helper vars
    std::string line, colname;
    int val;

    // Read the column names
    if(myFile.good())
    {
        // Extract the first line in the file
        std::getline(myFile, line);

        // Create a stringstream from line
        std::stringstream ss(line);

        // Extract each column name
        while(std::getline(ss, colname, ',')){
            
            // Initialize and add <colname, int vector> pairs to result
            result.push_back({colname, std::vector<int> {}});
        }
    }

    // Read data, line by line
    while(std::getline(myFile, line))
    {
        // Create a stringstream of the current line
        std::stringstream ss(line);
        
        // Keep track of the current column index
        int colIdx = 0;
        
        // Extract each integer
        while(ss >> val){
            
            // Add the current integer to the 'colIdx' column's values vector
            result.at(colIdx).second.push_back(val);
            
            // If the next token is a comma, ignore it and move on
            if(ss.peek() == ',') ss.ignore();
            
            // Increment the column index
            colIdx++;
        }
    }

    // Close file
    myFile.close();

    return result;
}

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    /* Containers for the performance evaluation MP7/MP8/MP9 */
    double t_detector, t_descriptor;
    vector<pair<string, vector<double>>> results; // Variables which holds results for the performance evalutation
    vector<double> imageIndexData, keypointsNumberData, neighborhoodSizeMeanData, matchIndexData, matchesNumberData,
        detectorTimeData, descriptorTimeData, totalTimeData;
    string perfEvalTaskNumber = "MP8_MP9";

    string detectorType = "SHITOMASI";
    string descriptorType = "SIFT"; // BRIEF, ORB, FREAK, AKAZE, SIFT

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        if(dataBuffer.size() == dataBufferSize) {
            dataBuffer.erase(dataBuffer.begin());
        }
        dataBuffer.push_back(frame);
        assert(dataBuffer.size() <= dataBufferSize);
        

        //// EOF STUDENT ASSIGNMENT
        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
        if (detectorType.compare("SHITOMASI") == 0)
        {
            t_detector = detKeypointsShiTomasi(keypoints, imgGray, false);
        }
        else if(detectorType.compare("HARRIS") == 0)
        {
            t_detector = detKeypointsHarris(keypoints, imgGray, false);
        }
        else 
        {
            t_detector = detKeypointsModern(keypoints, imgGray, detectorType, false);
        }
       
        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        
        if (bFocusOnVehicle)
        {
            vector<cv::KeyPoint> filteredKpt;
            for(auto kpt : keypoints) {
                if(vehicleRect.contains(kpt.pt)) {
                    filteredKpt.push_back(kpt);
                }
            }
            keypoints = filteredKpt;
        }
        //cout << "Number of keypoints on the preceding vehicle: " << keypoints.size() << endl;

        
        imageIndexData.push_back(imgIndex);
        keypointsNumberData.push_back(keypoints.size());
        auto neighborhoodMean = accumulate(keypoints.begin(), keypoints.end(),0.0,
            [](const double sum, const cv::KeyPoint& kp) { return sum + kp.size; }) / keypoints.size();
        //cout << "Neighborhood mean: " << neighborhoodMean << endl;
        neighborhoodSizeMeanData.push_back(neighborhoodMean);


        if(imgIndex == imgEndIndex && (perfEvalTaskNumber.compare("MP7") == 0)) {
            string filename = "../results/MP7_results/MP7_" + detectorType + ".csv";

            pair<string, vector<double>> imageIndex("Image index", imageIndexData);
            pair<string, vector<double>> keypointsNumber("Keypoints number", keypointsNumberData);
            pair<string, vector<double>> neighborhoodSizeMean("Keypoints neighborhood size", neighborhoodSizeMeanData);
            results.push_back(imageIndex);
            results.push_back(keypointsNumber);
            results.push_back(neighborhoodSizeMean);

            write_csv(filename, results);
        }
        

        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;

        t_descriptor = descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);

        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
            string descriptorCategory; // DES_BINARY, DES_HOG
            if(descriptorType.compare("SIFT") == 0) descriptorCategory = "DES_HOG";
            else descriptorCategory = "DES_BINARY";
            string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp
            
            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorCategory, matcherType, selectorType);

            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            matchIndexData.push_back(imgIndex-1);
            matchesNumberData.push_back(matches.size());
            detectorTimeData.push_back(t_detector);
            descriptorTimeData.push_back(t_descriptor);
            totalTimeData.push_back(t_detector + t_descriptor);

            if(imgIndex == imgEndIndex && (perfEvalTaskNumber.compare("MP8_MP9") == 0)) {
                string filename = "../results/MP8_MP9_results/" + detectorType + "_" + descriptorType + "_" + matcherType + "_" + selectorType +".csv";

                pair<string, vector<double>> matchIndex("Matches index", matchIndexData);
                pair<string, vector<double>> matchesNumber("Matches number", matchesNumberData);
                pair<string, vector<double>> detectorTime("Detector time (ms)", detectorTimeData);
                pair<string, vector<double>> descriptorTime("Descriptor time (ms)", descriptorTimeData);
                pair<string, vector<double>> totalTime("Total time (ms)", totalTimeData);

                results.push_back(matchIndex);
                results.push_back(matchesNumber);
                results.push_back(detectorTime);
                results.push_back(descriptorTime);
                results.push_back(totalTime);

                write_csv(filename, results);
            }

            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            bVis = false;
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
        }

    } // eof loop over all images

    return 0;
}
