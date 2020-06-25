# SFND 2D Feature Tracking
## Sensor Fusion Engineer Nanodegree

	The aim of this project is to compare different combinations of keypoint detector/descriptors and matching techniques and to find which one is the most relevant in order to build a robust collision avoidance system. This system uses the Time-To-Collision metrics of a preceding vehicle in the traffic calculated based on a sequence of images given by a mono camera installed on our vehicle.


###__MP.1 Data buffer optimization :__

####__Task :__ 
	Implement a vector for dataBuffer objects whose size does not exceed a limit (e.g. 2 elements). This can be achieved by pushing in new elements on one end and removing elements on the other end. 

	I'm first checking if the size of `dataBuffer` is equal to the variable `dataBufferSize` before pushing back a new element. If yes, I remove the first element of `dataBuffer` using the erase method with `dataBuffer.begin()`, which is a pointer to the first element of the buffer, as parameter. Then I can push back the new element.

```
DataFrame frame;
frame.cameraImg = imgGray;
if(dataBuffer.size() == dataBufferSize) {
    dataBuffer.erase(dataBuffer.begin());
}
dataBuffer.push_back(frame);
assert(dataBuffer.size() <= dataBufferSize);
```


###__MP.2 Keypoint detection :__

####__Task :__ 
	Implement detectors HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT and make them selectable by setting a string accordingly.


```
 cv::Ptr<cv::FeatureDetector> detector;

    if(detectorType.compare("FAST") == 0) {
        int threshold = 30; // Difference between intensity of the central pixel and pixels of a circle around this pixel.
        bool nms = true; // Perform non-maxima suppression on keypoints.
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;
        detector = cv::FastFeatureDetector::create(threshold,nms, type);
    }
    else if(detectorType.compare("BRISK") == 0) {
        detector = cv::BRISK::create();
    }
    else if(detectorType.compare("ORB") == 0) {
        detector = cv::ORB::create();
    }
    else if(detectorType.compare("AKAZE") == 0) {
        detector = cv::AKAZE::create();
    }
    else if(detectorType.compare("SIFT") == 0) {
        detector = cv::xfeatures2d::SIFT::create();
    }
    else {
        throw std::invalid_argument("Detector not implemented : " + detectorType);
    }

    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << detectorType << " with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
```



###__MP.3 Keypoint Removal :__

####__Task :__ 
	Remove all keypoints outside of a pre-defined rectangle and only use the keypoints within the rectangle for further processing. 

	The idea is to focus on the detection collision system and keypoints on the preceding vehicle are of special interest to us. Therefore, we want to discard feature points that are not located on the preceding vehicle (points that are outside of a predefined box).
Box parameters are : cx = 535, cy = 180, w = 180, h = 150.

NB : Some keypoints detecting for example on the road can still occur. This technique should not be used and will be replaced by a more reliable one in the final project.

```
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
```

###__MP.4 Keypoint Descriptors :__

####__Task :__ 
	Implement descriptors BRIEF, ORB, FREAK, AKAZE and SIFT and make them selectable by setting a string accordingly.

```
cv::Ptr<cv::DescriptorExtractor> extractor;
if (descriptorType.compare("BRISK") == 0)
{
	int threshold = 30;        
	int octaves = 3;           
	float patternScale = 1.0f; 

	extractor = cv::BRISK::create(threshold, octaves, patternScale);
}
else if(descriptorType.compare("BRIEF") == 0)
{
	extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
}
else if(descriptorType.compare("ORB") == 0) {
	extractor = cv::ORB::create();
}
else if(descriptorType.compare("FREAK") == 0) {
	extractor = cv::xfeatures2d::FREAK::create();
}
else if(descriptorType.compare("AKAZE") == 0) {
	extractor = cv::AKAZE::create();
}
else if(descriptorType.compare("SIFT") == 0) {
	extractor = cv::xfeatures2d::SIFT::create();
}
else {
	throw std::invalid_argument("Descriptor not implemented : " + descriptorType);
}

// perform feature description
double t = (double)cv::getTickCount();
extractor->compute(img, keypoints, descriptors);
t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
```


###__MP.5 Descriptor Matching :__

####__Task :__ 
	 	Implement FLANN matching as well as k-nearest neighbor selection. Both methods must be selectable using the respective strings in the main function. 

```
else if (matcherType.compare("MAT_FLANN") == 0)
{
	if (descSource.type() != CV_32F)
	{ // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
	    descSource.convertTo(descSource, CV_32F);
	    descRef.convertTo(descRef, CV_32F);
	}

matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
cout << "FLANN matching";
}
```


###__MP.6 Descriptor Distance Ratio :__

####__Task :__ 
	Use the K-Nearest-Neighbor matching to implement the descriptor distance ratio test, which looks at the ratio of best vs. second-best match to decide whether to keep an associated pair of keypoints.

```
// K-Nearest Neighbour Distance Ratio in order to lower the number of false positive by applying the threshold directly on the distance ratio.
double minDescDistRatio = 0.8;
for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
{
    if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
    {
        matches.push_back((*it)[0]);
    }
}
cout << "# keypoints removed = " << knn_matches.size() - matches.size() << endl;
```

###__MP.7 Performance Evaluation 1 :__

####__Task :__ 
	Count the number of keypoints on the preceding vehicle for all 10 images and take note of the distribution of their neighborhood size. Do this for all the detectors you have implemented. 


###__MP.8 Performance Evaluation 2 :__

####__Task :__ 
	 Count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. In the matching step, the BF approach is used with the descriptor distance ratio set to 0.8.


###__MP.9 Performance Evaluation 3 :__

####__Task :__
	Log the time it takes for keypoint detection and descriptor extraction. The results must be entered into a spreadsheet and based on this data, the TOP3 detector / descriptor combinations must be recommended as the best choice for our purpose of detecting keypoints on vehicles.


