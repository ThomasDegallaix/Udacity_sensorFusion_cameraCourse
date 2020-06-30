# SFND 2D Feature Tracking
## Sensor Fusion Engineer Nanodegree

The aim of this project is to compare different combinations of keypoint detector/descriptors and matching techniques and to find which one is the most relevant in order to build a robust collision avoidance system. This system uses the Time-To-Collision metrics of a preceding vehicle in the traffic calculated based on a sequence of images given by a mono camera installed on a vehicle.

### Overview

1. Load the images into a ring buffer. 
2. Use OpenCV to apply a variety of keypoint detectors.
    - SHI-TOMASI
    - HARRIS
    - FAST
    - BRISK
    - ORB
    - AKAZE
    - SIFT 
3. Use OpenCV to extract keypoint descriptors.
    - BRISK
    - BRIEF
    - ORB
    - FREAK
    - AKAZE
    - SIFT 
4. Use FLANN or Brute force matching to find matches between the extracted descriptors anduse kNN to improve the matching.
5. Run different combinations of these algorithms and evaluate their performances.

### MP.1 Data buffer optimization :

#### Task : 
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


### MP.2 Keypoint detection :

#### Task : 
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



### MP.3 Keypoint Removal :

#### Task : 
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

### MP.4 Keypoint Descriptors :

#### Task : 
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


### MP.5 Descriptor Matching :

#### Task : 
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


### MP.6 Descriptor Distance Ratio :

#### Task : 
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

### MP.7 Performance Evaluation 1 :

#### Task : 
Count the number of keypoints on the preceding vehicle for all 10 images and take note of the distribution of their neighborhood size. Do this for all the detectors you have implemented. 

Results are stored in the "/results/MP7_results/" folder in csv files under the name "MP7_" + detector type.


### MP.8 Performance Evaluation 2 :

#### Task : 
Count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. In the matching step, the BF approach is used with the descriptor distance ratio set to 0.8.

Results are stored in the "/results/MP8_MP9_results/" folder in csv files under the name "DETECTORTYPE_DESCRIPTORTYPE_MATCHINGMETHOD_SELECTINGMETHOD.csv".

### MP.9 Performance Evaluation 3 :

#### Task :
Log the time it takes for keypoint detection and descriptor extraction. The results must be entered into a spreadsheet and based on this data, the TOP3 detector / descriptor combinations must be recommended as the best choice for our purpose of detecting keypoints on vehicles.

Results are stored in the "/results/MP8_MP9_results/" folder in csv files under the name "DETECTORTYPE_DESCRIPTORTYPE_MATCHINGMETHOD_SELECTINGMETHOD.csv".

From these csv files we can compute the average matches number and the average computing time for each detector/descriptor combination (See MP9_results in the results folder):

| Detector type | Descriptor type | Average matches | Average time (ms) | 
|---------------|-----------------|-----------------|-------------------| 
| BRISK         | SIFT            | 182             | 50.4         	| 
| SIFT          | BRISK           | 65              | 72.35        	| 
| AKAZE         | BRISK           | 135             | 47.37        	| 
| FAST          | BRIEF           | 122             | 1.52         	| 
| SHITOMASI     | BRIEF           | 104             | 11.6         	| 
| ORB           | BRIEF           | 60              | 6.05         	| 
| HARRIS        | FREAK           | 16              | 39.37        	| 
| AKAZE         | BRIEF           | 140             | 47.13        	| 
| SHITOMASI     | FREAK           | 85              | 37.15        	| 
| ORB           | BRISK           | 83              | 6.24         	| 
| FAST          | BRISK           | 99              | 2.2          	| 
| ORB           | SIFT            | 84              | 30.76        	| 
| FAST          | FREAK           | 97              | 30.03        	| 
| BRISK         | BRISK           | 174             | 32.31        	| 
| AKAZE         | AKAZE           | 139             | 85.64        	| 
| FAST          | ORB             | 120             | 3.7          	| 
| BRISK         | FREAK           | 169             | 59.17        	| 
| BRISK         | ORB             | 167             | 38.75        	| 
| HARRIS        | SIFT            | 18              | 26.36        	| 
| BRISK         | BRIEF           | 189             | 31.2         	| 
| AKAZE         | ORB             | 131             | 52.97        	| 
| SHITOMASI     | SIFT            | 103             | 25.11        	| 
| SHITOMASI     | BRISK           | 85              | 11.8         	| 
| ORB           | FREAK           | 46              | 33.82        	| 
| AKAZE         | SIFT            | 141             | 62.82        	| 
| SIFT          | BRIEF           | 78              | 83.9         	| 
| HARRIS        | BRISK           | 15              | 13.83        	| 
| FAST          | SIFT            | 116             | 15.68        	| 
| ORB           | ORB             | 84              | 15.05        	| 
| AKAZE         | FREAK           | 132             | 75.34        	| 
| HARRIS        | ORB             | 17              | 15.46        	| 
| HARRIS        | BRIEF           | 19              | 13.14        	| 
| SIFT          | FREAK           | 66              | 115.82      	| 
| SHITOMASI     | ORB             | 100             | 13.94        	| 
| SIFT          | SIFT            | 88              | 132.99       	| 



## Conclusion

Regarding all the data gathered:

- If we value __speed__, the TOP 3 detector/descriptor combination would be:


| TOP | Detector type | Descriptor type | Average time (ms) | 
|-----|---------------|-----------------|-------------------| 
| 1   | FAST          | BRIEF           | 1.52         	    | 
| 2   | FAST          | BRISK           | 2.2        	    | 
| 3   | FAST          | ORB             | 3.7        	    | 

FAST as a detector seems to be clearly ahead in terms of computing speed.

- If we value __accuracy__, the TOP 3 detector/descriptor combination would be:

| TOP | Detector type | Descriptor type | Average matches | 
|-----|---------------|-----------------|-----------------| 
| 1   | BRISK         | BRIEF           | 189         	  | 
| 2   | BRISK         | SIFT            | 182        	  | 
| 3   | BRISK         | BRISK           | 174        	  | 

BRISK as a detector seems to be clearly ahead in terms of accuracy.

For our application, although accuracy is an important property, I do not recommend using one of these 3 combinations as one of their major downside is that they might be too slow (~ 30-50 ms) for a real time application involving security, such as our Collision Avoidance System.

However, if we look carefully at our top 3 speed detector/descriptor combinations, we can notice that not only they are fast, but they also provide an interesting number of matches (FAST-BRIEF = 122 matches, FAST-ORB = 120 matches).

That's why, for the purpose of our project, I suggest using the __FAST-BRIEF__ combination as it gives a good accuracy while being also super fast.

