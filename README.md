
[//]: # (Image References)

[image01]: ./camera_cal/chessboard16.jpg "Chessboard"
[image02]: ./output_images/rewarped1.jpg "Detected Lane"
[image03]: ./report%20images/P1_lanes_Detect.png "Project 1 detected Lanes"
[image04]: ./report%20images/topview02.jpg "Birds-eye mask"
[image05]: ./report%20images/top_slide_window_detect_lines.png "Sliding Windows Birds-eye"
[image06]: ./report%20images/distorted.jpg "Distorted Photo"
[image07]: ./report%20images/undistorted.jpg "Undistorted Photo"
[image08]: ./report%20images/lane_pixel_mask3.jpg "Lane Pixel Thresholding"
[image09]: ./report%20images/top_slide_window_detect_lines.png "Sliding Window detection"
[image10]: ./report%20images/binary_warped.jpg "Lane on Binary Birds-Eye View"
[image1]: ./report%20images/u-net-architecture.png "U-Net Architecture"
[image2]: ./report%20images/img_mask_detection0.png "Example Image, Mask, Prediction"
[image3]: ./report%20images/mask_1479498371963069978.jpg "Mask from Bounding Boxes"


# Project 4/5: Advanced Lane Finding and Vehicle Detection #

## Introduction ##

Autonomous vehicles need to be perceive and understand their surroundings to be able to respond according to it. 
I.e. they need to know where obstacles are to avoid collisions and they need to be aware of lanes so that they can stay 
within them. Essentially autonomous vehicles need to understand the world around them and its rules to navigate this environment
 succesfully.<br><br>

Both projects, four and five, focus on the understanding of the car surroundings based on footage of a single dash-cam style
video stream. More specifically project 4 comes back to the detection of lane lines as studied in project 1 (basic lane line detction) but 
goes into more depth and approaches it on a technically more advanced level to allow more accurate tracking of curved lane lines
by fitting a second order polynomial to detected lane line pixels.
<br>
Project five deals with the detection and tracking of vehicles in images from the dashcam videostream. For both projects the 
videostream to use for lane/vehicle detection was recorded on highways. This is a less complex environment than city roads and provides a more suitable scope for this project.
<br>
<br>
As both projects aim to be applied to the same video footage I decided to combine both projects and provided a single video showing both, my implementation of lane and vehicle detection.
 
<br>

## Source Code Oragnisation ##
The projects source code structure started with project 4 in mind but then got extended to include vehicle detection (project 5) as well.
The generation of the distortion correction matrix and its inverse were handled by the cam_calibration.py script. They are saved to a pickle file and get loaded by the camera_correction script.

Perception.py uses the pickled transformation matrices to provided input for the lane_detection pipeline. It as well processes test images and handles video processing passing each frame to the road_processor.
  The road_processor is the controller of dataflows between the lane line tracker and vehicle detection it prepares the camera input by transforming the perspective to a top down view, extracting the lane line pixel masks
  and drawing detected lanes.  

The vehicle_detection_UNet python notebook was used to generate the semantic segmentation network which gets used in vehicle_inference.py to detect vehicles in road scene images.

A lane_line_tracker class has been created to analyse the prepared birds-eye view depending on the detection success in the past. When lightweight detection of lane lines fails for too long a full scan is carried out to detect them.

The tracker uses a lane line class to track key properties of lane lines, like lane line pixels, fitted polynomial coefficients, curvature, last n fits, etc. 

## Lane Finding ##
### Approach ###
Already in project 1 lanes were detected. Back then the basic approach was limited to finding straight lines, which resulted
in inaccurate lane lines which were often comprised of many shorter lines drawn. In this project we revisit detecting lane lines but
improve on the basic canny edge detection approach.  
<br>
![alt_text][image03]
*Lane lines detected through canny edge*
<br>
Improvement in this take on lane line detection is, first and foremost, that we enable detection of curves by fitting a second order polynomial to the line pixels.
This has the advantage that we can draw curved lane boundaries in clean way, i.e. we do not have to compromise between a single line somewhat accurately, especially in curves, aligning with the actual lane lines
and multiple small fragments fuzzily approximating the curved lane line. <br>
Further lower level improvements include the undistortion of images through applycation of a correction matrix, using different colourspaces and the sobel function, a directional
version of the canny edge detection method used in project 1.  
Having the coefficients of the fitted polynomials allowed the calculation of curve radius and relative position of the car with respect to the lane center.

### 1. Camera Calibration and Undistortion ###
Even though current camera optics are at a high standard, images still show small distortions. To remove those a calibration matrix was calculated using photographs off chessboards patterns from different angles. Based on the location of the chessboards tiles' 
corners a correction matrix was computed and used to correct them.
![alt_text][image01]
*One out of several chessboard camera calibration images*

Below the same image is depicted in its original (first) and undistorted version (second). It is hardly possible to see distortions but the car at the left edge is no longer part of the image.

![alt_text][image06]
*Original image*
<br>
![alt_text][image07]
*Undistorted image. The fact that the car at the left edge has been cut off is an indicator that the correction altered the image*


### 2. Detecting Lane Line Pixels
To detect pixels which depict lane boundary markings the function colour_threshold of the road processor class analyses the saturation channel from the HLS colourspace and the value channel from the HSV colourspace were extracted and 
thresholded.

For the saturation channel the range from 100 to 255 was extracted where for value channel 50 to 255 were extracted. When both these layers agreed the colour_threshold function returned "1" for this pixel in a binary mask.
These values as well as this function were proposed in the Q&A Video for project 4 and was found to work well with the test images.

This information was combined with the output of sobel processing in the x and y dimension. 
For the x and y dimension 12-255 and 25-255 were used as the acceptance range. 
When those two functions agreed or the colour_threshold function returned 1 then the pixel was considered part of the lane line.
<br><br>
![alt_text][image08]
*Binary mask of extracted lane pixels*
<br>


### 3. Perspective transform

The obtained binary mask of lane pixels was transformed from the dashcam perspective to a birds-eye view to make lane detection easier. The top-down view is helpful because the comparison of lane line polynomials gets easier.
E.g. in the top down view the lane lines are parallel and this feature can be used to validate the found lane lines. <br>
The function get_trans_matrices in perception.py uses cv2.getPerspectiveTransform to warp a trapezoid depicting the two current lane lines to a rectangular image. cv2 does this by creating a matrix which stretches
the image so that the four points specified are the corners of the new images.
<br>
![alt_text][image04]
*Perspective transformed binary mask of lane pixels*
<br>

From this top down view the pixels in x and y dimension can be mapped to meters which will later be used to calculate the curve radius and distance
of the car to the center of the lane.

### 4. Detect Lanes
Two approaches have been implemented to identify lane lines.  
The first one aims to find lane lines without knowledge of previous lane lines. This will be used for detecting lanes on individual test images, the first frame of a video
and when the lane lines were not detected for an certain amount of frames.  
<br>
To find the lines the approach demonstrated in the lectures was applied. This involved creating a histogram of 
all lane pixels in each column of the lower half of the image. The two peaks identified in this were used as starting points of the lanes.
As depicted below, windows were anaylsed around the peaks to find the the center of the next window if a threshold of minimum lane pixels was detected. This sliding window
approach was applied to the entire image resulting in pixel indices of the left an right lane which then were used to fit a second order polynomial, the first lane line.

<br>

![alt_text][image05]

<br>

![alt_text][image10]

<br>

The second method operates with the knowledge of the last frames and the location of lanes lines in those. The new lane lines are found by
 searching near last frames fitted polynomial. A polynomial is fitted to these new lane line pixels.  
 This method is less computationally intense which helps speed up the lane detection process in video data.

The resulting lanes are sanity checked in the lane line tracker function sanity_check. The function was created to ensure that the lane lines are roughly parallel and that their distance to each other does not increase too much. This happend in the video occasionally when the road edge was detected as a the left lane line.
The lane line distances were monitored when the wrong line got detected and the allowed distance was chosen to avoid this.
To ensure parallel lane lines the difference in the first polynomial coefficient was observed and a sufficiently small limit chosen to 
sanity check for similar curvature.

### 5. Curve Radius and Vehicle Position
With the polynomials we are able to generate essential information for a self driving car: The curvature of the road and the location of the car with respect to the lane center. <br>
 Based on this information the vehicle can calculate the necessary steering angles to look ahead and stay in lane long term but as well micro-adjust to stay perfectly centered.
 I calculate the radius in the lane_line class in the function calc_curv_rad based on the "Measuring Curvature" chapter of the Udacity lectures.
 Vehicle position is calculated as the distance of the center point at the bottom edge of the image to the center of the two detected lane lines at the bottom edge.

### 6. Display Fitted Lane Lines
The fitted lane mask was transformed back to the original perspective in the draw_lanes function for test images and then merged with the original image. <br>
Curve radius and vehicle distance to center are drawn onto the image using cv2.putText.

![alt_text][image02]
*Visualization of Lane Lines and display of curvature and position with respect to center*

### 7. Video Processing
Processing an entire stream of consecutive frames can be seen as just a series of individual images but consecutive images will show a similiarity to the previous frames. 
 To speed up the detection of lane lines as described above, the lighter lane finding method has been used when past information of lanes is present and of sufficient quality (sanity checks).
To aid this scenario the RoadImageStreamProcessor class checks the status of detected lanes and decides whether the full scan or a light scan is to be issued.<br>
Additionally stabilization of lane lines was implemented by averaging polynomials coefficients for lane drawing over the last 5 frames.
The resulting video has been released on youtube.
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/GBXiHHyBoEg/0.jpg)](https://www.youtube.com/watch?v=GBXiHHyBoEg)

### Discussion
As it can be seen in the linked video the resulting lane detector works well. 
There were cases where the detector recognised the edge of the boundary of the road as a lane line but these situation were rectified with the sanity checker ensuring the distance of lane lines is as expected.
In these situations the tracker indicates that previous searches were not successful and uses the full scan to find lanes again.
If I were to do this again I would like to solve this as well using semantic segmentation as the tuning of colour thresholds can be quite delicate and may not generalize as well as a neural network.
  

## Vehicle Detection ##
### Approach ###
The Udacity lessons suggest to use traditional machine learning based on histograms of gradients (HOG). However solutions using deep learning for vehicle detection have been accepted as well.
I personally prefer to apply deep learning approaches and implemented a semantic segmentation neural network. More specifically a U-Net. The basic principle 
of U-Nets is that in the first half the individual images are scaled down, but increase in depth, through convolutional and pooling layers to then bring the scale up to the original x and y dimensions of the image to form a mask
which possess the value 1 where a vehicle is detected and 0 where it is not. These convolutions and deconvolutions are horizontally connected
to feed smaller features, from early layers, into deeper layers as well. This architecture forms a U when graphed as seen below.

![alt_text][image1]
*U-Net Architecture<cite>[Ronneberger et al.][1]</cite>*

This architecture was successfully used in semantic segmentation, the pixel wise classifaction of images, of images <cite>[Ronneberger et al.][1]</cite><cite>[2]</cite>. In this project it was used to just a detect a single class, vehicles, from the Udacity provided dataset. The provided dataset contained 
cars and trucks which have been combined to a single vehicle class as all kinds of vehicles needed to be detected independent of the type. <br>

The dataset provided did not entail the accurate pixelwise masks that were usable for training but bounding boxes containing vehicles. The bounding boxes were converted to filled rectangles and used as input to train the UNet to output pixel masks 
denoting vehicles. <br> 
Based on its origin this network could easily be modified to allow detection of more classes like sky, lane markings,road seperation, trees, etc. However this was not required for this task
so things were kept simple.

### Preparing the Training Data ###
For the generation of the model that has been used for vehicle detection a Jupyter notebook has been used (Vehicle_detection_UNet.ipynb).
<br>
In it the crowdai dataset was prepared for use in a semantic segmentation neural network. The dataset consists of more than 66.000 frames of roadscene images with bounding box coordinates in a csv file.
This format is in this form not suitable for a semantic segmentation network as we would need a mask that overlayed on the image shows the classes of depicted objects. This is typically done using colour coding.
Here we merged the car and truck class of the dataset to a single vehicle class. We hence only need to distinguish between vehicle or no vehicle in the mask which will be the label of our network.

To obtain areas marking the presence of vehicles the bounding boxes were converted into filled rectangles using the cv2.rectangle function with a line thickness of -1.
The masks were saved to files for use in the generator function.

The generator was necessary as the entire dataset is too large to be processed at once and provides batches of 16 images at once. As subsequent frames in order would be too similar and would lead to a very biased model are the images chosen at random from the dataset.

![alt_text][image3]
*Vehice mask generated from bounding box data*

### The U-Net Model ###
Semantic segmentation competitions often evaluate models on Intersection over Union (IoU) which describes the intersection of the prediction and the ground truth devided by the union of both. 
This penalises too large predictions as false positives as well as false negatives are equally harmful in this metric. The negative IoU was used as a loss function for training.
A lambda layer was used to normalize the image integers to be zero centered.
 and range from -1 to 1. The original images as well as masks have been resized to 320 by 608 pixels to improve training times as well as lowering the memory requirements of the U-Net.
<br>
After good experience with the ADAM optimizer it was chosen for training of this model as well.
The loss function as well as the basic network architecture was taken from [2]. 
![alt_text][image2]
*Original image, predicted mask and ground truth mask*
<br>
[image2] shows the original image, predicted mask and given mask side by side. This outcome together with a training IoU of ~0.9 suggested good performance. However when embedding it into the RoadImageStreamProcessor and processing the entire video 
false positives were detected in the road barriers on the left. The training dataset exploration showed predominantly urban scenes and few, if any, motorway scenes. This could make the model underperform in motorway situations. Possibly the 
metal of the barriers misleads the classifier. 


### Post-Processing ###
To avoid false postives a temporal heatmap was used. This was implemented by summing up the masks of the last 20 frames and requiring that a mininum of 19 frames have to agree on the detection of a vehicle.
Furthermore it was made a requirement for vehicle pixels to be detected on the last three consecutive frames in order to be displayed as a vehicle.
This helped and the smoothed the detection of vehicles over frames but did not solve the road boundary false positive.


### Discussion ###
The road boundary false positive posed a significant issue until the end. To improve the performance a dataset with more highway data could ease the problem. 

With the semantic segmentation network only the classification of pixel in one image gets considered. Due to the temporal relation of all frames in camer footage there is a temporal aspect which could be exploited as well.
I would like to try to combine the U-Net with long-short term modules (LSTM). I believe this could help avoid the boundaries beeing misclassified as cars.
<br> Another, possibly simpler, approach would be to generate a confidence score map for each mask and apply a confidence threshold to avoid false positives. <cite>[Kendall et al.][3]</cite>

[1]:Ronneberger et al; https://arxiv.org/abs/1505.04597  
[2]:https://github.com/jocicmarko/ultrasound-nerve-segmentation  
[3]:Alex Kendall, Vijay Badrinarayanan and Roberto Cipolla "Bayesian SegNet: Model Uncertainty in Deep Convolutional Encoder-Decoder Architectures for Scene Understanding." arXiv preprint arXiv:1511.02680, 2015
 


