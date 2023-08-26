# ASL_Translation_using_object_detection_models
REAL TIME CLOSED CAPTION ASL TRANSLATOR USING   OBJECT DETECTION MODELS
Can computer vision bridge the gap for the deaf and hard of hearing by learning American Sign Language? If ASL can accurately be interpreted through a machine learning application, even if it starts with just the alphabet, we can mark a step in providing greater accessibility and educational resources for our deaf and hard-of-hearing communities.

American Sign Language (ASL) is a complete, natural language that has the same linguistic properties as spoken languages, with grammar that differs from English. ASL is expressed by movements of the hands and face. It is the primary language of many North Americans who are speech and hearing impaired, along with some hearing people as well. In most cases, people communicating in sign language use actions and gestures to express blocks of words instead of just individual letters or symbols. 
## DEMO PREDICTION OF HAND GESTURE 

![DEMO](https://github.com/miraj0507/ASL_Translation_using_object_detection_models/assets/62544210/dcff17a3-1e3d-41ab-a683-8051789fb18a)



## DEMO VIDEO

https://github.com/miraj0507/ASL_Translation_using_object_detection_models/assets/62544210/ba7448bd-76ba-4bd2-9faf-8988533b446e



## LSTM MODEL
 
For the model we trained, we collected 30 videos for 3 classes. Each of the videos are 2-3 seconds long and consists of 30 FPS. The extracted keypoints of landmarks are obtained using MediaPipe[12] and are saved in a .npy file. We created our custom dataset by using a Python script utilizing OpenCV package to capture videos from a webcam and storing them in .npy format after MediaPipe processing[6][7]. The files were then separated into folders specific to each class.

We utilize the Python packages of matplotlib (for graphical representations), opencv-python (for image manipulation and accepting real time images), pytorch[11] (for model loading and construction), along with the yolov5s package.

We used OpenCV to video capture the action of 30 videos of 30fps of 3 seconds each. We used this as our input to the MediaPipe API. The landmarks were extracted and fed to the input layer of the LSTM model. We trained the LSTM model with our custom made dataset of 3 classes, each for the action detection[9] sequence [‘hello’, ‘iloveyou’, and ‘thanks’]. The model was trained over 150 epochs but was stopped early using EarlyStopping at 126 epochs. 

For testing, opencv-python is used to access the webcam. The model detects the required action that it has been trained with and displays the output with captioning [13].

The output produced from our LSTM model can detect and predict action labels for the trained classes ‘hello’, ‘thankyou’ and ‘iloveyou’ from real time feed.
The max training accuracy achieved was around 92.3% after 126 epochs following which the training was stopped early. The categorical loss that was recorded was around 0.2675. The graphs generated from training using tensorboard are given below. 


## MEDIAPIPE

Mediapipe[10] as itself offers a wide range of solutions, within the holistic pipeline of mediapipe[8][10], it consists of three components: pose, hand and face. The holistic model extracts a sum of 1662 individual landmark features (21 ∗ 3 + 21 ∗ 3 + 33 ∗ 4 + 468 ∗ 3 = 1662). We are using the MediaPipe holistic model included in the MediaPipe python module. We defined a function (mediapipe_detection) that takes two arguments:
1. Image: The image or the frame in which it performs detection
2. Model: The mediapipe model that we are using for detection (“Holistic model”). 

The function converts the image from BGR to RGB format, the image is then passed through model.process() function and the result is stored. The function then converts the image back to BGR format, and then returns the image and the stored result. We define another function (draw_styled_landmarks), it takes the returned output from the previous functions as its arguments. Using this function we draw the landmarks, which helps visualize the real-time hand detection. Extract Key-points: After real-time hand detection is successfully achieved, we then extract the X,Y and Z coordinates of the key points from the detection results(the stored result returned by the mediapipe_detection function mentioned above). 

We define another method (extract_keypoints) that takes the detection results as its argument, to perform the extraction of the coordinates. This function returns the concatenated array of all the arrays containing the key point coordinates of the holistic model. This function is called during data collection. While we used the holistic model and extracted key points for all three: hand, face and pose, we only showed hand landmarks on the screen during the recognition phase to avoid clusters of landmarks on the screen.



We use the LRCN  in our comparison study as it differs from the YOLO model in respect that it does not use bounding boxes or image localisation for frame processing and from the LSTM model as it does not use Holistic Pose and Hand detection to extract feature points. The LRCN model uses the entire frame for feature extraction without detecting the hand points or human pose which is then passed to the sequential model for temporal processing. For the LRCN model, we use a CNN to extract spatial features at a given time step in the input sequence (video) and then an LSTM to identify temporal relations between frames.
