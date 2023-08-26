# ASL_Translation_using_object_detection_models
REAL TIME CLOSED CAPTION ASL TRANSLATOR USING   OBJECT DETECTION MODELS

![Screenshot 2023-08-26 153814](https://github.com/miraj0507/ASL_Translation_using_object_detection_models/assets/62544210/5f659a9b-390d-4a39-95a3-f5e4edfaf516)
Can computer vision bridge the gap for the deaf and hard of hearing by learning American Sign Language? If ASL can accurately be interpreted through a machine learning application, even if it starts with just the alphabet, we can mark a step in providing greater accessibility and educational resources for our deaf and hard-of-hearing communities.

American Sign Language (ASL) is a complete, natural language that has the same linguistic properties as spoken languages, with grammar that differs from English. ASL is expressed by movements of the hands and face. It is the primary language of many North Americans who are speech and hearing impaired, along with some hearing people as well. In most cases, people communicating in sign language use actions and gestures to express blocks of words instead of just individual letters or symbols. 

# DEMO VIDEO

https://github.com/miraj0507/ASL_Translation_using_object_detection_models/assets/62544210/ba7448bd-76ba-4bd2-9faf-8988533b446e



LSTM MODEL
 
For the model we trained, we collected 30 videos for 3 classes. Each of the videos are 2-3 seconds long and consists of 30 FPS. The extracted keypoints of landmarks are obtained using MediaPipe[12] and are saved in a .npy file. We created our custom dataset by using a Python script utilizing OpenCV package to capture videos from a webcam and storing them in .npy format after MediaPipe processing[6][7]. The files were then separated into folders specific to each class.

We utilize the Python packages of matplotlib (for graphical representations), opencv-python (for image manipulation and accepting real time images), pytorch[11] (for model loading and construction), along with the yolov5s package.

We used OpenCV to video capture the action of 30 videos of 30fps of 3 seconds each. We used this as our input to the MediaPipe API. The landmarks were extracted and fed to the input layer of the LSTM model. We trained the LSTM model with our custom made dataset of 3 classes, each for the action detection[9] sequence [‘hello’, ‘iloveyou’, and ‘thanks’]. The model was trained over 150 epochs but was stopped early using EarlyStopping at 126 epochs. 

For testing, opencv-python is used to access the webcam. The model detects the required action that it has been trained with and displays the output with captioning [13].

The output produced from our LSTM model can detect and predict action labels for the trained classes ‘hello’, ‘thankyou’ and ‘iloveyou’ from real time feed.
The max training accuracy achieved was around 92.3% after 126 epochs following which the training was stopped early. The categorical loss that was recorded was around 0.2675. The graphs generated from training using tensorboard are given below. 


