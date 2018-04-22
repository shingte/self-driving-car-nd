# **Behavioral Cloning** 

## Writeup


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image0]: ./ref/nvidia_model.jpg "Model Layers"
[image1]: ./ref/Nvidia-cnn.png "Model Visualization"
[image2]: ./ref/1-center.jpg "Center driving"
[image3]: ./ref/1-left.jpg "Recovery Image"
[image4]: ./ref/1-right.jpg "Recovery Image"
[image5]: ./ref/0-left.jpg "Left Camera"
[image6]: ./ref/0-center.jpg "Center Camera"
[image7]: ./ref/0-right.jpg "Right Camera"
[image8]: ./ref/flip_before.jpg "Before Flip"
[image9]: ./ref/flip_after.jpg "After Flip"
[image10]: ./ref/crop_0.jpg "Before Crop"
[image11]: ./ref/crop_1.jpg "After Crop"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* video.mp4 recording of the vehicle driving autonomously around the track 1.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a neural network with 5 convolution layers (model.py lines 303-315) 

![alt text][image0]

The model includes RELU layers to introduce nonlinearity (code lines 305-309), and the data is normalized in the model using a Keras lambda layer (code line 255). 

The Keras cropping layer is used to crop top 70 pixels and bottom 20 pixels (code line 256)

Data can be fed to the model through model.fit_generator or model.fit (code lines 342-355) 
    

#### 2. Attempts to reduce overfitting in the model

I experimented adding dropout layers in order to reduce overfitting, however I didn't find it helpful for this model, so did not include it in the final architecture.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. The training loss and validation loss are close enough to show that there's no overfitting issue.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 341).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, corssing the bridge and sharp turns data. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with some existing models, compare the performance and results, then choose one as the final model architecture.

My first step was to use a convolution neural network model similar to the LeNet model used in previous project. I built the model layer by layer to see which component or hyperparameters matter most to the result.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

I decided to benchmark the 3 models I chose using the Udacity dataset, and found Nvidia model is the easiest to stabilize and finish the round trip, so I used it to continue my project.

Then I collect the data myself using the training mode in the simulator.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I collected data in those trouble spots and added them to the dataset.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

I tried 3 models for this project. [LeNet model](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) that was used in project 2; [Comma.ai model](https://github.com/commaai/research/blob/master/train_steering_model.py); and [Nvidia model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). 
Based on the testing results on Udacity data, I picked Nvidia model for this project.

Here is a visualization of the Nvidia model architecture -

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. 

Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to make it to the turns. These images show what a recovery looks like starting from left or right edge of the road :

![alt text][image3]
![alt text][image4]


I repeated this process in order to get more data points and cover more road conditions.

The simulator records the images from cameras in 3 different locations of the car - center, left and right. To augment the dataset, I added the images from left and right cameras with angle correction of 0.2. The csv file only contains the steering angle from the center camera. Corrections are applied to the left/right images to be included to the dataset.

- Left  steering angle = center steering angle + correction
- Right steering angle = center steering angle - correction

![alt text][image5] ![alt text][image6] ![alt text][image7]


I flipped images and angles to increase the size of the dataset.

For example, here is an image that has then been flipped:

![alt text][image8]
![alt text][image9]

I also cropped images so that the model can ignore the details not related to the driving training and prediction.
I did this using the Cropping2D layer in Keras. Top 70 pixels and bottom 20 pixels are cropped.
While cropping the images before train can reduce the memory usage, I found doing it using Keras layer results in less total parameters in the model.

For example, here is an image that has then been cropped:

![alt text][image10]

![alt text][image11]

After the collection process, I had 6240 number of data points. I then preprocessed this data by converting the image format from BGR to RGB, as cv2 reads in images as BGR, while drive.py gets images from simulator in RGB.
In the beginning, I didn't do this conversion, then no matter how I changed the model, the car just running wild and didn't respond to the changes. It took me quite a while to figure out the issue and fixed it.

I tried many other preprocessing such as blurring the images, convert to YUV, etc. However they do not improve the results in my experiments, so I didn't include them in the codes.
I also tried reducing the images by resizing. It can help to reduce the memory consumption and takes less time to train the model. I can still finish the round trip with the smaller images and model, however it seems the car waggles more compare to the original size. Maybe it just needs more time or data to train, but I can get pretty good result without it, so I just leave it alone . I may explore this size shrinking and other tuning later. 


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by the convergence of the result, and the validation loss is close to the training loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
