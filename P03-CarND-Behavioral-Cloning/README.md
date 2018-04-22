# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


[//]: # (Image References)

[image0]: ./ref/crop_0.jpg "Before Crop"
[image1]: ./ref/crop_1.jpg "After Crop"
[image2]: ./ref/1-center.jpg "Center driving"
[image3]: ./ref/1-left.jpg "Recovery Image"
[image4]: ./ref/1-right.jpg "Recovery Image"
[image5]: ./ref/0-left.jpg "Left Camera"
[image6]: ./ref/0-center.jpg "Center Camera"
[image7]: ./ref/0-right.jpg "Right Camera"
[image8]: ./ref/flip_before.jpg "Before Flip"
[image9]: ./ref/flip_after.jpg "After Flip"
[image10]: ./ref/d_l_turn.jpg "Before Crop"
[image11]: ./ref/d_r_turn.jpg "After Crop"
[image12]: ./ref/d_bridge.jpg "Before Crop"
[image13]: ./ref/d_open_space.jpg "After Crop"
[image14]: ./ref/hist1.png "unbalanced histogram"
[image15]: ./ref/hist2.png "balanced histogram"

Overview
---
This is a fun project to work on. While training the car manually in simulator, I feel like playing the video game. Teaching the car to drive autonomously is like teaching a baby to walk. I need to hand hold it, and be patient to see it grows slowly, from crawl to walk, and finally... run. I cannot help laughing while watching the car went off the road and fell under cliff or sank into the ocean. It's such a joy to see it eventually runs on the track tirelessly, hands free.

You can take a journey with my baby car by clicking the animation below -

[![My Self-Driving Car](./ref/MyCar.gif)](https://youtu.be/9m8kmz26n4Y)



The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

#### Details About Files In This Directory

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* video.mp4 recording of the vehicle driving autonomously around the track 1.

### Approach

I went through this project with several iterations.

First I picked 3 models to train the data - [LeNet model](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf), [Comma.ai model](https://github.com/commaai/research/blob/master/train_steering_model.py) and [Nvidia self driving car model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). 

The first show stopper I encountered came from the image format. The cv2 package I used to handle the images read in training images as BGR format, while the real time images received from the simulator is RGB format. I need to make sure the trainer in model.py and predictor in drive.py use same image format and preprocessing. 

This was reminded in the project handout, however I failed to pay attention to it. The car went wild as the predictions are off due to the wrong images sent to the model. A lot of debugging and testing time was spent to finally figuring this out.

I trained all 3 models using Udacity dataset, and Nvidia model is the first one finish the round trip using same hyperparameters, so I continue the project using this model only.

Here is a summary of the Nvidia model architecture:

```sh
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 70, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 33, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 15, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 6, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 4, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 2, 33, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 4224)              0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 4224)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               422500    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 559,419
Trainable params: 559,419
Non-trainable params: 0
_________________________________________________________________

```

#### Data collection

With the initial working codes, I started to collect my own data.
I paid special attention to left and right turns, open space to exit, and road condition change such as bridge.

These are some example of the data collected -

* Left and right turns -

![alt text][image10]
![alt text][image11]

* Bridge and open space to exit -

![alt text][image12]
![alt text][image13]

#### Data processing and augmentation

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

![alt text][image0]

![alt text][image1]

I experimented to resize the image to 66x200 (the size used in Nvidia paper) or 64x64. I was able to still make round trips for both sizes, with very little training time. On on Intel I7 machine with Nvidia GTX 1070TI GPU, I was about to finish the training in 25 seconds and still complete track 1. However I noticed it waggles more than using the bigger size images, which has more parameters in the model to tune, and could generate more stable results.

I also implemented the data generator in the Keras model. It only loads data to memory when needed, one batch a time, therefore can handle very big dataset. Also it can do data augmentation on the fly, which is a convenient and efficient way to try out different kinds of data manipulation.


#### Data analysis

Below is the histogram of steering angle distribution of track2 data I collected.
Most of the data points collected have steering angle as 0. The +0.2 and -0.2 data are from the left and right camera which centered at 0 degree, with correction set at 0.2.

![alt text][image14]

Some people reported that balancing data by removing data from bins with more data as shown below can help to improve the results. I tried it on my model and data, and found the result seems even worse than the original dataset, as less data points are used for training and testing, the trade-off does not justify the benefit in my experiment.

![alt text][image15]


### Conclusion

This project is supervised machine learning. The (image, steering angle) data points are labeled by training the car in the simulator manually. However this process is more of an art than a science. The quality of the data depends on your skill to keep the cars on track in the simulator. This is easier for track 1, but more challenging for track 2. 

I also tried the approach to apply the lane finding project here. Finding the lanes to estimate the steering angles, then compare and adjust the simulated data. However the approach does not get better results, as it has problems to find the lanes for centain road condition as well. 

As mentioned in the beginning, this is an interesting project. I spent many hours on it. However the project does not end here, I would like to revisit it again after completing the advanced lane finding project, or with the path planning project.

