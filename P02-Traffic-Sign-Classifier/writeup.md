# **Traffic Sign Recognition** 

## Writeup


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[i_sample]: ./ref/sample_data.png "Visualization"
[i_bar_train]: ./ref/bar_train.png "Bar train data"
[i_bar_valid]: ./ref/bar_valid.png "Bar validation data"
[i_bar_test]: ./ref/bar_test.png "Bar test data"
[i_before_YUV]: ./ref/before_YUV.png "Before YUV"
[i_after_YUV]: ./ref/after_YUV.png "After YUV"
[i_before_ImageDataGen]: ./ref/before_ImageDataGen.png "Before ImageDataGen"
[i_after_ImageDataGen]: ./ref/after_ImageDataGen.png "After ImageDataGen"
[i_lenet]: ./ref/lenet.png "LeNet model"
[new_signs]: ./ref/new_signs.png "New Traffic Signs"
[new_signs_prediction]: ./ref/new_signs_prediction.png "New Traffic Signs Prediction"
[new_0]: ./ref/new_0.png "Traffic Sign 1"
[new_1]: ./ref/new_1.png "Traffic Sign 2"
[new_2]: ./ref/new_2.png "Traffic Sign 3"
[new_3]: ./ref/new_3.png "Traffic Sign 4"
[new_4]: ./ref/new_4.png "Traffic Sign 5"
[new_5]: ./ref/new_5.png "Traffic Sign 6"
[new_6]: ./ref/new_6.png "Traffic Sign 7"
[new_7]: ./ref/new_7.png "Traffic Sign 8"
[new_8]: ./ref/new_8.png "Traffic Sign 9"
[new_9]: ./ref/new_9.png "Traffic Sign 10"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/shingte/self-driving-car-nd/blob/master/P02-Traffic-Sign-Classifier/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python commands to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a random sample of traffic signs in each calss. 
![alt text][i_sample]

Below are bar chars to show the distribution of data in each class from the train/validation/test dataset.

![alt text][i_bar_train]

![alt text][i_bar_valid]

![alt text][i_bar_test]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to YUV and only use the Y channel.This is inspired by the paper from Sermanet and LeCun.

Here is an example of a traffic sign image before and after RGB2YUV.

![alt text][i_before_YUV]     ![alt text][i_after_YUV]

I also normalized the image data because neural networks works better when the input data is normailized.

I decided to generate additional data because that improves the accuracy of the results.

To add more data to the the data set, I used the ImageDataGenerator utility in Keras.  

Here is an example of an original image and an augmented image:

Before data augmentation -

![alt text][i_before_ImageDataGen]     

After data augmentation -

![alt text][i_after_ImageDataGen]

The difference between the original data set and the augmented data set is the following -

* Images are shuffled.

* Images can be rotated within 15 degree.

* Zooming range is with in 20%.

* Width and height can be shifted within 20%.
                                
                                


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I decide to use the LeNet model used in the lecture. 
This is the model architecture of LeNet -
![alt text][i_lenet]

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 image   							    | 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6 	|
| Activation RELU	    |												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 3x3	    | 1x1 stride, same padding, outputs 10x10x16 	|
| Activation RELU	    |												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Flatten         		| output 400  					    		    | 
| Fully connected		| output 120						        	|
| Activation RELU	    |												|
| Fully connected		| output 84 						        	|
| Activation RELU	    |												|
| Fully connected		| output 43 						        	|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an the following hyperparameters -

* Type of optimiser: AdamOptimizer
* Loss function: cross entropy
* Batch size: 128
* Batch per Epoch: 5000
* Training Epochs: 30
* Learning rate: 0.001


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.982
* test set accuracy of 0.953

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

I chose LeNet first, as it is the model used in the class exercise.

* What were some problems with the initial architecture?

I found LeNet worked reasonably well for this project, so decided to use it to complete the project.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I tried to add dropout layers, but didn't find it helps in this case, therefore stick with the original design.

* Which parameters were tuned? How were they adjusted and why?

I tried different values of learning rate, such as 0.01, 0.0001 etc. Pick 0.001 as it converge reasonably well compare to other values.
I also change the traing epochs value, to allow enough runs to stablize the training.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

I did several experiments and decide to use the LeNet architecture unchanged.
There are some other models I tried might work better, however I decided to present this model becuase of it's relevence to the class and simplicity.

If a well known architecture was chosen:
* What architecture was chosen?

LeNet
* Why did you believe it would be relevant to the traffic sign application?

There's a paper "Traffic Sign Recognition with Multi-Scale Convolutional Networks" from Pierre Sermanet and Yann LeCun that already described this model can apply to traffice sign recongition with good accuracy.  

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

All accuracies are 95% or above.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web:

![alt text][new_signs] 

image 1 - the numbers in the sign might be diffcult to classify.
image 2 - the background is dark, and there's a partial sign underneath it.
image 3 - the number may be confused with other speed limit signs.
image 5 - the shape of the sign may be confsued by the sing underneath and the backound.
image 7 - the picture is blurry.
image 8 - the picture is blurry.
image 10 - background is very bright.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction. Please compare the predictions in the picture with the labels in the picture above.

![alt text][new_signs_prediction] 

The model was able to correctly guess 9 of the 10 traffic signs, which gives an accuracy of 90%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 145th cell of the Ipython notebook.

These are the predictions with top 5 probabilities -

New image 1 -

![alt text][new_0] 

New image 2 -

![alt text][new_1] 

New image 3 -

![alt text][new_2] 

New image 4 -

![alt text][new_3] 

New image 5 -

![alt text][new_4] 

New image 6 -

![alt text][new_5] 

New image 7 -

![alt text][new_6] 

New image 8 -

![alt text][new_7] 

New image 9 -

![alt text][new_8] 

New image 10 -

![alt text][new_9] 


