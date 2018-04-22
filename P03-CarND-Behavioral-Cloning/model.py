import argparse
import cv2
import csv
import os
import math
import pandas

import numpy as np
import tensorflow as tf
import keras # Use Keras 2.x version
from keras import losses
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Lambda, Conv2D, MaxPooling2D, Cropping2D
from keras.layers.advanced_activations import ELU
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


"""
Utils
"""
def show_cv2(title, img):
	cv2.imshow(title,img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def show_dataset(X,y,y_pred=None):
    '''
    format the data from the dataset (image, steering_angle) and display
    '''
    for i in range(len(X)):
        if y_pred is not None:
            img = process_img_to_show(X[i], y[i], y_pred[i], i)
        else: 
            img = process_img_to_show(X[i], y[i], None, i)
        show_cv2(' ',img)       


def process_img_to_show(image, angle, pred_angle, frame):
    '''
    Used by show_dataset method to format image prior to displaying. 
    Converts colorspace back to original BGR, 
    applies text to display steering angle and frame number 
    (within batch to be visualized), 
    and applies lines representing steering angle 
    and model-predicted steering angle (if available) to image.
    '''    

    #font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
    img = cv2.resize(img,None,fx=3, fy=3, interpolation = cv2.INTER_CUBIC)
    h,w = img.shape[0:2]
    # apply text for frame number and steering angle
    cv2.putText(img, 'frame: ' + str(frame), org=(2,18), fontFace=font, fontScale=.5, color=(200,100,100), thickness=1)
    cv2.putText(img, 'angle: ' + str(angle), org=(2,33), fontFace=font, fontScale=.5, color=(200,100,100), thickness=1)
    # apply a line representing the steering angle
    cv2.line(img,(int(w/2),int(h)),(int(w/2+angle*w/4),int(h/2)),(0,255,0),thickness=4)
    if pred_angle is not None:
        cv2.line(img,(int(w/2),int(h)),(int(w/2+pred_angle*w/4),int(h/2)),(0,0,255),thickness=4)
    return img
    


"""
Load data and store in memory
"""
def load_data(dir):
	samples = []
	with open(dir+'driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		for i in rangle(args.skip_lines):
			next(reader)
		for line in reader:
			samples.append(line)
	return samples
	#train_samples, validation_samples = train_test_split(samples, test_size=args.split_ratio)
	#return train_samples, validation_samples

"""
Preprocess data for data balancing
add paths and steering angles from center camera
then add left and right image paths with steering correctioin
"""
def process_data(dir):
	csvfile = 'driving_log_add.csv' if args.use_lane_info else 'driving_log.csv'
	csvpath = dir+csvfile
	colnames = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed', 's_center', 's_left', 's_right']
	correction = args.correction
	data = pandas.read_csv(csvpath, skiprows=args.skip_lines, names=colnames)
	paths = data.center.tolist()
	steerings = data.steering.tolist()
	steerings_left = [min(x+correction, 1) for x in steerings]
	steerings_right = [max(x-correction, -1) for x in steerings]
	
	# add left and right camera images to list with steering correction
	paths.extend(data.left.tolist())
	steerings.extend(steerings_left)
	
	paths.extend(data.right.tolist())
	steerings.extend(steerings_right)
	
	for i in range(len(paths)):
		paths[i] = dir+paths[i].strip().replace('\\', '/')

	print('# of images=', len(paths))
	print('# of steerings=', len(steerings))
	print(paths[0])
	print(paths[-1])

	return paths, steerings
	
"""
Create 21 bins for steering angles -1 to 1. 
bin 10 is for angle 0 only
"""
def create_bins(paths, steerings, rebalance, plot_histogram=True):
	n_bins = 21
	hist = [0] * n_bins  # [[] for i in range(21)]
	
	for j in range(len(steerings)):
		steer=steerings[j]
		idx = get_bin(steer)
		hist[idx] += 1

	if plot_histogram:
		show_histogram(hist)

	if rebalance:
		paths, steerings = delete_samples(paths, steerings, hist)
		if plot_histogram:
			create_bins(paths, steerings, False, plot_histogram)
			
	return paths, steerings
	
def get_bin(steering):
	if abs(steering) < 1.0e-7:
		bin = 10
	elif steering>0:
		bin = int(math.ceil(steering*10+10))
	else:
		bin = int(math.floor(steering*10+10))

	return bin

def show_histogram(hist):
	x_pos = [i-10 for i in range(21)]
	plt.bar(range(21), hist)
	plt.xticks(range(21), x_pos)
	plt.xlabel('Steering angle (x10)')
	plt.show()

	return

def delete_samples(paths, steerings, hist):
	n_bins = len(hist)
	n_samples = len(steerings)
	delete_list = []
	delete_probs = [0.0] * n_bins
	avg_num_in_bin = n_samples / n_bins
	target = avg_num_in_bin # * 0.5
	for i in range(n_bins):
		if hist[i] > target:
			delete_probs[i] = 1 - target / hist[i]
	for i in range(n_samples):
		ibin = get_bin(steerings[i])
		if (delete_probs[ibin] > 0):
			if np.random.rand() < delete_probs[ibin]:
				delete_list.append(i)
	
	paths = np.delete(paths, delete_list)
	steerings = np.delete(steerings, delete_list)

	return paths, steerings


"""
Image processing
Load images from center, left, right cameras, 
Convert from BGR2RGB - cv2 read in BGR, drive.py takes RGB format
Corrections on steering values for left/right cameeas.
"""
def process_samples(image_paths, steerings):
	#correct the steering for images from left(+) and right(-)
	#correction = args.correction
	images = []
	for i in range(len(image_paths)):
		file = image_paths[i]
		if not os.path.isfile(file):
			print('Error! file ', file, ' does not exist')
			return images, steerings

		image = cv2.imread(file)

		# It's very important to do BGR2RGB, as drive.py takes RGB image format
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR2YUV)  
		# crop and resize to 64x64x3
		# image = image[60:140, 0:320] # cv2.resize(image[60:140,:], (64,64))
		# show_cv2('crop', image)
		images.append(image)

	if args.debug:
		show_dataset(images, steerings)

	images, steerings = flip_images(images, steerings)
	return images, steerings



def flip_images(images, steerings):
	add_images = []
	add_steerings = []
	for image, angle in zip(images, steerings):
		if angle < args.flip_threshold: 
			continue
		add_images.append(image)
		add_steerings.append(angle)
		flip_image = cv2.flip(image, 1)
		flip_steering = angle * -1.0
		add_images.append(flip_image)
		add_steerings.append(flip_steering)

	X = np.array(add_images)
	y = np.array(add_steerings)
	return X, y


"""
Generate the required images and steerings for training/validation
samples is a list of pairs (image_path, steering).
"""
def generator(X_samples, y_samples, batch_size=32):
    num_samples = len(X_samples)
 
    while True: # Loop forever so the generator never terminates
        X_samples, y_samples = shuffle(X_samples, y_samples)
        for offset in range(0, num_samples, batch_size):
            X_batch = X_samples[offset:offset+batch_size]
            y_batch = y_samples[offset:offset+batch_size]
            X, y = process_samples(X_batch, y_batch)
            yield shuffle(X, y)

"""
Get one of the following models -
1 - nvida_model
2 - commaai_model
3 - lenet_model
Preprocessing - 
* normalization 
* crop top 70 pixels and bottom 20 pixels
"""
def get_model():
	default_model=args.default_model
	model = Sequential()
	model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160,320,3))) #(64,64,3)))
	model.add(Cropping2D(cropping=((70,20),(0,0))))
	if (default_model==2):
		return commaai_model(model)
	elif (default_model==3):
		return lenet_model(model)

	return nvidia_model(model)

""" 
LeNet model 
"""
def lenet_model(model):
	print('LeNet_model used')
	model.add(Conv2D(6, (5, 5), activation='relu'))
	model.add(MaxPooling2D())
	model.add(Dropout(args.keep_prob))
	model.add(Conv2D(16, (5, 5), activation='relu'))
	model.add(MaxPooling2D())
	model.add(Dropout(args.keep_prob))
	model.add(Flatten())
	model.add(Dense(120)) 
	model.add(Dense(84))
	model.add(Dense(1)) 
	return model

""" 
Model from comma.ai 
"""
def commaai_model(model):
	print('CommaAi_model used')
	model.add(Conv2D(16, (8, 8), strides=(4, 4), padding='same'))
	model.add(ELU())
	model.add(Conv2D(32, (5, 5), strides=(2, 2), padding='same'))
	model.add(ELU())
	model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
	model.add(Flatten())
	model.add(Dropout(args.keep_prob)) # 0.2
	model.add(ELU())
	model.add(Dense(512))
	model.add(Dropout(args.keep_prob)) # 0.5
	model.add(ELU())
	model.add(Dense(1))
	return model

""" 
Nvidia End to End Self-driving Car CNN model 
"""
def nvidia_model(model):
	print('nvidia_model used')
	model.add(Conv2D(24, (5, 5), activation='relu', strides=(2, 2)))
	model.add(Conv2D(36, (5, 5), activation='relu', strides=(2, 2)))
	model.add(Conv2D(48, (5, 5), activation='relu', strides=(2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))
	return model

""" 
Train the model
Add config to avoid tensorflow to claim all the GPU resource. 
Without the all_growth option, sometimes it gets cudnn out of memory error in Windows
"""
def train_model(): #X_train, y_train):
	paths, steerings = process_data(args.data_dir) 
	if args.data_dir2 != ' ':
		dir = args.data_dir2
		paths_add, steerings_add = process_data(args.data_dir2)
		paths.extend(paths_add)
		steerings.extend(steerings_add)

	# show image histograms
	# rebalancing before processing the images
	paths, steerings = create_bins(paths, steerings, args.rebalance, True)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess= tf.Session(config=config)

	model = get_model()
	print(model.summary())

	model.compile(loss=args.loss, optimizer='adam')
	if args.use_generator:
		train_X, validation_X, train_y, validation_y = train_test_split(paths, steerings, test_size=args.split_ratio)
		print('Train samples: {}'.format(len(train_X)))
		print('Validation samples: {}'.format(len(validation_X)))
		batch_size=args.batch_size
		train_generator = generator(train_X, train_y, batch_size)
		validation_generator = generator(validation_X, validation_y, batch_size)
		model.fit_generator(train_generator, steps_per_epoch= len(train_X)/batch_size,
							validation_data=validation_generator, validation_steps=len(validation_X)/batch_size, 
							epochs=args.epochs, verbose = 1)
	else:
		X_train, y_train = process_samples(paths, steerings)
		model.fit(X_train, y_train, validation_split=args.split_ratio, shuffle=True, 
					batch_size=args.batch_size, epochs=args.epochs)


	model.save('model.h5')
	print('model.h5 saved')



"""
Converts a string to y/n boolean
"""
def s2b(s):
	s = s.lower()
	return s == 'true' or s == 'yes' or s == 'y' or s == '1'

"""
parameters and defaults setting
"""
def set_args():
    parser = argparse.ArgumentParser(description='Build and train Behavioral Cloning Model')
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default='./data/')
    parser.add_argument('-e', help='extra data directory', dest='data_dir2', type=str, default=' ')
    parser.add_argument('-s', help='# of lines to skip in csv file', dest='skip_lines', type=int, default=1)
    parser.add_argument('-u', help='use lane info (Y/n)', dest='use_lane_info', type=s2b, default='n')
    parser.add_argument('-g', help='use data generator (y/N)', dest='use_generator', type=s2b, default='y')
    parser.add_argument('-r', help='rebalance data (y/N)', dest='rebalance', type=s2b, default='n')
    parser.add_argument('-c', help='steering correction', dest='correction', type=float, default=0.2)
    parser.add_argument('-f', help='flip threshold', dest='flip_threshold', type=float, default=0)
    parser.add_argument('-m', help='model: 1=Nvidia, 2=CommaAi, 3=LeNet', dest='default_model',    type=int,   default=1)
    parser.add_argument('-t', help='train/validation split ratio', dest='split_ratio', type=float, default=0.2)
    parser.add_argument('-k', help='keep probability', dest='keep_prob', type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs', dest='epochs', type=int, default=10)
    parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=128)
    parser.add_argument('-l', help='loss function', dest='loss', type=str, default='mse')
    parser.add_argument('-z', help='debug mode (y/N)', dest='debug', type=s2b, default='n')
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    return args

""" 
Main function
Use options and defaults to handle the inputs
"""
args = set_args()

def main():
	train_model()

if __name__ == '__main__':
    main()
