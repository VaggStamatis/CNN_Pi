import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten,Dense
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator




#adding the sequential to combine/stack all the layers
classifier = Sequential()

#	convolutional layer with 16 filters of 3x3
#and image input size of 28x28 and in grayscale(1) for rgb use (3)
#using an activation function ReLU
#pooling layer using a max pooling 2x2 filter
classifier.add(Conv2D(16,(3,3), input_shape=(28,28,1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2))) 
classifier.add(Conv2D(32,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2))) 

#convert the 2d output array into a 1d array in order to pass through the fully connected layer 
#doing so by using the flatten library import
classifier.add(Flatten())
classifier.add(Dense(64, activation='relu'))

#in order to predict a class, we need another activation function that is called softmax
#Softmax is callculating the probability of each class the image belongs to
#here we use 10 cause we have 10 classses one from each number from 0 to 9
classifier.add(tf.keras.layers.Dense(10, activation='softmax'))

#backpropagation with "Adam" optimizer 
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen=ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
#data loading testing
training_set= train_datagen.flow_from_directory('/home/pi/Desktop/MNIST Dataset JPG format/MNIST - JPG - training',
												target_size=(28,28),
												batch_size=32,
												color_mode='grayscale',
												class_mode='categorical',
												subset='training')										

#data loading testing
testing_set= train_datagen.flow_from_directory('/home/pi/Desktop/MNIST Dataset JPG format/MNIST - JPG - testing',
												target_size=(28,28),
												batch_size=32,
												color_mode='grayscale',
												class_mode='categorical',
												subset='validation')

#epoch is how many times we are gonna pass the data through the cnn
#the batch tells us that we are gonna pass the data as baches of 32 on each epoch
#Pass the data to the neural network we created above
classifier.fit(training_set, steps_per_epoch=10, epochs=150, validation_data=testing_set, validation_steps=10)
classifier.save('CNN_NUMBERS.h5')
