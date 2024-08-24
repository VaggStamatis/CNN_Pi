# CNN_Pi
Implementing a Convolutional Neural Network on a **Raspberry Pi model B+** using Python. Based on the Udemy Course "Embedded Systems Bootcamp: RTOS, Iot, AI, Vision and FPGA", we implemented a Convolutional Neural Network trained to recognise hand written numbers between 0 and 9.

## Data Set
For training and later evaluating the nueral network we used the MNIST-JPG data set.You can find the data set and download it from the following link:
https://github.com/teavanist/MNIST-JPG

## Required Packages and Libraries
In order to build such a CNN you need to install certain packages, listed below on your machine and create a virtual environment. 
- libhdf5-dev
- libc-ares-dev
- libeigen3-dev
- h5py (use version 2.9.0)
- openmpi-bin libopenmpi-dev
- libaltlas-base-dev

Python Libraries installed via pip
```sh
$ pip installvirtualenv
$ pip install numpy==1.20.0
$ pip install tensorflow
```
🆘 For Tensorflow versions above 2.x , Tensorflow comes with a pre installed Keras version so it's not necessary to download and install it manually.

# Training 
The CNN is trained via the training set of the MNIST data set using 60.000 photos of hand written numbers between 0 and 9. After 150 epoches we can see in the image below that our CNN reached an **Accuracy = 0.89**

<img src="https://github.com/VaggStamatis/CNN_Pi/blob/main/epoch%20and%20accuracy.png" width="800">

# Validation
After training the CNN we can save it for later use with the **_.h5_** file format. To validate that the CNN can classify a random input of a hand written number correctly we built the validation.py script. By loading the previously saved CNN and giving it a input of a random number '2' from the dataset we can see in the image below that the CNN correctly predicts that tha number is a '2' with a **_probability = 0.881_** or **_88%_**

<img src= "https://github.com/VaggStamatis/CNN_Pi/blob/main/validation.png" width="800">

# Creator 
*Evagelos Stamatis* email: [evanstamatis@gmail.com]