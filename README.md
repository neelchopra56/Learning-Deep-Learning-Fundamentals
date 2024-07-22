# Learning-Deep-Learning-Fundamentals

## Overview

This repository contains a series of deep learning experiments focused on image classification tasks. The primary objectives of this project are:

1. Data preprocessing and augmentation.
2. Implementing a basic Convolutional Neural Network (CNN) for classifying images of cats and dogs.
3. Utilizing transfer learning with the VGG-16 model for the same classification task.
4. Implementing a pre-built MobileNet model.
5. Using MobileNet for predicting sign language digits.

## Table of Contents

- [Data Preprocessing](#data-preprocessing)
- [Basic CNN Model](#basic-cnn-model)
- [Transfer Learning with VGG-16](#transfer-learning-with-vgg-16)
- [MobileNet Model](#mobilenet-model)
- [Sign Language Digits Classification](#sign-language-digits-classification)
- [Data Augmentation](#data-augmentation)
- [Results](#results)


## Data Preprocessing

Data preprocessing is a crucial step in any machine learning project. In this project, the following preprocessing steps were performed:

- **Data Loading**: Loading images from directories.
- **Data Shuffling**: Shuffling the data to ensure randomness.
- **Data Scaling**: Scaling the pixel values to a range of 0 to 1.
- **Data Splitting**: Splitting the data into training, validation, and test sets.

## Basic CNN Model

A simple Convolutional Neural Network (CNN) was implemented to classify images of cats and dogs. The architecture of the CNN is as follows:

- **Conv2D Layer**: Convolutional layer with ReLU activation.
- **MaxPool2D Layer**: Max pooling layer to reduce spatial dimensions.
- **Flatten Layer**: Flattening the 2D arrays into a 1D vector.
- **Dense Layer**: Fully connected layer with softmax activation for classification.

## Transfer Learning with VGG-16

Transfer learning was utilized to improve the performance of the image classification task. The VGG-16 model, pre-trained on the ImageNet dataset, was used. The following steps were performed:

- **Model Loading**: Loading the pre-trained VGG-16 model.
- **Model Modification**: Modifying the last layer to fit the binary classification task (cats vs. dogs).
- **Model Freezing**: Freezing the layers of VGG-16 to retain the pre-trained weights.
- **Model Training**: Training the modified model on the cats vs. dogs dataset.

## MobileNet Model

The MobileNet model, known for its efficiency and small size, was implemented for image classification tasks. The following steps were performed:

- **Model Loading**: Loading the pre-built MobileNet model.
- **Model Prediction**: Using the model to predict the class of new images.

## Sign Language Digits Classification

MobileNet was fine-tuned to classify sign language digits. The following steps were performed:

- **Data Preparation**: Organizing the dataset into train, validation, and test sets.
- **Model Modification**: Modifying the MobileNet model to fit the 10-class classification task.
- **Model Training**: Training the modified model on the sign language digits dataset.

## Data Augmentation

Data augmentation techniques were applied to increase the size and variability of the training dataset. The following augmentations were used:

- **Rotation**: Rotating the images.
- **Width and Height Shift**: Shifting the images horizontally and vertically.
- **Shear**: Shearing the images.
- **Zoom**: Zooming in and out of the images.
- **Channel Shift**: Shifting the color channels.
- **Horizontal Flip**: Flipping the images horizontally.

## Results

The results of the experiments are as follows:

- **Basic CNN Model**: Achieved an accuracy of X% on the test set.
- **VGG-16 Transfer Learning**: Achieved an accuracy of Y% on the test set.
- **MobileNet Model**: Achieved an accuracy of Z% on the test set.
- **Sign Language Digits Classification**: Achieved an accuracy of W% on the test set.
