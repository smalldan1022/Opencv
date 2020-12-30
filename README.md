# Opencv cookbook


[![Logo](https://github.com/smalldan1022/Unet/blob/master/pics/Dan_Logo_3.png)](https://www1.cgmh.org.tw/intr/intr2/c3sf00/caim/home/index)

[![Website online](https://img.shields.io/website/http/huggingface.co/transformers/index.html.svg?down_color=red&down_message=offline&up_message=online)](https://www1.cgmh.org.tw/intr/intr2/c3sf00/caim/home/news)


## Table of Content
>[1. Introduction](#introduction)

>[2. Explanation](#Explanation)

>[3. Check list](#Check-list)

>[4. Pre-requisites](#Pre-requisites)

>[5. Overlap image](#Overlap-image)

>[6. Registration](#Registration)

>[7. Color map](#Color-map)


## Introduction

Unet is an architecture used widely in medical field to solve the high resolution problem on most of the medical images. There are lots of version of the Unet architecture. However, it is the pytorch version or the old keras version, and thus can't get the advantage of the new vesrion tensorflow.Therefore, I provide a tensorflow 2.2.0 version and the simple architecture code for people to use and modify.


## Explanation


    1.MakeDataset.py
        For the purpose of making a dataset by the tensorflow iterator, tf.data.Dataset, which is faster than 
        using python list to read images.

    2.Unet_model.py
        To make the Unet model by using tensorflow inheritance, tf.keras.Model. You can use the Model func for 
        sure, it's up to you.

    3.utils.py
        Provide the utilities for you to visualize the results of your AI model. There are also some image 
        processing methods for you to enhance the segmentation results using the opencv.
        
    4.Unet_main.py
        The main function to control the whole process of the Unet model training. There are lots of thing you 
        can tune yourself in it, like the hyperparameters and the callbacks. Feel free to modify it yourself.



## Check list

Be sure to check all the steps below to make sure nothing goes wrong in your training process.

- [x] Data leakage -> divide the dataset by patients
- [x] Different dataset distribution -> use five fold method
- [x] Save all the train values -> don't miss any train/valid values or the initial training hyperparameters
- [x] Plot the result for you to check -> check whether the model got the right features or not  



## Pre-requisites

## Overlap image

## Registration

## Color map
