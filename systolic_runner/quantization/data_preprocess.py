#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------------------------------
# Copyright (c) Microsoft, Intel Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from PIL import Image
import os
import sys
import numpy as np

def set_preprocess(preprocess_func_name):
    '''
    Set up the data preprocess function name and function dict. 
        parameter preprocess_func_name: name of the preprocess function 
        return: function pointer 
    '''
    funcdict = {'preprocess_method1': preprocess_method1, 
                'preprocess_method2': preprocess_method2}
    return funcdict[preprocess_func_name]

def preprocess_method1(image_filepath, height, width):
    '''
    Resizes image to NCHW format. Image is scaled to range [-1, 1].
    This method is suitable for the mobilenet model from mlperf inference git repo.
        parameter image_filepath: path to image files
        parameter height: image height in pixels
        parameter width: image width in pixels
        return: matrix characterizing image
    '''
    pillow_img = Image.open(image_filepath).resize((width, height))
    input_data = np.float32(pillow_img)/127.5 - 1.0 # normalization
    input_data -= np.mean(input_data) # normalization
    nhwc_data = np.expand_dims(input_data, axis=0)
    nchw_data = nhwc_data.transpose(0, 3, 1, 2) # ONNX Runtime standard
    return nchw_data

def preprocess_mxnet_raw(raw_image):
    r_channel = raw_image[:, 0, :, :]
    g_channel = raw_image[:, 1, :, :]
    b_channel = raw_image[:, 2, :, :]

    r_channel = ((r_channel/255.0) - 0.485)/0.229
    g_channel = ((g_channel/255.0) - 0.456)/0.224
    b_channel = ((b_channel/255.0) - 0.406)/0.225
    return np.stack([b_channel, g_channel, r_channel], axis=1)

def preprocess_caffe_raw(raw_image):
    r_channel = raw_image[:, 0, :, :]
    g_channel = raw_image[:, 1, :, :]
    b_channel = raw_image[:, 2, :, :]

    b_channel = (b_channel - 103.94)*0.017
    g_channel = (g_channel - 116.78)*0.017
    r_channel = (r_channel - 123.68)*0.017
    
    return np.stack([b_channel, g_channel, r_channel], axis=1)

def preprocess_caffe2_raw(raw_image):
    r_channel = raw_image[:, 0, :, :]
    g_channel = raw_image[:, 1, :, :]
    b_channel = raw_image[:, 2, :, :]

    b_channel = (b_channel - 103.939)
    g_channel = (g_channel - 116.779)
    r_channel = (r_channel - 123.68)
    
    return np.stack([b_channel, g_channel, r_channel], axis=1)

# The version below is given for rcnn from model zoo
# For version exported from pytorch, use the commented one below
def preprocess_rcnn_raw(raw_image):
    r_channel = raw_image[0, :, :]
    g_channel = raw_image[1, :, :]
    b_channel = raw_image[2, :, :]

    b_channel = (b_channel - 102.9801)
    g_channel = (g_channel - 115.9465)
    r_channel = (r_channel - 122.7717)
    
    return np.stack([b_channel, g_channel, r_channel])

# def preprocess_rcnn_raw(raw_image):
#     r_channel = raw_image[:, 0, :, :]
#     g_channel = raw_image[:, 1, :, :]
#     b_channel = raw_image[:, 2, :, :]

#     r_channel = ((r_channel/255.0) - 0.485)/0.229
#     g_channel = ((g_channel/255.0) - 0.456)/0.224
#     b_channel = ((b_channel/255.0) - 0.406)/0.225
    
#     return np.stack([r_channel, g_channel, b_channel], axis=1)

def preprocess_method2(image_filepath, height, width):
    '''
    Resizes and normalizes image to NCHW format. 
    This method is suitable for the resnet50 model from mlperf inference git repo. 
        parameter image_filepath: path to image files
        parameter height: image height in pixels
        parameter width: image width in pixels
        return: matrix characterizing image
    '''
    pillow_img = Image.open(image_filepath).resize((width, height))
    input_data = np.float32(pillow_img) - np.array([123.68, 116.78, 103.94], dtype=np.float32)
    nhwc_data = np.expand_dims(input_data, axis=0)
    nchw_data = nhwc_data.transpose(0, 3, 1, 2) # ONNX Runtime standard
    return nchw_data


def load_batch(images_folder, height, width, preprocess_func_name, size_limit=0):
    '''
    Loads a batch of images
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    parameter preprocess_func_name: name of the preprocess function
    return: list of matrices characterizing multiple images
    '''
    image_names = os.listdir(images_folder)
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []
    
    preprocess_func = set_preprocess(preprocess_func_name)
    for image_name in batch_filenames:
        image_filepath = images_folder + '/' + image_name
        nchw_data = preprocess_func(image_filepath, height, width)
        unconcatenated_batch_data.append(nchw_data)
    batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    return batch_data