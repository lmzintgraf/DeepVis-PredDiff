# -*- coding: utf-8 -*-
"""
From this script, experiments for ImageNet pictures can be started.
See "configuration" below for the different possible settings.
The results are saved automatically to the folder ./results

It is recommended to run caffe in gpu mode when overlapping is set
to True, otherwise the calculation will take a very long time.

@author: Luisa M Zintgraf
"""

from prediction_difference_analysis import PredDiffAnalyser
import utils_classifiers as utlC
import utils_data as utlD
import utils_sampling as utlS
import utils_visualise as utlV
import sensitivity_analysis_caffe as SA

import numpy as np
import time
import os
import matplotlib
# the following is needed to avoid some error that can be thrown when
# using matplotlib.pyplot in a linux shell
matplotlib.use('Agg')

# ------------------------ CONFIGURATION ------------------------
# -------------------> CHANGE SETTINGS HERE <--------------------

# pick neural network to run experiment for (alexnet, googlenet, vgg)
netname = 'googlenet'

# pick for which layers the explanations should be computet
# (names depend on network, output layer is usually called 'prob')
blobnames = ['prob']
# blobnames = ['conv1','conv2','conv3','conv4','conv5','fc6','fc7','fc8','prob'] # alexnet
# blobnames = ['conv1/7x7_s2', 'conv2/3x3_reduce', 'conv2/3x3', 'conv2/norm2', 'inception_3a/output', 'inception_3b/output', 'inception_4a/output', 'inception_4b/output', 'inception_4c/output', 'inception_4d/output', 'inception_4e/output','inception_5a/output', 'inception_5b/output', 'loss3/classifier', 'prob']
# blobnames = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7', 'fc8', 'prob'] # vgg

# is caffe running in gpu mode?
gpu = False

# pick image indices which are analysed (in alphabetical order as in the ./data folder) [0,1,2,...]
# (if None, all images in './data' will be analysed)
test_indices = None

# window size (i.e., the size of the pixel patch that is marginalised out in each step)
win_size = 10               # k in alg 1 (see paper)

# indicate whether windows should be overlapping or not
overlapping = False

# settings for sampling 
sampl_style = 'conditional'  # choose: conditional / marginal
num_samples = 10
padding_size = 2            # important for conditional sampling,
                            # l = win_size+2*padding_size in alg 1
                            # (see paper)

# set the batch size - the larger, the faster computation will be
# (if caffe crashes with memory error, reduce the batch size)
batch_size = 128


# ------------------------ SET-UP ------------------------

utlC.set_caffe_mode(gpu=gpu)

net = utlC.get_caffenet(netname)

# get the data
X_test, X_test_im, X_filenames = utlD.get_imagenet_data(net=net)

# get the label names of the 1000 ImageNet classes
classnames = utlD.get_imagenet_classnames()

if not test_indices:
    test_indices = [i for i in range(X_test.shape[0])]      

# make folder for saving the results if it doesn't exist
path_results = './results/'
if not os.path.exists(path_results):
    os.makedirs(path_results)          
          
# ------------------------ EXPERIMENTS ------------------------

# change the batch size of the network to the given value
net.blobs['data'].reshape(batch_size, X_test.shape[1], X_test.shape[2], X_test.shape[3])

# target function (mapping input features to output probabilities)
target_func = lambda x: utlC.forward_pass(net, x, blobnames)

# for the given test indices, do the prediction difference analysis
for test_idx in test_indices:
      
    # get the specific image (preprocessed, can be used as input to the target function)
    x_test = X_test[test_idx]
    # get the image for plotting (not preprocessed)
    x_test_im = X_test_im[test_idx]
    # prediction of the network
    y_pred = np.argmax(utlC.forward_pass(net, x_test, ['prob']))
    y_pred_label = classnames[y_pred]
                           
    # get the path for saving the results
    if sampl_style == 'conditional':
        save_path = path_results+'{}_{}_winSize{}_condSampl_numSampl{}_paddSize{}_{}'.format(X_filenames[test_idx],y_pred_label,win_size,num_samples,padding_size,netname)
    elif sampl_style == 'marginal':
        save_path = path_results+'{}_{}_winSize{}_margSampl_numSampl{}_{}'.format(X_filenames[test_idx],y_pred_label,win_size,num_samples,netname)

    if os.path.exists(save_path+'.npz'):
        print('Results for ', X_filenames[test_idx], ' exist, will move to the next image. ')
        continue
                 
    print("doing test...", "file :", X_filenames[test_idx], ", net:", netname, ", win_size:", win_size, ", sampling: ", sampl_style)

    # compute the sensitivity map
    layer_name = net.blobs.keys()[-2] # look at penultimate layer (like in Simonyan et al. (2013))
    sensMap = SA.get_sensitivity_map(net, x_test[np.newaxis], layer_name, np.argmax(target_func(x_test)[-1][0]))

    start_time = time.time()
    
    if sampl_style == 'conditional':
        sampler = utlS.ConditionalSamplerImagenet(win_size=win_size, padding_size=padding_size, image_dims=net.crop_dims, netname=netname)
    elif sampl_style == 'marginal':
        sampler = utlS.MarginalSamplerImagenet(X_test, net)
        
    pda = PredDiffAnalyser(x_test, target_func, sampler, num_samples=num_samples, batch_size=batch_size)
    pred_diff = pda.get_rel_vect(win_size=win_size, overlap=overlapping)
    
    # plot and save the results
    utlV.plot_results(x_test, x_test_im, sensMap, pred_diff[0], target_func, classnames, test_idx, save_path)
    np.savez(save_path, *pred_diff)
    print("--- Total computation took {:.4f} minutes ---".format((time.time() - start_time)/60))
