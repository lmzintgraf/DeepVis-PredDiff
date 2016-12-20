# -*- coding: utf-8 -*-
"""

Utility methods for handling the ImageNet data:
    get_imagenet_data(net, preprocess)
    get_imagenet_classnames()
    
"""

import numpy as np
import os
import PIL


path_data = "./data"


def get_imagenet_data(net):
    """
    Returns a small dataset of ImageNet data.
    Input:  
            net         a neural network (caffe model)
    Output:
            X           the feature values, in a matrix 
                        (numDatapoints, [imageDimensions])
            X_im        the features as uint8 values, to display
                        using plt.imshow()
            X_filenames the filenames, with the dots removed
    """

    # get a list of all the images (note that we use networks trained on ImageNet data)
    img_list = os.listdir(path_data)

    # throw away files that are not in the allowed format (png or jpg)
    for img_file in img_list[:]:
        if not (img_file.endswith(".png") or img_file.endswith(".jpg")):
            img_list.remove(img_file)
        
    # fill up data matrix
    img_dim = net.crop_dims
    X = np.empty((0, img_dim[0], img_dim[1], 3))
    X_filenames = []
    for i in range(len(img_list)):
        np_img = np.float32(PIL.Image.open('{}/{}'.format(path_data, img_list[i])))
        if np_img.shape[0] >= img_dim[0] and np_img.shape[1] >= img_dim[1]:
            o = 0.5*np.array([np_img.shape[0]-img_dim[0], np_img.shape[1]-img_dim[1]])
            X = np.vstack((X, np_img[o[0]:o[0]+img_dim[0], o[1]:o[1]+img_dim[1], :][np.newaxis]))
            X_filenames.append(img_list[i].replace(".",""))
        else:
            print("Skipped ",img_list[i],", image dimensions were too small.")

    # the number of images we found in the folder
    num_imgs = X.shape[0]

    # cast to image values that can be displayed directly with plt.imshow()
    X_im = np.uint8(X)
        
    # preprocess
    X_pre = np.zeros((X.shape[0], 3, img_dim[0], img_dim[1]))
    for i in range(num_imgs):
        X_pre[i] = net.transformer.preprocess('data', X[i])
    X = X_pre
        
    return X, X_im, X_filenames


def get_imagenet_classnames():
    """ Returns the classnames of all 1000 ImageNet classes """
    return np.loadtxt(open(path_data+'/ilsvrc_2012_labels.txt'), dtype=object, delimiter='\n')
