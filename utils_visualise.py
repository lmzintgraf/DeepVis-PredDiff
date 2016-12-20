# -*- coding: utf-8 -*-
"""
Some utility functions for visualisation, not documented properly
"""

from skimage import color
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pylab


def plot_results(x_test, x_test_im, sensMap, predDiff, tarFunc, classnames, testIdx, save_path):
    '''
    Plot the results of the relevance estimation
    '''
    imsize = x_test.shape  
    
    tarIdx = np.argmax(tarFunc(x_test)[-1])
    tarClass = classnames[tarIdx]
    #tarIdx = 287
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(x_test_im, interpolation='nearest')
    plt.title('original')
    frame = pylab.gca()
    frame.axes.get_xaxis().set_ticks([])
    frame.axes.get_yaxis().set_ticks([]) 
    plt.subplot(2,2,2)
    plt.imshow(sensMap, cmap=cm.Greys_r, interpolation='nearest')
    plt.title('sensitivity map')
    frame = pylab.gca()
    frame.axes.get_xaxis().set_ticks([])
    frame.axes.get_yaxis().set_ticks([]) 
    plt.subplot(2,2,3)
    p = predDiff.reshape((imsize[1],imsize[2],-1))[:,:,tarIdx]
    plt.imshow(p, cmap=cm.seismic, vmin=-np.max(np.abs(p)), vmax=np.max(np.abs(p)), interpolation='nearest')
    plt.colorbar()
    #plt.imshow(np.abs(p), cmap=cm.Greys_r)
    plt.title('weight of evidence')
    frame = pylab.gca()
    frame.axes.get_xaxis().set_ticks([])
    frame.axes.get_yaxis().set_ticks([]) 
    plt.subplot(2,2,4)
    plt.title('class: {}'.format(tarClass))
    p = get_overlayed_image(x_test_im, p)
    #p = predDiff[0,:,:,np.argmax(netPred(net, x_test)[0]),1].reshape((224,224))
    plt.imshow(p, cmap=cm.seismic, vmin=-np.max(np.abs(p)), vmax=np.max(np.abs(p)), interpolation='nearest')
    #plt.title('class entropy')
    frame = pylab.gca()
    frame.axes.get_xaxis().set_ticks([])
    frame.axes.get_yaxis().set_ticks([]) 
    
    fig = plt.gcf()
    fig.set_size_inches(np.array([12,12]), forward=True)
    plt.tight_layout()
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def get_overlayed_image(x, c, gray_factor_bg = 0.3):    
    '''
    For an image x and a relevance vector c, overlay the image with the 
    relevance vector to visualise the influence of the image pixels.
    '''
    imDim = x.shape[0]
    
    if np.ndim(c)==1:
        c = c.reshape((imDim,imDim))
    if np.ndim(x)==2: # this happens with the MNIST Data
        x = 1-np.dstack((x, x, x))*gray_factor_bg # make it a bit grayish
    if np.ndim(x)==3: # this is what happens with cifar data        
        x = color.rgb2gray(x)
        x = 1-(1-x)*0.5
        x = np.dstack((x,x,x))
        
    alpha = 0.8
    
    # Construct a colour image to superimpose
    im = plt.imshow(c, cmap = cm.seismic, vmin=-np.max(np.abs(c)), vmax=np.max(np.abs(c)), interpolation='nearest')
    color_mask = im.to_rgba(c)[:,:,[0,1,2]]
    
    # Convert the input image and color mask to Hue Saturation Value (HSV) colorspace
    img_hsv = color.rgb2hsv(x)
    color_mask_hsv = color.rgb2hsv(color_mask)
    
    # Replace the hue and saturation of the original image
    # with that of the color mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha
    
    img_masked = color.hsv2rgb(img_hsv)
    
    return img_masked
    
    
    
    