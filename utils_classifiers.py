# -*- coding: utf-8 -*-
"""

Utility methods for handling the classifiers:
    set_caffe_mode(gpu)
    get_caffenet(netname)
    forward_pass(net, x, blobnames='prob', start='data')

"""

# this is to supress some unnecessary output of caffe in the linux console
import os
os.environ['GLOG_minloglevel'] = '2'
         
import numpy as np
import caffe 


def set_caffe_mode(gpu):
    ''' Set whether caffe runs in gpu or not, input is boolean '''
    if gpu:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()    
        
     
def get_caffenet(netname):
    
    if netname=='googlenet':
     
        # caffemodel paths
        model_path = './Caffe_Models/googlenet/'
        net_fn   = model_path + 'deploy.prototxt'
        param_fn = model_path + 'bvlc_googlenet.caffemodel'
        
        # get the mean (googlenet doesn't do this per feature, but per channel, see train_val.prototxt)
        mean = np.float32([104.0, 117.0, 123.0]) 
        
        # define the neural network classifier
        net = caffe.Classifier(net_fn, param_fn, caffe.TEST, channel_swap = (2,1,0), mean = mean)

    elif netname=='alexnet':
            
        # caffemodel paths
        model_path = './Caffe_Models/bvlc_alexnet/'
        net_fn   = model_path + 'deploy.prototxt'
        param_fn = model_path + 'bvlc_alexnet.caffemodel'
        
        # get the mean
        mean = np.load('./Caffe_Models/ilsvrc_2012_mean.npy')
        # crop mean
        image_dims = (227,227) # see deploy.prototxt file
        excess_h = mean.shape[1] - image_dims[0]
        excess_w = mean.shape[2] - image_dims[1]
        mean = mean[:, excess_h:(excess_h+image_dims[0]), excess_w:(excess_w+image_dims[1])]
        
        # define the neural network classifier
        net = caffe.Classifier(net_fn, param_fn, caffe.TEST, channel_swap = (2,1,0), mean = mean)
        
    elif netname == 'vgg':
    
        # caffemodel paths
        model_path = './Caffe_Models/vgg network/'
        net_fn   = model_path + 'VGG_ILSVRC_16_layers_deploy.prototxt'
        param_fn = model_path + 'VGG_ILSVRC_16_layers.caffemodel'
        
        mean = np.float32([103.939, 116.779, 123.68])    
        
        # define the neural network classifier    
        net = caffe.Classifier(net_fn, param_fn, caffe.TEST, channel_swap = (2,1,0), mean = mean)
        
    else:
        
        print 'Provided netname unknown. Returning None.'
        net = None
    
    return net  
     


def forward_pass(net, x, blobnames='prob', start='data'):
    ''' 
    Defines a forward pass (modified for our needs) 
    Input:      net         the network (caffe model)
                x           the input, a batch of imagenet images
                blobnames   for which layers we want to return the output,
                            default is output layer ('prob')
                start       in which layer to start the forward pass
    '''    
    
    # get input into right shape
    if np.ndim(x)==3:
        x = x[np.newaxis]  
    if np.ndim(x)<4:
        input_shape = net.blobs[start].data.shape
        x = x.reshape([x.shape[0]]+list(input_shape)[1:])

    # reshape net so it fits the batchsize (implicitly given by x)
    if net.blobs['data'].data.shape[0] != x.shape[0]:
        net.blobs['data'].reshape(*(x.shape))
        
    # feed forward the batch through the next
    net.forward_all(data=x)
    
    # collect outputs of the blobs we're interested in
    returnVals = [np.copy(net.blobs[b].data[:]) for b in blobnames]
    
    return returnVals

