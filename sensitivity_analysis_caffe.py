# -*- coding: utf-8 -*-
"""

This script is used to produce sensitivity maps, see this paper:

Simonyan, K., Vedaldi, A., & Zisserman, A. (2013): "Deep inside 
convolutional networks: Visualising image classification models 
and saliency maps.", arXiv preprint arXiv:1312.6034.

Parts of this script are copied from Yosinskis deepvis toolbox, see
https://github.com/yosinski/deep-visualization-toolbox

"""

import numpy as np

def get_sens_map(net, x_test, backprop_layer, backprop_unit):
    """a
    Given a caffe network, an image and a backpropagation layer and
    unit, this returns a sensitivity map which, loosely speaking, 
    reflects the sensitivity with which input pixels react to small 
    pertubations in their value.
    """
    
    # run x_test forward through the network once
    net.blobs['data'].data[0] = np.copy(x_test)
    net.forward()

    # set the diffs of the backprop_layer to 0
    diffs = net.blobs[backprop_layer].diff * 0
    # set the target unit to its initial value
    diffs[0][backprop_unit] = net.blobs[backprop_layer].data[0,backprop_unit]
    # save diffs in the network
    net.blobs[backprop_layer].diff[...] = diffs
    
    past_start = False
    for blob_name, blob in net.blobs.items():
        if past_start:
            blob.diff[...] = 0
        if blob_name == backprop_layer:
            past_start = True
    
    # get the sensitivity map
    sMap = net.backward(start=backprop_layer)['data'][0]
    
    # take the maximum from each color channel (from the absolute values)
    sMap = np.max(np.abs(sMap),axis=0)
    
    return np.copy(sMap)
        
 
