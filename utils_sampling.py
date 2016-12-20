# -*- coding: utf-8 -*-
"""
Classes for (1) marginal and (2) conditional sampling for the IMAGENET data.

Note that every sampler has to implement the method 
    get_samples(sampleIndices, featVect, numSamples)

"""

import numpy as np
import random
import scipy
import os.path
# utilities
import utils_data as utlD
import utils_classifiers as utlC


class marg_sampler_imagenet:
    ''' 
    Marginal sampler for image patches
    '''
    
    def __init__(self, X, net):
        ''' 
        Sampler to draw marginal samples 
        Input:  
                X       imagenet data from which we will take our samples
                        for the feature values
        '''
        # we will use raveled arrays for each image
        self.X = X.reshape(X.shape[0],-1)
        
    def get_samples(self, sampleIndices, featVect=None, numSamples=10):
        '''
        Draw marginal samples for the given indices
        Input:
                sampleIndices   the indices in the image for which we want to
                                draw samples; is expected to reflect the
                                indices of the raveled (!) image
                featVect        (irrelevant because we don't condition on it)
                numSamples      the number of samples to return
        Output: 
                samples         the marginal samples, in a matrix of size
                                (numSamples)*(number of sampleIndices)
        '''   
        return np.take(self.X[:numSamples], sampleIndices.ravel(), axis=1)


class cond_sampler_imagenet:
    '''
    Conditional sampler for image patches
    using a multivariate Gaussian distribution
    '''
    
    def __init__(self, win_size, padding_size, image_dims, netname, num_samples_fit=20000):
        '''
        Sampler to conditionally sample pixel patches using a gaussian model
        Input: 
                win_size        the size (=width=height) of the window which
                                we want to sample values for
                padding_size    the padding size that is added on each side of
                                the window to condition the sampled values on
                image_dim       the (2d) dimensions of the image, i.e.,
                                (width, height)
                netname         the name of the convnet which is being used
                                (needed because each net preprocesses the data
                                differently)
                num_samples_fit number of samples used to fit the gaussian
        '''
        
        # inputs
        self.win_size = win_size
        self.padding_size = padding_size
        self.image_dims = image_dims
        self.netname = netname
        self.num_samples_fit = num_samples_fit
        
        # the whole patch size whose pixel distribution we model with a gaussian
        self.patchSize = win_size+2*self.padding_size
        # the mean and covariance for the gaussian distribution on the whole patch
        self.meanVects, self.covMats = self._get_gauss_params()
        
        # the min/max values for the features seen in the data, so that we can cut off overfloating values
        if not os.path.exists('./gaussians/{}_minMaxVals.npy'.format(netname)):
            save_minmax_values(self.netname)
        self.minMaxVals = np.load('./gaussians/{}_minMaxVals.npy'.format(netname))
        
        self.location = None
        self.dotProdForMean = None
        self.cond_cov = None

    def _get_gauss_params(self):
        '''
        Returns the mean and covariance for the gaussian model on the whole
        patch (i.e., window to sample plus padding around it)
        '''
        
        means = np.zeros((3, self.patchSize*self.patchSize))
        covs = np.zeros((3, self.patchSize*self.patchSize, self.patchSize*self.patchSize)) 

        path_mean = './gaussians/{}_means{}_indep'.format(self.netname, self.patchSize)
        path_cov = './gaussians/{}_covs{}_indep'.format(self.netname, self.patchSize)
        
        # check if  values are already precomputed and saved; otherwise do so first
        if os.path.exists(path_mean+'.npy') and os.path.exists(path_cov+'.npy'):
            
            means = np.load(path_mean+'.npy')
            covs = np.load(path_cov+'.npy')       
            
        else:
                 
            for c in [0,1,2]:
 
                net = utlC.get_caffenet(self.netname)

                # get the imagenet data
                X, _, _ = utlD.get_imagenet_data(net)
                
                # get samples for fitting the distribution
                patchesMat = np.empty((0,self.patchSize*self.patchSize), dtype=np.float)
                for i in xrange(int(self.num_samples_fit/X.shape[0])+1):
                    # get a random (upper left) position of the patch
                    idx = random.sample(range((self.image_dims[0]-self.patchSize)*(self.image_dims[1]-self.patchSize)), 1)[0]
                    idx = np.unravel_index(idx, (self.image_dims[0]-self.patchSize, self.image_dims[1]-self.patchSize))   
                    idx = [idx[0],idx[1]]
                    # get the patch from all the images in X, from the given channel
                    patch = X[:,c,idx[0]:idx[0]+self.patchSize,idx[1]:idx[1]+self.patchSize]
                    patchesMat = np.vstack((patchesMat, patch.reshape((X.shape[0],self.patchSize*self.patchSize))))
                    
                # compute the mean and covariance of the collected samples
                means[c] = np.mean(patchesMat, axis=0)
                covs[c] = np.cov(patchesMat.T)
                
            # save the mean and the covariance
            np.save(path_mean, means)
            np.save(path_cov, covs)
            
        return means, covs
             
    
    def _get_cond_params(self, surrPatch, inPatchIdx, channel):
        '''
        Input:
                surrpatch   the variables over which we have a distribution
                inPatchIdx  the index/indices from what we want to sample
        Output: 
                cond_mean    the conditional mean of the inner patch,
                            conditioned on the surrounding pixels
                cond_cov     the conditional covariance
        '''    
        
        # get the part of the surrPacth vector which we use to condition the values on
        x2 = np.delete(surrPatch, inPatchIdx)
        # split the mean vector into mu1 and mu2 (matching what we want to sample/condition on)
        mu1 = np.take(self.meanVects[channel], inPatchIdx)
        mu2 = np.delete(self.meanVects[channel], inPatchIdx)       
        
        path_dotProdForMean = './gaussians/{}_cov{}_win{}_dotProdForMean_{}_{}'.format(self.netname, self.patchSize, self.win_size, inPatchIdx[0], inPatchIdx[-1])     
        
        # get the dot product for the mean (check if precomputed, otherwise do this first)
        if not os.path.exists(path_dotProdForMean+'.npy'):
            cov11 = self.covMats[channel][inPatchIdx][:,inPatchIdx]
            cov12 = np.delete(self.covMats[channel][inPatchIdx,:], inPatchIdx, axis=1) if np.ndim(inPatchIdx>1) else np.delete(self.covMats[channel][inPatchIdx,:], inPatchIdx) 
            cov21 = np.delete(self.covMats[channel][:,inPatchIdx], inPatchIdx, axis=0)
            cov22 = np.delete(np.delete(self.covMats[channel], inPatchIdx, axis=0), inPatchIdx, axis=1)
            # compute the conditional mean and covariance
            dotProdForMean = np.dot(cov12,scipy.linalg.inv(cov22))
            np.save(path_dotProdForMean, dotProdForMean)
        else:
            dotProdForMean = np.load(path_dotProdForMean+'.npy')
            
        # with the dotproduct, we can now evaluate the conditional mean
        cond_mean = mu1 + np.dot(dotProdForMean, x2-mu2)
        
        path_condCov = './gaussians/{}_cov{}_win{}_cond_cov_{}_{}_indep'.format(self.netname, self.patchSize, self.win_size, inPatchIdx[0], inPatchIdx[-1])
        
        # get the conditional covariance
        if not os.path.exists(path_condCov+'.npy'):        
            cov11 = self.covMats[channel][inPatchIdx][:,inPatchIdx]
            cov12 = np.delete(self.covMats[channel][inPatchIdx,:], inPatchIdx, axis=1) if np.ndim(inPatchIdx>1) else np.delete(self.covMat[inPatchIdx,:], inPatchIdx) 
            cov21 = np.delete(self.covMats[channel][:,inPatchIdx], inPatchIdx, axis=0)
            cov22 = np.delete(np.delete(self.covMats[channel], inPatchIdx, axis=0), inPatchIdx, axis=1)
            cond_cov = cov11 - np.dot(np.dot(cov12,scipy.linalg.inv(cov22)),cov21)
            np.save(path_condCov, cond_cov)                
        else:
            cond_cov = np.load(path_condCov+'.npy')
            
        return cond_mean, cond_cov
        
    
    def _get_surr_patch(self, x, sampleIndices):
        '''
        Gets a patch around the sampleIndices
        Input: 
                x               the whole feature vector
                sampleIndices   the (raveled) indices for which we want to
                                get samples
        '''
        height = self.image_dims[0]
        width = self.image_dims[1]
        # get the 2d values of the sample indices (since we sample from all color at once channels anyway)
        subset3d = np.unravel_index(sampleIndices.ravel(), [3, self.image_dims[0], self.image_dims[1]])
        subset2d = [subset3d[1],subset3d[2]]
        # we will need this to find the index of the sample inside the surrounding patch
        inPatchIdx = np.tile(np.array([i for i in xrange(self.patchSize*self.patchSize)]),3).reshape(3,self.patchSize,self.patchSize)
        # indices of the subset relative to the whole feature map x
        upperIdx = subset2d[0][0]
        lowerIdx = subset2d[0][-1]
        leftIdx = subset2d[1][0]  
        rightIdx = subset2d[1][-1]
        # indices of the subset relative to the surrounding patch
        upperIdxPatch = upperIdx
        lowerIdxPatch = self.patchSize-(height-lowerIdx)
        leftIdxPatch = leftIdx
        rightIdxPatch = self.patchSize-(width-rightIdx)   
        # get a surrounding patch, depending on where the subset lies
        # and find the indices inside the patch where the subset is
        self.location = None
        if leftIdx<self.padding_size and upperIdx<self.padding_size: # upper left
            surrPatch = x[:, :self.patchSize, :self.patchSize]
            inPatchIdx = inPatchIdx[:, upperIdxPatch:upperIdxPatch+self.win_size, leftIdxPatch:leftIdxPatch+self.win_size]
        elif leftIdx<self.padding_size and lowerIdx>(height-1-self.padding_size): # lower left
            surrPatch = x[:, -self.patchSize:, :self.patchSize]
            inPatchIdx = inPatchIdx[:, lowerIdxPatch-self.win_size+1:lowerIdxPatch+1, leftIdxPatch:leftIdxPatch+self.win_size]
        elif rightIdx>(width-1-self.padding_size) and lowerIdx>(height-1-self.padding_size): # lower right
            surrPatch = x[:, -self.patchSize:, -self.patchSize:]
            inPatchIdx = inPatchIdx[:, lowerIdxPatch-self.win_size+1:lowerIdxPatch+1, rightIdxPatch-self.win_size+1:rightIdxPatch+1]
        elif rightIdx>(width-1-self.padding_size) and upperIdx<self.padding_size: # upper right
            surrPatch = x[:, :self.patchSize, -self.patchSize:]
            inPatchIdx = inPatchIdx[:, upperIdxPatch:upperIdxPatch+self.win_size, rightIdxPatch-self.win_size+1:rightIdxPatch+1]
        elif leftIdx<self.padding_size: # left side
            surrPatch = x[:, upperIdx-self.padding_size:lowerIdx+self.padding_size+1, :self.patchSize]
            inPatchIdx = inPatchIdx[:, self.padding_size:-self.padding_size, leftIdxPatch:leftIdxPatch+self.win_size]
        elif rightIdx>(width-1-self.padding_size): # right side
            surrPatch = x[:, upperIdx-self.padding_size:lowerIdx+self.padding_size+1, -self.patchSize:]
            inPatchIdx = inPatchIdx[:, self.padding_size:-self.padding_size, rightIdxPatch-self.win_size+1:rightIdxPatch+1]
        elif upperIdx<self.padding_size: # upper side
            surrPatch = x[:, :self.patchSize, leftIdx-self.padding_size:rightIdx+self.padding_size+1]
            inPatchIdx = inPatchIdx[:, upperIdxPatch:upperIdxPatch+self.win_size, self.padding_size:-self.padding_size]
        elif lowerIdx>(height-1-self.padding_size): # lower side
            surrPatch = x[:, -self.patchSize:, leftIdx-self.padding_size:rightIdx+self.padding_size+1]
            inPatchIdx = inPatchIdx[:, lowerIdxPatch-self.win_size+1:lowerIdxPatch+1, self.padding_size:-self.padding_size]
        else: # somewhere in the middle
            self.location = 'middle'
            surrPatch = x[:, upperIdx-self.padding_size:lowerIdx+self.padding_size+1, leftIdx-self.padding_size:rightIdx+self.padding_size+1]
            inPatchIdx = inPatchIdx[:, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size]     
        # return the patch and the indices of the subset relative to that
        return surrPatch, inPatchIdx
        
        
    def get_samples(self, sampleIndices, featVect_orig, numSamples=100):
        '''
        Input   featVect        the complete feature vector
                sampleIndices   the raveled(!) indices which we want to sample
                numSamples      how many samples to draw
                
        '''         
        
        featVect = np.copy(featVect_orig)
        
        # to avoid mistakes, remove the feature values of the part that we want to sample
        featVect.ravel()[sampleIndices.ravel()] = 0
        
        # reshape inputs if necessary
        if np.ndim(sampleIndices)==1:
            sampleIndices = sampleIndices.reshape(3, self.win_size, self.win_size)
        if np.ndim(featVect)==1:
            featVect = featVect.reshape([3, self.image_dims[0], self.image_dims[1]])
             
        # get a patch surrounding the sample indices and the indices relative to that
        patch, patchIndices = self._get_surr_patch(featVect, sampleIndices)
        
        # For each color channel, we will conditionally sample pixel
        # values from a multivariate distribution

        samples = np.zeros((numSamples, 3, self.win_size*self.win_size))
        
        for c in [0,1,2]:
            
            patch_c = patch[c].ravel()
            patchIndices_c = patchIndices[c].ravel()
                
            # get the conditional mean and covariance
            if self.padding_size==0:
                cond_mean = self.meanVects[c]
                cond_cov = self.covMat[c]
            else:
                cond_mean, cond_cov = self._get_cond_params(patch_c, patchIndices_c, c)

            # sample from the conditional distribution
    #        samples = np.random.multivariate_normal(cond_mean, cond_cov, numSamples)
            # -- FASTER:
            dimGauss = self.win_size*self.win_size
            # --- (1) find real matrix A such that AA^T=Sigma ---
            A = np.linalg.cholesky(cond_cov)
            # --- (2) get (numSamples) samples from a standard normal ---        
            z = np.random.normal(size=numSamples*dimGauss).reshape(dimGauss,numSamples)
            # --- (3) x=mu+Az ---
            samples[:,c] = cond_mean[np.newaxis,:] + np.dot(A,z).T

        samples = samples.reshape((numSamples, -1))            
            
        # get the min/max values for this particular sample    
        # (since the data is preprocessed these can be different for each pixel!)\
        minVals_sample = self.minMaxVals[0].ravel()[sampleIndices.ravel()] 
        maxVals_sample = self.minMaxVals[1].ravel()[sampleIndices.ravel()] 
        # clip the values
        for i in xrange(samples.shape[0]):
            samples[i][samples[i]<minVals_sample] = minVals_sample[samples[i]<minVals_sample]
            samples[i][samples[i]>maxVals_sample] = maxVals_sample[samples[i]>maxVals_sample]    
                
        return samples      
 
 
#%%     
        
        
def save_minmax_values(netname):
    '''
    When X.npy is updated, this can be executed to also update the min/max
    values of the data (which is being used to cut off the values in the
    sampler so that we don't have overflowing values)
    '''
    net = utlC.get_caffenet(netname)
    X, _, _ = utlD.get_imagenet_data(net)
    minMaxVals = np.zeros((2,3,X.shape[-1],X.shape[-1]))
    minMaxVals[0] = np.min(X,axis=0)
    minMaxVals[1] = np.max(X,axis=0)
    np.save('./gaussians/{}_minMaxVals'.format(netname), minMaxVals)
   
   
    
        