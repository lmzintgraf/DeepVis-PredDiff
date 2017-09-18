"""


@author: Luisa Zintgraf
"""

import numpy as np
import time


class PredDiffAnalyser:
    """
    This class implements the prediction difference analysis, i.e., a method
    which estimates how important the individual input features are to a 
    (already trained) classifier, given a specific input to the classifier.    
    To this end, a relevance is estimated which is of the same size as the 
    input and reflects the importance of each feature.
    
    Note: this version implements the method for RGB-image classification!
    However, the method can be used with any kind of data.
    Also note that we assume that the color channels are along axis 0, as it 
    is common with convolutional neural networks.
    """
        
    def __init__(self, input_vect, tar_func, sampler, num_samples=10, batch_size=10, prob_tar=True):
        """
        Input:  
            input_vect  the feature vector for which we want to make the analysis (can be a hidden layer!)
                        Has to be numpy array of the dimension that fits to targetFunc
            tar_func    the target function, can be the output of classifier or intermediate layer
                        (must take x as input, keep this in mind when starting at intermediate layers!)
            num_samples the number of samples used for marginalising out features
            batch_size  batch size for caffe network (in tar_func)
            prob_tar    boolean, indicates if the target values are probabilities 
                        (not necessarily the case when we look at hidden nodes)
        """
        
        # inputs
        self.x = np.copy(input_vect)
        self.tar_func = tar_func
        self.sampler = sampler
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.prob_tar = prob_tar
        
        # some other useful values
        self.num_feats = int(len(self.x.ravel())/3)  # we make the analysis not per color channel,
        # but for all channels at once, therefore we divide the number of features by 3
        self.true_tar_val = self.tar_func(self.x)  # true network state for the given input
        self.num_blobs = len(self.true_tar_val)
        self.num_metrics = 2                    # the number of metrics we use for evaluating
        # the prediction difference (avg and max of the weight of evidence per feature map)
        self.tests_per_batch = int(self.batch_size/self.num_samples)  # rounds down
        
        # drop the first dimension of the elements in the true target value list,
        # since it is not necessary (since we only forwarded a single feature vector)
        self.true_tar_val = [t[0] for t in self.true_tar_val]

    def get_rel_vect(self, win_size, overlap=True):
        """
        Main method to use, will return a relevance vector.
        Input:  win_size    the window size (k in alg. 1)
                overlap     whether the windows should be overlapping, default is True
        Output: rel_vects   the relevance vectors, dimensions are:
                            - number of features (input)
                            - number of outputs (usually output layer, can be hidden layer)
                            to interpret the result, look at one output (e.g., the predicted class)
                            and visualise the input features in some way
        """
        
        # create array for relevance vectors, each element has dimensions (num_feats)*blobdimension
        # where the relevance of each feature on the different activations in that blob is stored
        rel_vects = [np.zeros((self.num_feats, self.true_tar_val[b].shape[0]), dtype=np.float64) for
                     b in range(self.num_blobs)]

        # a counts vector to keep track of how often a feature is marginalised out
        counts = np.zeros(self.num_feats, dtype=np.int)

        # a matrix where each entry reflects the index in the flattened input (image)
        all_feats = np.reshape([i for i in range(self.num_feats*3)], self.x.shape)
        
        if overlap:
            
            windows = np.zeros((self.tests_per_batch, win_size*win_size*3), dtype=int)
            win_idx = 0
            for i in range(self.x.shape[1]-win_size+1):  # rows
                start_time = time.time()
                for j in range(self.x.shape[2]-win_size+1):  # columns
                    # get the window which we want to simulate as unknown
                    window = all_feats[:, i:i+win_size, j:j+win_size].ravel()
                    windows[win_idx] = window
                    win_idx += 1
                    if win_idx == self.tests_per_batch:
                        # evaluate the prediction difference
                        pred_diffs = self._get_rel_vect_subset(windows)
                        for w in range(self.tests_per_batch):
                            window = windows[w]
                            for b in range(self.num_blobs):
                                rel_vects[b][window[window < self.num_feats]] += pred_diffs[b][w]
                            counts[window[window < self.num_feats]] += 1
                        win_idx = 0
                print("row {}/{} took: --- {:.4f} seconds --- ".format(i, self.x.shape[1]-win_size+1,
                                                                       (time.time() - start_time)))

            # evaluate the rest that didn't fill last batch
            pred_diffs = self._get_rel_vect_subset(windows[:win_idx+1])
            for w in range(win_idx+1):
                window = windows[w]
                for b in range(self.num_blobs):
                    rel_vects[b][window[window < self.num_feats]] += pred_diffs[b][w]
                counts[window[window < self.num_feats]] += 1
                
        else: 
            
            windows = np.zeros((self.tests_per_batch, win_size*win_size*3), dtype=int)
            win_idx = 0
            for i in range(self.x.shape[1]/win_size):  # rows
                start_time = time.time()
                for j in range(self.x.shape[2]/win_size):  # columns
                    # get the window which we want to simulate as unknown
                    window = all_feats[:, i*win_size:i*win_size+win_size, j*win_size:j*win_size+win_size].ravel()
                    windows[win_idx] = window
                    win_idx += 1
                    if win_idx == self.tests_per_batch:
                        # evaluate the prediction difference
                        pred_diffs = self._get_rel_vect_subset(windows)
                        for w in range(self.tests_per_batch):
                            window = windows[w]
                            for b in range(self.num_blobs):
                                rel_vects[b][window[window < self.num_feats]] += pred_diffs[b][w]
                            counts[window[window < self.num_feats]] += 1
                        win_idx = 0
                print("row {}/{} took: --- {:.4f} seconds --- ".format(i, self.x.shape[1]/win_size-1, (time.time() - start_time)))
#                sys.stdout.write("\033[F")
                
            # evaluate the rest that didn't fill last batch
            pred_diffs = self._get_rel_vect_subset(windows[:win_idx+1])
            for w in range(win_idx+1):
                window = windows[w]
                for b in range(self.num_blobs):
                    rel_vects[b][window[window < self.num_feats]] += pred_diffs[b][w]
                counts[window[window < self.num_feats]] += 1
                            
        # get average relevance of each feature
        for b in range(self.num_blobs):
            rel_vects[b][counts != 0] = (rel_vects[b][counts != 0]/counts[counts != 0][:, np.newaxis]).astype(np.float16)

        return rel_vects
        
    def _get_rel_vect_subset(self, feature_sets):
        """
        Returns the relevance vector, given the features that are unknown.
        Input:  notGivenFeats   indices of the raveled (!) feature vector
                                that are unknown
        """  
        for feature_set in feature_sets:
            # make variable notGivenFeats ready to use it as an index
            if type(feature_set) == set or type(feature_set) == list:
                feature_set = np.array(list(feature_set)) 
            if type(feature_set) == int:
                feature_set = np.array([feature_set]) 
            # if there are no features given, return a contribution of 0
            if (type(feature_set) == list and feature_set == []) or (type(feature_set) != int and len(feature_set) == 0):
                return 0
        
        # for each data point in X, replace the (not-given-)feature with the value for it seen in X
        x_new = np.zeros((self.tests_per_batch, self.num_samples, len(self.x.ravel())))
        x_new[:] = np.copy(self.x).ravel()[np.newaxis]

#        for f in xrange(feature_sets.shape[0]):
        for f in range(len(feature_sets)):
            x_new[f, :, feature_sets[f].ravel()] = self.sampler.get_samples(feature_sets[f], self.x, self.num_samples).T
            
        # get prediction for the altered x-values
        target_vals = self.tar_func(x_new.reshape((self.tests_per_batch*self.num_samples, -1)))
        
        for b in range(self.num_blobs):
            target_vals[b] = target_vals[b].reshape((self.tests_per_batch, self.num_samples, -1))
        
        # evaluate the prediction difference
        rel_vect = self._evaluate_prediction_difference(target_vals)

        return rel_vect

    def _evaluate_prediction_difference(self, target_values):
        """
        Evaluating the prediction difference using the weight of evidence 
        Input:  tarVals     the values of all the blobs for several altered inputs
                            length is self.num_blobs, dimensions of each blob 
                            are (num_featsures)*(shape of blob)
        """
        # average over all predictions received by using altered input values
        # we will save a value per feature map (instead of for each activation)
        # therefore we loog at the avg over all activations, and the max
        prediction_diffs = []
        # For the laplace correction, we need the number of training instances
        imagenet_trainsize = 100000  # TODO: this shouldn't be hard-coded
        for b in range(self.num_blobs):
            pred_diffs = np.zeros((self.tests_per_batch, target_values[b].shape[-1]))
            for t in range(self.tests_per_batch):
                avg_prob = np.average(target_values[b][t], axis=0)
                # if we deal with probabilities, i.e., the last blobs, use this:
                if b == (self.num_blobs-1):
                    # do a laplace correction to avoid problems with zero probabilities
                    target_val_laplace = (self.true_tar_val[b]*imagenet_trainsize+1)/(imagenet_trainsize+len(self.true_tar_val[b]))
                    avg_prob_laplace = (avg_prob*imagenet_trainsize+1)/(imagenet_trainsize+len(self.true_tar_val[b]))
                    # calculate the odds for the true targets and  the targets with some features marginalised out
                    odds_target_val = np.log2(target_val_laplace/(1-target_val_laplace))
                    odds_avg_prob = np.log2(avg_prob_laplace/(1-avg_prob_laplace))
                    # take average over feature maps
                    pd = odds_target_val-odds_avg_prob
                # if we do not deal with probabilities, we just return the distance to the average
                else:
                    pd = self.true_tar_val[b] - avg_prob
                # avg/max for the feature maps if we have feature maps in conv layers
                pd = pd.reshape((pd.shape[0], -1))
                pred_diffs[t] = np.average(pd, axis=1)  # will only have an effect for convolutional layers
                # return the weight of evidence
            prediction_diffs.append(pred_diffs)
        return prediction_diffs
