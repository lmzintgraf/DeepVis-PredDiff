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
import utils_data
import utils_classifiers


class MarginalSamplerImagenet:
    """ 
    Marginal sampler for image patches
    """

    def __init__(self, data_mat, net):
        """
        Sampler to draw marginal samples
        Input:
                X       imagenet data from which we will take our samples
                        for the feature values
        """
        # we will use raveled arrays for each image
        self.X = data_mat.reshape(data_mat.shape[0], -1)

    def get_samples(self, sample_indices, feature_vector=None, num_samples=10):
        """
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
        """
        return np.take(self.X[:num_samples], sample_indices.ravel(), axis=1)


class ConditionalSamplerImagenet:
    """
    Conditional sampler for image patches
    using a multivariate Gaussian distribution
    """

    def __init__(self, win_size, padding_size, image_dims, netname, num_samples_fit=20000):
        """
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
        """

        # inputs
        self.win_size = win_size
        self.padding_size = padding_size
        self.image_dims = image_dims
        self.netname = netname
        self.num_samples_fit = num_samples_fit

        self.path_folder = './gaussians/'
        if not os.path.exists(self.path_folder):
            os.makedirs(self.path_folder)

            # the whole patch size whose pixel distribution we model with a gaussian
        self.patchSize = win_size + 2 * self.padding_size
        # the mean and covariance for the gaussian distribution on the whole patch
        self.meanVects, self.covMats = self._get_gauss_params()

        # the min/max values for the features seen in the data, so that we can cut off overfloating values
        if not os.path.exists(self.path_folder + '{}_minMaxVals.npy'.format(netname)):
            save_minmax_values(self.netname)
        self.minMaxVals = np.load(self.path_folder + '{}_minMaxVals.npy'.format(netname))

        self.location = None
        self.dotProdForMean = None
        self.cond_cov = None

    def _get_gauss_params(self):
        """
        Returns the mean and covariance for the gaussian model on the whole
        patch (i.e., window to sample plus padding around it)
        """

        means = np.zeros((3, self.patchSize * self.patchSize))
        covs = np.zeros((3, self.patchSize * self.patchSize, self.patchSize * self.patchSize))

        path_mean = self.path_folder + '{}_means{}_indep'.format(self.netname, self.patchSize)
        path_cov = self.path_folder + '{}_covs{}_indep'.format(self.netname, self.patchSize)

        # check if  values are already precomputed and saved; otherwise do so first
        if os.path.exists(path_mean + '.npy') and os.path.exists(path_cov + '.npy'):

            means = np.load(path_mean + '.npy')
            covs = np.load(path_cov + '.npy')

        else:

            for c in [0, 1, 2]:

                net = utils_classifiers.get_caffenet(self.netname)

                # get the imagenet data
                data_mat, _, _ = utils_data.get_imagenet_data(net)

                # get samples for fitting the distribution
                patches_mat = np.empty((0, self.patchSize * self.patchSize), dtype=np.float)
                for i in range(int(self.num_samples_fit / data_mat.shape[0]) + 1):
                    # get a random (upper left) position of the patch
                    idx = random.sample(range((self.image_dims[0] - self.patchSize) * (self.image_dims[1] - self.patchSize)), 1)[0]
                    idx = np.unravel_index(idx, (self.image_dims[0] - self.patchSize, self.image_dims[1] - self.patchSize))
                    idx = [idx[0], idx[1]]
                    # get the patch from all the images in data_mat, from the given channel
                    patch = data_mat[:, c, idx[0]:idx[0] + self.patchSize, idx[1]:idx[1] + self.patchSize]
                    patches_mat = np.vstack((patches_mat, patch.reshape((data_mat.shape[0], self.patchSize * self.patchSize))))

                # compute the mean and covariance of the collected samples
                means[c] = np.mean(patches_mat, axis=0)
                covs[c] = np.cov(patches_mat.T)

            # save the mean and the covariance
            np.save(path_mean, means)
            np.save(path_cov, covs)

        return means, covs

    def _get_cond_params(self, surr_patch, in_patch_idx, channel):
        """
        Input:
                surrpatch   the variables over which we have a distribution
                inPatchIdx  the index/indices from what we want to sample
        Output:
                cond_mean    the conditional mean of the inner patch,
                            conditioned on the surrounding pixels
                cond_cov     the conditional covariance
        """

        # get the part of the surrPacth vector which we use to condition the values on
        x2 = np.delete(surr_patch, in_patch_idx)
        # split the mean vector into mu1 and mu2 (matching what we want to sample/condition on)
        mu1 = np.take(self.meanVects[channel], in_patch_idx)
        mu2 = np.delete(self.meanVects[channel], in_patch_idx)

        path_dot_prod_for_mean = self.path_folder + '{}_cov{}_win{}_dotProdForMean_{}_{}'.format(self.netname,
                                                                                                 self.patchSize,
                                                                                                 self.win_size,
                                                                                                 in_patch_idx[0],
                                                                                                 in_patch_idx[-1])

        # get the dot product for the mean (check if precomputed, otherwise do this first)
        if not os.path.exists(path_dot_prod_for_mean + '.npy'):
            cov11 = self.covMats[channel][in_patch_idx][:, in_patch_idx]
            cov12 = np.delete(self.covMats[channel][in_patch_idx, :], in_patch_idx, axis=1) if np.ndim(
                in_patch_idx > 1) else np.delete(self.covMats[channel][in_patch_idx, :], in_patch_idx)
            cov21 = np.delete(self.covMats[channel][:, in_patch_idx], in_patch_idx, axis=0)
            cov22 = np.delete(np.delete(self.covMats[channel], in_patch_idx, axis=0), in_patch_idx, axis=1)
            # compute the conditional mean and covariance
            dot_prod_for_mean = np.dot(cov12, scipy.linalg.inv(cov22))
            np.save(path_dot_prod_for_mean, dot_prod_for_mean)
        else:
            dot_prod_for_mean = np.load(path_dot_prod_for_mean + '.npy')

        # with the dotproduct, we can now evaluate the conditional mean
        cond_mean = mu1 + np.dot(dot_prod_for_mean, x2 - mu2)

        path_cond_cov = self.path_folder + '{}_cov{}_win{}_cond_cov_{}_{}_indep'.format(self.netname, self.patchSize,
                                                                                        self.win_size, in_patch_idx[0],
                                                                                        in_patch_idx[-1])

        # get the conditional covariance
        if not os.path.exists(path_cond_cov + '.npy'):
            cov11 = self.covMats[channel][in_patch_idx][:, in_patch_idx]
            cov12 = np.delete(self.covMats[channel][in_patch_idx, :], in_patch_idx, axis=1) if np.ndim(
                in_patch_idx > 1) else np.delete(self.covMat[in_patch_idx, :], in_patch_idx)
            cov21 = np.delete(self.covMats[channel][:, in_patch_idx], in_patch_idx, axis=0)
            cov22 = np.delete(np.delete(self.covMats[channel], in_patch_idx, axis=0), in_patch_idx, axis=1)
            cond_cov = cov11 - np.dot(np.dot(cov12, scipy.linalg.inv(cov22)), cov21)
            np.save(path_cond_cov, cond_cov)
        else:
            cond_cov = np.load(path_cond_cov + '.npy')

        return cond_mean, cond_cov

    def _get_surr_patch(self, x, sample_indices):
        """
        Gets a patch around the sampleIndices
        Input:
                x               the whole feature vector
                sampleIndices   the (raveled) indices for which we want to
                                get samples
        """
        height = self.image_dims[0]
        width = self.image_dims[1]
        # get the 2d values of the sample indices (since we sample from all color at once channels anyway)
        subset3d = np.unravel_index(sample_indices.ravel(), [3, self.image_dims[0], self.image_dims[1]])
        subset2d = [subset3d[1], subset3d[2]]
        # we will need this to find the index of the sample inside the surrounding patch
        inner_patch_indices = np.tile(np.array([i for i in range(self.patchSize * self.patchSize)]), 3).reshape(3,
                                                                                                       self.patchSize,
                                                                                                       self.patchSize)
        # indices of the subset relative to the whole feature map x
        upper_idx = subset2d[0][0]
        lower_idx = subset2d[0][-1]
        left_idx = subset2d[1][0]
        right_idx = subset2d[1][-1]
        # indices of the subset relative to the surrounding patch
        upper_idx_patch = upper_idx
        lower_idx_patch = self.patchSize - (height - lower_idx)
        left_idx_patch = left_idx
        right_idx_patch = self.patchSize - (width - right_idx)
        # get a surrounding patch, depending on where the subset lies
        # and find the indices inside the patch where the subset is
        self.location = None
        if left_idx < self.padding_size and upper_idx < self.padding_size:  # upper left
            surrounding_patch = x[:, :self.patchSize, :self.patchSize]
            inner_patch_indices = inner_patch_indices[:, upper_idx_patch:upper_idx_patch + self.win_size,
                                  left_idx_patch:left_idx_patch + self.win_size]
        elif left_idx < self.padding_size and lower_idx > (height - 1 - self.padding_size):  # lower left
            surrounding_patch = x[:, -self.patchSize:, :self.patchSize]
            inner_patch_indices = inner_patch_indices[:, lower_idx_patch - self.win_size + 1:lower_idx_patch + 1,
                                  left_idx_patch:left_idx_patch + self.win_size]
        elif right_idx > (width - 1 - self.padding_size) and lower_idx > (height - 1 - self.padding_size):  # lower right
            surrounding_patch = x[:, -self.patchSize:, -self.patchSize:]
            inner_patch_indices = inner_patch_indices[:, lower_idx_patch - self.win_size + 1:lower_idx_patch + 1,
                                  right_idx_patch - self.win_size + 1:right_idx_patch + 1]
        elif right_idx > (width - 1 - self.padding_size) and upper_idx < self.padding_size:  # upper right
            surrounding_patch = x[:, :self.patchSize, -self.patchSize:]
            inner_patch_indices = inner_patch_indices[:, upper_idx_patch:upper_idx_patch + self.win_size,
                                  right_idx_patch - self.win_size + 1:right_idx_patch + 1]
        elif left_idx < self.padding_size:  # left side
            surrounding_patch = x[:, upper_idx - self.padding_size:lower_idx + self.padding_size + 1, :self.patchSize]
            inner_patch_indices = inner_patch_indices[:, self.padding_size:-self.padding_size, left_idx_patch:left_idx_patch + self.win_size]
        elif right_idx > (width - 1 - self.padding_size):  # right side
            surrounding_patch = x[:, upper_idx - self.padding_size:lower_idx + self.padding_size + 1, -self.patchSize:]
            inner_patch_indices = inner_patch_indices[:, self.padding_size:-self.padding_size,
                                  right_idx_patch - self.win_size + 1:right_idx_patch + 1]
        elif upper_idx < self.padding_size:  # upper side
            surrounding_patch = x[:, :self.patchSize, left_idx - self.padding_size:right_idx + self.padding_size + 1]
            inner_patch_indices = inner_patch_indices[:, upper_idx_patch:upper_idx_patch + self.win_size, self.padding_size:-self.padding_size]
        elif lower_idx > (height - 1 - self.padding_size):  # lower side
            surrounding_patch = x[:, -self.patchSize:, left_idx - self.padding_size:right_idx + self.padding_size + 1]
            inner_patch_indices = inner_patch_indices[:, lower_idx_patch - self.win_size + 1:lower_idx_patch + 1, self.padding_size:-self.padding_size]
        else:  # somewhere in the middle
            self.location = 'middle'
            surrounding_patch = x[:, upper_idx - self.padding_size:lower_idx + self.padding_size + 1,
                                left_idx - self.padding_size:right_idx + self.padding_size + 1]
            inner_patch_indices = inner_patch_indices[:, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size]
            # return the patch and the indices of the subset relative to that
        return surrounding_patch, inner_patch_indices

    def get_samples(self, sample_indices, feature_vect, num_samples=100):
        """
        Input   feature_vect    the complete feature vector
                sampleIndices   the raveled(!) indices which we want to sample
                numSamples      how many samples to draw

        """

        feature_vect = np.copy(feature_vect)

        # to avoid mistakes, remove the feature values of the part that we want to sample
        feature_vect.ravel()[sample_indices.ravel()] = 0

        # reshape inputs if necessary
        if np.ndim(sample_indices) == 1:
            sample_indices = sample_indices.reshape(3, self.win_size, self.win_size)
        if np.ndim(feature_vect) == 1:
            feature_vect = feature_vect.reshape([3, self.image_dims[0], self.image_dims[1]])

        # get a patch surrounding the sample indices and the indices relative to that
        patch, patch_indices = self._get_surr_patch(feature_vect, sample_indices)

        # For each color channel, we will conditionally sample pixel
        # values from a multivariate distribution

        samples = np.zeros((num_samples, 3, self.win_size * self.win_size))

        for c in [0, 1, 2]:

            patch_c = patch[c].ravel()
            patch_indices_c = patch_indices[c].ravel()

            # get the conditional mean and covariance
            if self.padding_size == 0:
                cond_mean = self.meanVects[c]
                cond_cov = self.covMat[c]
            else:
                cond_mean, cond_cov = self._get_cond_params(patch_c, patch_indices_c, c)

                # sample from the conditional distribution
                #        samples = np.random.multivariate_normal(cond_mean, cond_cov, numSamples)
            # -- FASTER:
            dim_gauss = self.win_size * self.win_size
            # --- (1) find real matrix A such that AA^T=Sigma ---
            A = np.linalg.cholesky(cond_cov)
            # --- (2) get (numSamples) samples from a standard normal ---
            z = np.random.normal(size=num_samples * dim_gauss).reshape(dim_gauss, num_samples)
            # --- (3) x=mu+Az ---
            samples[:, c] = cond_mean[np.newaxis, :] + np.dot(A, z).T

        samples = samples.reshape((num_samples, -1))

        # get the min/max values for this particular sample
        # (since the data is preprocessed these can be different for each pixel!)\
        min_vals_sample = self.minMaxVals[0].ravel()[sample_indices.ravel()]
        max_vals_sample = self.minMaxVals[1].ravel()[sample_indices.ravel()]
        # clip the values
        for i in range(samples.shape[0]):
            samples[i][samples[i] < min_vals_sample] = min_vals_sample[samples[i] < min_vals_sample]
            samples[i][samples[i] > max_vals_sample] = max_vals_sample[samples[i] > max_vals_sample]

        return samples


def save_minmax_values(netname):
    """
    When X.npy is updated, this can be executed to also update the min/max
    values of the data (which is being used to cut off the values in the
    sampler so that we don't have overflowing values)
    """
    net = utils_classifiers.get_caffenet(netname)
    X, _, _ = utils_data.get_imagenet_data(net)
    min_max_vals = np.zeros((2, 3, X.shape[-1], X.shape[-1]))
    min_max_vals[0] = np.min(X, axis=0)
    min_max_vals[1] = np.max(X, axis=0)
    path_folder = './gaussians/'
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)
    np.save(path_folder + '{}_minMaxVals'.format(netname), min_max_vals)
