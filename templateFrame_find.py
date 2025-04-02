#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 11:21:08 2023

@author: jane
"""


import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit
from cv2 import warpAffine,getRotationMatrix2D

import mainpaths

from kettlewell.propagation.multi_animal_loading_funcs import load_dfs_no_length_cutoff
from kettlewell.propagation import sequence_clustering as sc
from kettlewell.basic_tools.event_tools import get_timepoints
from kettlewell.basic_tools.event_tools import spatial_filter as run_spatial_filter

from HM import fit2Dgaussian


# TODO! 
#   turn this into a command line arg script with __main__
#   get rid of as many dependencies as possible, especially the HM and kettlewells


def workflow_gettemplates():
    ferret = 'F336'
    params = {'max_filter':10,
              'spatial_filter':True}
    fdir_base = os.path.join(mainpaths.LUNA_PATH, "template_frames", ferret)
    # =============================================================================

    seq_PFS_list, df, roi, seq_bin_list = load_data(ferret, max_frames=params['max_filter'])
    df = df.reset_index(drop=True)

    #get template corr mat. Verify correlation thresholds.
    seq_PFS_stack = np.hstack(seq_PFS_list).T
    seq_bin_stack = np.hstack(seq_bin_list).T
    nFrames = len(seq_PFS_stack)
    ump = 1000/171*8; params['ump'] = ump;
    im = roi.astype('double')
    im[~roi] = np.nan
    imbin = roi.copy()

    template_corr_mat = np.zeros((nFrames,nFrames))
    thresholds = np.zeros((nFrames))
    composites = np.zeros_like(seq_PFS_stack)
    event_params = []
#    event_params_mat = np.zeros((nFrames,7))
    for i in range(4000, len(seq_PFS_stack)):
        im[roi] = seq_PFS_stack[i,:]
        imbin[roi] = seq_bin_stack[i,:]
        if i%500==0:
            saveFig=None
            print(f'{i} of {nFrames}: {i/nFrames:.2f}')
        else:
            saveFig=None

        _,event_params_, composite = fit2Dgaussian.fit2Dgauss2events_LKmethod(im, imbin, roi, ump=ump, saveFig=saveFig,
                                                                              fdir = fdir_base)
        threshold, _, _ = correlation_control(im[roi], roi, gauss_fit=composite[roi], plot_verify=False)

        thresholds[i] = threshold
        template_corr_mat[i,:] = 1 - cdist(np.expand_dims(composite[roi],0), seq_PFS_stack, metric='correlation')
        composites[i,:] = composite[roi]
        event_params.append(event_params_)

        if False:
            fig, ax = plt.subplots(1,3)
            ax[0].imshow(im)
            ax[1].imshow(imbin)
            ax[2].imshow(composite)

    #save all files in directory
    fdir = os.path.join(mainpaths.LUNA_PATH, "templateFramesClassifiersTests","template_frames", ferret)
    file = os.path.join(fdir, "template_corr_mat.npy")
    if ~os.path.exists(file):
        np.save(file, template_corr_mat)
    file = os.path.join(fdir, "thresholds.npy")
    if ~os.path.exists(file):
        np.save(file, thresholds)
    file = os.path.join(fdir, "composites.npy")
    if ~os.path.exists(file):
        np.save(file, composites)
    file = os.path.join(fdir, "event_params.pkl")
    if ~os.path.exists(file):
        with open(file, 'wb') as fp:
            pickle.dump(event_params, fp)
    file = os.path.join(fdir, "params.npy")
    if ~os.path.exists(file):
        with open(file, 'wb') as fp:
            pickle.dump(params, fp)


# %%Loading and saving
def load_data(ferret, max_frames=10, spatial_filter=True):
    all_df, roi_list, seq_bin_list, seq_PFS_list = load_dfs_no_length_cutoff(pix_thresh=500, ferrets=[int(ferret[1:])])
    df = all_df[all_df.ferret == ferret]
    df['start_times_s'] = get_timepoints(ferret, df.indice_start, df.tser)
    ferret, roi, n = sc.get_ferret(ferret, roi_list)
    roi = roi_list[0]
    n=0

    seq_PFS_list = seq_PFS_list[n]
    seq_bin_list = seq_bin_list[n]
    n_seqs = len(seq_PFS_list)
    seq_PFS_list = [seq_PFS_list[s] for s in range(n_seqs) if (seq_PFS_list[s].shape[1]<=max_frames)]
    seq_bin_list = [seq_bin_list[s] for s in range(n_seqs) if (seq_bin_list[s].shape[1]<=max_frames)]
    df = df[df.nFrames<=max_frames]
    if spatial_filter:
        print('running spatial filter')
        seq_PFS_list = [run_spatial_filter(s, roi)[:, roi].T for s in seq_PFS_list]


    return seq_PFS_list, df, roi, seq_bin_list

class TemplateFrames():
    def __init__(self, ferret, event_params=False):
        self.ferret=ferret
        self._eventParamsFlag = event_params
        self._load_files()

    def _load_files(self):
        fdir = os.path.join(mainpaths.LUNA_PATH, "templateFramesClassifiersTests", "template_frames", self.ferret)
        file = os.path.join(fdir, "template_corr_mat.npy")

        if os.path.exists(file):
            self.templateCorrMat = np.load(file)

        file = os.path.join(fdir, "composites.npy")
        if os.path.exists(file):
            self.estimatedImages = np.load(file)

        file = os.path.join(fdir, "thresholds.npy")
        if os.path.exists(file):
            self.thresholds = np.load(file)

        if self._eventParamsFlag:
            file = os.path.join(fdir, "event_params.pkl")
            if os.path.exists(file):
                with open(file, 'rb') as fp:
                    self.eventsParams = pickle.load(fp)

        file = os.path.join(fdir, "params.npy")
        if os.path.exists(file):
            with open(file, 'rb') as fp:
                self.params = pickle.load(fp)



def load_template_frames_data(ferret):

    data_dict = {}
    fdir = os.path.join(mainpaths.LUNA_PATH, "template_frames", ferret)
    file = os.path.join(fdir, "template_corr_mat.npy")
    if os.path.exists(file):
        data_dict['template_corr_mat'] = np.load(file)
    file = os.path.join(fdir, "composites.npy")
    if os.path.exists(file):
        data_dict['composites'] = np.load(file)
    file = os.path.join(fdir, "thresholds.npy")
    if os.path.exists(file):
        data_dict['thresholds'] = np.load(file)
    file = os.path.join(fdir, "event_params.pkl")
    if os.path.exists(file):
        with open(file, 'rb') as fp:
            data_dict['event_params'] = pickle.load(fp)
    file = os.path.join(fdir, "params.npy")
    if os.path.exists(file):
        with open(file, 'rb') as fp:
            data_dict['params'] = pickle.load(fp)

    return data_dict





# %%Computations
def fit_2d_gaussian(template_frame, roi, equation='circular'):
    '''
    template_frame: 1D np array
        pix
    equation: str, 'elliptical' or 'circular'
        the difference being if x/y can have different spread
        both fit the data to the grid, i.e., no rotation

    returns
    est_params:
        [x_0, y_0, sigma_x, sigma_y]

    for now, assuming A_0 is zero since data is spatial filtered (mean=0)
    '''
    X = np.stack(np.where(roi))[::-1]
    Z = template_frame
    #center of roi is the initial guess. sigmas are 5
    param_guess = [Z.max(), *np.mean(X, axis=1, dtype=int), 5]

    if equation=='elliptical':
        def func(X, A_1, x0, y0, sigma_x, sigma_y):
            A_0 = 0
            x_term = (((X[0,:]-x0)**2)/2/sigma_x**2)
            y_term = (((X[1,:]-y0)**2)/2/sigma_y**2)
            return A_0 + A_1*np.exp(-(x_term+y_term))
        param_guess.extend([5])

    elif equation=='circular':
        def func(X, A_1, x0, y0, sigma):
            A_0 = 0
            return A_0 + A_1*np.exp(-((((X[0,:]-x0)**2))+(((X[1,:]-y0)**2)))/(2*sigma**2))

    bounds = ([0, 0, 0, 1],
              [np.inf, roi.shape[0], roi.shape[1], 6])
    est_params, _ = curve_fit(func, X, Z, param_guess, bounds=bounds)
    gauss_fit = func(X, *est_params)
    #round so that it's in pixel-space and indexable
    est_params = [int(np.round(est_params[i+1])) for i in range(len(est_params[1:]))]


    im = roi.astype(float)
    im[roi] = gauss_fit


    return est_params, gauss_fit


# %% Controls
def correlation_control(frame, roi, gauss_fit=None, nSur=None, mode='intersection', plot_verify=True):
    """
    get the correlation threshold for a particular template.
    There are a lot of options here that I havent explicitly coded as available.

    with default parameters, this function will take a template frame, fit a 2d gaussian (to discount noise),
    find the 1 std ellipse around the gaussian, and do a sweep for every xy pair within the ellipse.
    Correlations are then taken against the original gaussian and the lowest is taken as our threshold.

    Parameters
    ----------
    frame : 1d array of pixel values
    roi : 2d bool
    nSur : int. if you are doing random parameters instead of a sweep
        DESCRIPTION. The default is None
    mode : str, "intersection" or "noise"
        intersection: takes the corrs only of globally shared pixels
        noise: fills in nans with gaussian distributed noise

    Returns
    -------
    threshold : TYPE
        DESCRIPTION.

    """

    if gauss_fit is None:
        est_params, gauss_fit = fit_2d_gaussian(frame, roi)
    else:
        est_params = [0,0,4]
    if plot_verify:
        fig, ax = plt.subplots(1,3)
        for ax_ in ax:
            ax_.spines[:].set_visible(False)
            ax_.set_xticks([])
            ax_.set_yticks([])
        vmin = np.percentile(frame,0.01)
        vmax = np.percentile(frame,99.99)
        im = roi.astype('single')
        im[roi] = frame
        im[~roi] = np.nan
        ax[0].imshow(im, vmin=vmin, vmax=vmax)
        im[roi] = gauss_fit
        ax[1].imshow(im, vmin=vmin, vmax=vmax)
        im[roi] = frame-gauss_fit
        ax[2].imshow(im, vmin=vmin, vmax=vmax)

        ax[0].set_title('template')
        ax[1].set_title('gaussian fit')
        ax[2].set_title('difference')

    #get data into proper format
    im = roi.astype('single')
    im[~roi] = np.nan
    #i've only written this to work with one frame at a time
    nFrames = 1
    im[roi] = gauss_fit
    data = np.expand_dims(im, 0)
    # elif len(frame.shape)==2:
    #     nFrames = frame.shape[1]
    #     data = np.zeros((nFrames, *roi.shape))
    #     for i in range(nFrames):
    #         im[roi] = frame[:,i]
    #         data[i,:,:] = im
    rng = np.random.RandomState(20210111)

    sigma_x, sigma_y = est_params[2], est_params[2]
    x_0, y_0 = (np.array(roi.shape)/2).astype(int) #est_params[0:2]
    xx, yy = np.where(roi)
    xy_coords = np.where( (((xx-x_0)**2/sigma_x**2) + ((yy-y_0)**2/sigma_y**2)) <= 1)[0]
    xxx =xx[xy_coords] - x_0
    yyy = yy[xy_coords] - y_0
    xy_coords = [(xxx[s],yyy[s]) for s in range(len(xxx))]

    kwargs = {
        #'center_coords':est_params[0:2],
        'xy_space': xy_coords,
        'do_sweep': True
        }

    nSur = len(xy_coords)
    controls = get_shifted_activity(
        data,nSur,nFrames,None,
        roi,rng,surrogate_in_2d=False,do_shift=True,
        do_rotation=False, **kwargs)[0,:,:]

    if False:
        h, w= 68,80
        ctrx = float(kwargs["center_coords"][0])
        ctry = float(kwargs["center_coords"][1])
        new_roi = np.zeros((h*2, w*2), dtype=bool)
        x_l = int(w-ctrx)
        y_l = int(h-ctry)
        new_roi[y_l:y_l+h,x_l:x_l+w] = roi


        fig, ax = plt.subplots(7,7)
        ax = ax.flatten()
        #im=roi.astype('single')
        #im[~roi] = np.nan
        for i in range(100):
            im = controls[0,i,:,:]
            ax[i].imshow(im)
            ax[i].plot([80,80], [0,136], 'k:')
            ax[i].plot([0,160], [68,68], 'k:')
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].set_ylim(92,32)
            ax[i].set_xlim(54,122)
        im = new_roi.astype('single')
        im[new_roi] = data[0,roi]
        im[~new_roi] = np.nan
        ax[0].imshow(im)

    template_corrs = np.zeros(nSur) #, nFramesAll))
    template_frame = np.expand_dims(gauss_fit, 0)
    is_sizes = []

    if mode=='intersection':
        intersection = np.where(~np.isnan(np.mean(controls, axis=0)))[0] #"full" intersection
        for i in range(nSur):
            control_ = np.expand_dims(controls[i,:],0)
            #intersection = np.where(~np.isnan(control_))[1]
            is_sizes.append(len(intersection))
            template_corrs[i] = cdist(control_[:,intersection], template_frame[:,intersection],
                                      metric='correlation')
    elif mode=='noise':
        mu_t = template_frame.mean()
        sigma_t = template_frame.std()
        for i in range(nSur):
            control_ = np.expand_dims(controls[i,:],0)
            intersection = len(np.where(np.isnan(control_))[1])
            control_[np.isnan(control_)] = np.random.normal(mu_t, sigma_t, intersection)
            template_corrs[i] = cdist(control_, template_frame, metric='correlation')

    template_corrs = 1-template_corrs
    threshold = template_corrs.min()


    #finally, looking to output a 95% threshold for significant correlations
    return threshold, template_corrs, gauss_fit


def get_shifted_activity(data,Nsur,nframes,norm_activity,roi,rng,surrogate_in_2d=False,\
						do_shift=False):
	'''
	compute randomly shifted, rotated and mirrored realizations of data

	input:
	data : 		activity frames, shape is: # frames x height x width, will be transformed to float32
	Nsur:		# of surrogate frames per data frame
	nframes		# frames you actually want to have surrogates from (can be smaller than # of frames of data)
	norm_activity		normalise activity by this array (e.g. normalise sd to 1)
	roi 				roi
	surrogate_in_2d		if True output shape is: # of frames x Nsur x height x width else # of frames x Nsur x # pixels in roi
	rng					seed for random conversions on activity patterns
	do_shift:			if False activity patterns will not be shifted in x,y direction (only rotation
						and reflection)

	output:
	data_sh: 	control data (rotated,mirrored and possibly shifted in x,y compared to original data)
				as type float32
	'''

	nall_frames,h,w = data.shape
	npatterns = np.sum(roi)
	idy,idx = np.where(roi)
	miny,maxy = min(idy),max(idy)
	minx,maxx = min(idx),max(idx)
	ctry,ctrx = (maxy-miny)//2+miny,(maxx-minx)//2+minx
	ctry, ctrx = int(ctry), int(ctrx)
	delta_gamma = 10		# rotate in units of 10 degrees
	data_sh = np.empty((nframes,Nsur,npatterns))*np.nan

	# randomly create rotation angles
	gammas = rng.choice(np.arange(0,1440,delta_gamma),size=Nsur*nframes,replace=True)	#4*360=1440
	gammas = gammas.reshape(Nsur,nframes)

	do_replace = nall_frames<nframes
	for isur in range(Nsur):
		selected_frames = rng.choice(np.arange(nall_frames),size=nframes,replace=do_replace)#np.arange(nall_frames)#
		data_part = data[selected_frames,:,:]
		#if norm_activity is not None:
			#data_part = data_part/np.nanstd(data_part.reshape(nframes,h*w),axis=1)[:,None,None]*norm_activity[:,None,None]

		for im in range(nframes):
			### rotate and mirror original activity pattern
			frame = data_part[im,:,:]
			gamma = gammas[isur,im]

			''' rotation '''
			M = getRotationMatrix2D((ctrx,ctry),gamma%360,1)
			dst = warpAffine(np.float32(frame),M,(w,h),borderValue=np.nan)

			''' translation (can make effective ROI smaller)'''
			if do_shift:
				shift = rng.choice(np.arange(-18,19,1),size=2, replace=False)		#changed on june 22th 2017, before: (-18,19,1)
				Mshift = np.float32([[1,0,shift[0]],[0,1,shift[1]]])
				dst = warpAffine(dst,Mshift,(w,h),borderValue=np.nan)

			''' reflection '''
			if gamma>720:
				flip = dst[miny:maxy+1,:]
				dst[miny:maxy+1,:] = flip[::-1,:]
			if ((gamma//delta_gamma)%2)!=0:
				flip = dst[:,minx:maxx+1]
				dst[:,minx:maxx+1] = flip[:,::-1]

			this_dst = dst[roi]
			if norm_activity is not None:
				this_dst = this_dst/np.nanstd(this_dst)*norm_activity[selected_frames[im]]
			data_sh[im,isur,:] = this_dst

			#print('phi={}, y,x=({},{}), refl-y={}, refl-x={}'.format(gamma%360,shift[0],shift[1], gamma>720, ((gamma//delta_gamma)%2)!=0))

	if surrogate_in_2d:
		data_sd_2d = np.empty(((nframes,Nsur)+roi.shape))*np.nan
		data_sd_2d[:,:,roi] = data_sh
		return data_sd_2d
	else:
		return data_sh