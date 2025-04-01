#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 17:05:38 2024

@author: ckettlew
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import StratifiedKFold as KFold
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm


# %%Class structures to organize and pass data
class ClassifierFramewise():
    def __init__(self, groupinfo=None):
        if groupinfo is not None:
            self.scores = np.zeros(groupinfo.nSteps)
            self.scores_folds = np.zeros((5, groupinfo.nSteps))
            self.CM = []

        else:
            self.scores_independent = []
            self.scores_folds = []
            self.CM = []

    def get_significance(self, control, alpha=95):
        control_upper_bound = np.percentile(control.scores, alpha)
        self.sig_alpha = alpha
        self.sig = control_upper_bound < self.scores
        return self.sig


class GroupInfo():
    def __init__(self,group_labels, nSteps):
        self.groupLabels = group_labels
        self.nGroups = len(np.unique(group_labels))
        self.nSteps = nSteps
        self.center = int(self.nSteps/2)


# %%Optimized controls, parallel processed
def classifierChanceLevel(func, seqCentered, nPermsControl=101, nPermsSubsample=100, groupSize=None, num_cpus=20):
    """
    Parameters
    ----------
    func : function
        must return object with a 'scores' attribute
        recommended to return a test score and/or kfold CV
    data : array
        seqs x nSteps x features
        features is likely Principal Components

    nPerms : int, optional
        number of permutations for the control

    Returns
    -------
    SVM_control : TYPE
        DESCRIPTION.
    """

    subsample_idx = [seqCentered.set_matched_groups().matchedGroupIdxs for i in range(nPermsSubsample)]
    decomp = get_decomposed_data(seqCentered, nPermsSubsample, subsample_idx, num_cpus=num_cpus)
    print('pca decomp done')

    classifier_data = controlMatchedGroupSubsampling(func, decomp, seqCentered.groupLabels,
                                                nPerms=nPermsSubsample, num_cpus=num_cpus)
    classifier_scores = classifier_data.scores
    print('data classifier done. starting control...')

    control_scores = np.zeros((nPermsControl, nPermsSubsample, seqCentered.nSteps))
    for i in range(nPermsControl):
        groupLabels = np.random.permutation(seqCentered.groupLabels)
        classifier = controlMatchedGroupSubsampling(func, decomp, groupLabels,
                                                    nPerms=nPermsSubsample, num_cpus=num_cpus)
        control_scores[i,:,:] = classifier.scores
        print(i+1)

    return classifier_scores, control_scores, subsample_idx


def get_decomposed_data(seqCentered, nPermsSubsample, idxs_, num_cpus=5):
    pool = Pool(num_cpus)
    func = partial(get_pca_decomp,
                    n_components=8,
                    frames='sep')
    data = [seqCentered.set_matched_groups(idxs_[i]) for i in range(nPermsSubsample)]
    results = list(tqdm(pool.imap_unordered(func, data)))
    pool.close()

    dims = results[0].shape
    decomp = np.concatenate([r.reshape((1, *dims)) for r in results], 0)

    return decomp


def controlMatchedGroupSubsampling(clffunc, decomp, groupLabels, nPerms, num_cpus=5):
    groupinfo = GroupInfo(groupLabels, decomp.shape[2])
    classifier = ClassifierFramewise()

    pool = Pool(num_cpus)
    parfunc = partial(clffunc,
                    groupinfo=groupinfo)
    results = list(tqdm(pool.imap_unordered(parfunc, [decomp[i,:,:,:] for i in range(len(decomp))])))
    pool.close()

    classifier.scores = np.stack([r.scores for r in results])

    return classifier


# %%Base functions
def get_svm_simple(decomp, groupinfo):
    '''
    decomp = nSeqs x nSteps x nPCs

    '''
    SVM_framewise = ClassifierFramewise(groupinfo)
    n_splits=5
    kf = KFold(n_splits=n_splits, shuffle=True)
    svc_sep = svm.SVC()

    for step in range(groupinfo.nSteps):
        data = decomp[:,step,:]
        score_ = 0
        for i, (train_index, test_index) in enumerate(kf.split(data, groupinfo.groupLabels)):
             svc_sep.fit(data[train_index,:], groupinfo.groupLabels[train_index])
             score_ += svc_sep.score(data[test_index,:], groupinfo.groupLabels[test_index])
        SVM_framewise.scores[step] = score_ / n_splits

    return SVM_framewise


def get_pca_decomp(seqCentered, n_components=8, frames='sep',
                   debug=False):
    """
    Parameters
    ----------
    seqCentered : class with
        attributes: groupLabels (array), nSteps (integer), matchedGroupIdxs (array)
            data (array, [nSequences, nSteps, nPixels])
        methods: set_matched_groups() 
    n_components : int, optional
        DESCRIPTION. Number of principal components for pca decomposition in the preprocessing.
        The default is 8.
    frames : string, optional
        Determines what data the PCA is run on.
        'center': PCA components computed from center frames only.
        'all': PCA components computed from all frames
        'sep': PCA components computed per frame, that is, there are multiple PCA decompositions
            equal to the number of frames in the data. This is the method for 
            LK's Journal of Neuroscience paper, Figure 5.
        
        DESCRIPTION. The default is 'sep'.
        
    debug : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    decomp
        array (nSeqs ,nSteps, nPCAs)
    """

    if not debug: #main function
        if frames=='center':
            pca_frames = seqCentered.data[:,seqCentered.centerFrame,:]
            pca = PCA(n_components=n_components, whiten=False, svd_solver='randomized')
            pca.fit(pca_frames)
            clf = pca.transform(seqCentered.get_framestacked())
            decomp = np.reshape(clf, (seqCentered.nSeqs, seqCentered.nSteps,n_components))

        elif frames=='all':
            pca_frames = seqCentered.get_framestacked()
            pca = PCA(n_components=n_components, whiten=False, svd_solver='randomized')
            pca.fit(pca_frames)
            clf = pca.transform(seqCentered.get_framestacked())
            decomp = np.reshape(clf, (seqCentered.nSeqs, seqCentered.nSteps,n_components))

        elif frames=='sep':
            decomp = np.zeros((seqCentered.nSeqs, seqCentered.nSteps, n_components))
            expl_var = []
            for step in range(seqCentered.nSteps):
                pca = PCA(n_components=n_components, whiten=False, svd_solver='randomized')
                clf = pca.fit_transform(seqCentered.data[:,step,:])
                expl_var.append( np.cumsum(pca.explained_variance_ratio_)[-1] )
                decomp[:,step,:] = clf

    elif debug:
        import matplotlib.pyplot as plt
        pca_full = PCA()
        pca_center = PCA()
        model_all = pca_full.fit(seqCentered.get_framestacked())
        model_center = pca_center.fit(seqCentered.centerFrame)

        all_ve_100 = []
        center_ve_100 = []
        for step in range(seqCentered.nSteps):
            current_frames = seqCentered.data[:,step,:]
            all_ve_100.append(np.cumsum(pca_full.explained_variance_(model_all, current_frames)[0:20]))
            center_ve_100.append(np.cumsum(pca_center.explained_variance_(model_center, current_frames)[0:20]))
            print(step)

        fig, ax = plt.subplots(1,seqCentered.nSteps, sharey=True, sharex=True)
        for step in range(seqCentered.nSteps):
            ax[step].plot(all_ve_100[step], label='all')
            ax[step].plot(center_ve_100[step], label='center')

        ax[-1].legend()
        ax[0].set_ylabel('var expl ratio')
        for step, ax_ in enumerate(ax):
            ax_.spines[['right', 'top']].set_visible(False)
            ax_.set_xlabel('n comp')
            ax_.set_title(f'{((step-seqCentered.centerFrame)*20)}ms')
        return all_ve_100, center_ve_100

    return decomp
