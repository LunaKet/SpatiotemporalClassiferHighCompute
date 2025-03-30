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

from kettlewell.propagation.GreedyGroupsClass import GroupInfo

# TODO! 
#   transform all the dumb rays into multiprocessing pools
#   get rid of GroupInfo

# %%Class
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


# %%Optimized controls
def classifierChanceLevel(func, seqCentered, nPermsControl=101, nPermsSubsample=100, groupSize=None, ncpus=20):
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

    subsample_idx = [seqCentered.set_matched_groups().matchedGroupIdxs for i in range(100)]
    decomp = get_decomposed_data_ray(seqCentered, nPermsSubsample, subsample_idx)

    groupLabels = seqCentered.groupLabels
    classifier_data = controlMatchedGroupSubsampling(func, decomp, groupLabels,
                                                nPerms=nPermsSubsample, ncpus=ncpus)
    classifier_scores = classifier_data.scores

    control_scores = np.zeros((nPermsControl, nPermsSubsample, seqCentered.nSteps))
    for i in range(nPermsControl):
        groupLabels = np.random.permutation(seqCentered.groupLabels)
        classifier = controlMatchedGroupSubsampling(func, decomp, groupLabels,
                                                    nPerms=nPermsSubsample, ncpus=ncpus)
        control_scores[i,:,:] = classifier.scores
        # p = np.mean(control_scores[i,:,:]<classifier_scores[:,:]
        print(i+1)
        # print(f'percentile of classifier: {p*100:.2f}')


    return classifier_scores, control_scores, subsample_idx


def controlMatchedGroupSubsampling(func, decomp, groupLabels, nPerms, ncpus=20):
    groupinfo = GroupInfo(groupLabels, decomp.shape[2])
    classifier = ClassifierFramewise()

    ray.shutdown()
    ray.init(num_cpus=ncpus)
    control_scores = [
        ray_shuffle.remote(
        [func,
         decomp[i,:,:,:],
         groupinfo]
        )
        for i in range(nPerms)]
    classifier.scores = np.stack(ray.get(control_scores))
    ray.shutdown()

    return classifier


def classifierMatchedGroupSubsampling(func, seqCentered, nPerms=100, groupSize=None, ncpus=20):
    #randIdx turns this into a control run
    groupinfo = GroupInfo(seqCentered.groupLabels, seqCentered.nSteps)
    classifier = ClassifierFramewise()


    ray.shutdown()
    ray.init(num_cpus=ncpus)
    scores = [
        ray_shuffle.remote(
        [func,
         get_pca_decomp(seqCentered.set_matched_groups(groupSize=groupSize)),
         groupinfo]
        )
        for i in range(nPerms)]
    #get_svm_simple(decomp, groupinfo)
    classifier.scores = np.stack(ray.get(scores))
    ray.shutdown()

    return classifier


# %%Helper functions
def get_decomposed_data_ray(seqCentered, nPermsSubsample, idxs, num_cpus=5):
    ray.shutdown()
    n = int(np.ceil(nPermsSubsample/num_cpus))
    decomp_agg = []
    for n_ in range(n):
        idxs_ = [idxs[i] for i in range(n_*num_cpus, (n_+1)*num_cpus)]
        ray.init(num_cpus=num_cpus)
        decomp = [raycompose.remote(seqCentered.set_matched_groups(idxs_[i])) for i in range(num_cpus)]
        decomp_agg.append(np.stack(ray.get(decomp)))
        ray.shutdown()
        print(f'{(n_+1)*num_cpus/nPermsSubsample*100:.2f}% pca done')
    print('stacking decomps')
    decomp = np.vstack(decomp_agg)
    return decomp


@ray.remote
def ray_shuffle(args):
    '''
    args = [func, groupinfo, decomp]
    func must have a 'scores' attribute
    '''
    func = args.pop(0)
    return func(*args).scores


@ray.remote
def raycompose(seqCentered):
    return get_pca_decomp(seqCentered, n_components=8, frames='sep')


# %%helper functions
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

    if not debug: #this odd construction is so that the main function is on top
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
        center_frames = seqCentered.data[:,:,seqCentered.centerFrame]
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