#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 17:08:04 2024

@author: ckettlew
"""

import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use('agg')

import mainpaths
from kettlewell.propagation.SequenceClass import Sequences, SequencesCenteredLabeled, \
    load_PFS_sequences_centered_fast
from kettlewell.propagation.GreedyGroupsClass import GreedyGroups
from scripts.templateFrame_find import TemplateFrames
from scripts.templateFrame_analysis_controls import classifierChanceLevel, get_svm_simple


matplotlib.rcParams['savefig.pad_inches'] = 0

# TODO! 
#   Try to get rid of dependencies but I might already be minimized
#   Consider getting rid of "get_groups_subsets"

def main():
    # Parse command line arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-F", "--ferret", type=int, help="ferret number", required=True)
    parser.add_argument("-s", "--savedir", type=str, default='templateFramesClassifier', help="directory to save results")
    parser.add_argument("-g", "--ngroups", type=int, default=7, help="n groups to analyze")
    parser.add_argument("-t", "--timesteps", type=int, default=5000, help="time window in ms")
    parser.add_argument("-k", "--skip", type=int, default=5, help="timesteps to skip between")
    parser.add_argument("-p", "--nPerms", type=int, default=99, help="permutations in control")
    parser.add_argument("-b", "--nSubsamples", type=int, default=100, help="subsamples in data and control")
    parser.add_argument("--nosave", type=bool, default=False, help="prevent script from saving anything. mostly for debugging")
    parser.add_argument("--reqSurround", type=bool, default=False, help="requires surround active frames")
    parser.add_argument("--groupSize", type=int, default=0,
                        help="a control for determing how large a dataset *really* needs to be")
    parser.add_argument("-C", "--cpus", type=int, default=20, help="how many cpus to use")
    args = vars(parser.parse_args())

    # # Set up parameters
    ferret = "F" + str(args["ferret"])
    savedir = args["savedir"]
    ngroups = args["ngroups"]
    save_bool = np.logical_not(args["nosave"])
    t_steps = int(args["timesteps"] / 20)
    skip = args["skip"]
    nPerms = args["nPerms"]
    nSubsamples = args["nSubsamples"]
    reqSurround = args["reqSurround"]
    groupSize = args["groupSize"]
    ncpus = args["cpus"]
    if groupSize==0: groupSize=None

    #ferret='F335';savedir='template_frames';ngroups=7;save_bool=False;

    if save_bool:
        fdir = create_dirs(ferret, savedir)

    templateStruct = TemplateFrames(ferret, event_params=True)

    sequences = Sequences(int(ferret[1:]))
    sequences.removeSeqsThatDontPass(sequences.df.active_pix>=500)
    sequences.removeSeqsThatDontPass(sequences.df.nFrames<11)

    #get "same" template frame groups. Here is the greedy method. alternatives: pca/nmf->kmeans
    assert templateStruct.estimatedImages.shape[0] == sequences.nFrames
    templateGroups = GreedyGroups(templateStruct.templateCorrMat, templateStruct.thresholds, sequences)
    templateGroups.set_min_groupsize(15)
    templateGroups.get_greedy_groups()
    templateGroups.set_subsetofGroups(get_groups_subsets(ferret, ngroups))
    if reqSurround:
        print('Frames restricted to those with surround activity -2 and +2 frames away')
        templateGroups.set_reqActWindow([2,2])
    templateGroups.set_noRepeatSeqs()

    print('loading sequences')
    print(f'timestep: {t_steps}\tskip: {skip}')
    seqCentered, seqCenteredBin = load_PFS_sequences_centered_fast(sequences,
                                                                   templateGroups, t_steps=t_steps)
    seqCentered = SequencesCenteredLabeled(seqCentered[:,0::skip,:])
    seqCenteredBin = SequencesCenteredLabeled(seqCenteredBin)
    seqCentered.set_groups(templateGroups.clust_assignments_groupFrames)
    seqCenteredBin.set_groups(templateGroups.clust_assignments_groupFrames)
    seqCentered.set_matched_groups(groupSize=groupSize)
    seqCenteredBin.set_matched_groups(seqCentered.matchedGroupIdxs)
    xx = np.arange(-t_steps,t_steps+1, skip)*0.02

    print(f'running classifier with group size {ngroups}')
    # classifier = classifierMatchedGroupSubsampling(get_svm_simple, seqCentered, nPerms=nPerms, groupSize=groupSize, ncpus=ncpus)
    #svm_framewise = np.mean(classifier.scores, axis=0)

    classifier, control_scores, subsample_idx = classifierChanceLevel(
        get_svm_simple, seqCentered,
        nPermsControl=nPerms, nPermsSubsample=nSubsamples,
        groupSize=groupSize, ncpus=ncpus)

    sorted_mean = np.mean(np.sort(control_scores, axis=1), axis=0)
    pvals = np.zeros(len(xx))
    htest = mannwhitneyu
    for i in range(len(xx)):
        pvals[i] = htest(sorted_mean[:,i], classifier[:,i], alternative='less')[1]

    file = os.path.join(fdir, "classifierScores.npy")
    np.save(file, classifier)
    file = os.path.join(fdir, "controlScores.npy")
    np.save(file, control_scores)
    file = os.path.join(fdir, "controlSubsampleScores.npy")
    np.save(file, sorted_mean)
    file = os.path.join(fdir, "timesteps.npy")
    np.save(file, xx)
    file = os.path.join(fdir, "pvals.npy")
    np.save(file, np.array(pvals))
    file = os.path.join(fdir, "subsample_idxs.npy")
    np.save(file, np.array(subsample_idx))


def load_classifier(savedir):
    classifier = {}
    fdir = os.path.join(mainpaths.LUNA_PATH, savedir)

    file = os.path.join(fdir, "classifierScores.npy")
    classifier['classifierScores'] =  np.load(file)
    file = os.path.join(fdir, "controlScores.npy")
    classifier['controlScores'] =  np.load(file)
    file = os.path.join(fdir, "controlSubsampleScores.npy")
    classifier['controlSubsampleScores'] =  np.load(file)
    file = os.path.join(fdir, "timesteps.npy")
    classifier['timesteps'] =  np.load(file)
    file = os.path.join(fdir, "pvals.npy")
    classifier['pvals'] =  np.load(file)
    file = os.path.join(fdir, "subsample_idxs.npy")
    classifier['subsample_idxs'] =  np.load(file)

    return classifier


def create_dirs(ferret, savedir):
    fdir = os.path.join(mainpaths.LUNA_PATH, "templateFramesClassifiersTests", savedir)
    print("Save directory:")
    print(fdir)
    if os.path.exists(fdir)==False:
        print("Directory does not exist. Creating save directory")
        os.mkdir(fdir)
    fdir = os.path.join(mainpaths.LUNA_PATH, "templateFramesClassifiersTests", savedir, ferret)
    print("Save directory:")
    print(fdir)
    if os.path.exists(fdir)==False:
        print("Directory does not exist. Creating save directory")
        os.mkdir(fdir)

    return fdir


def get_groups_subsets(ferret, ngroups):
    #predetermined orders for each ferret
    #this also gives some default functionality at the end for additional groups or removing groups
    if ferret=='F261':
        groups = [5,4,1,6,2,0,3]
    elif ferret == 'F335':
        groups = [6,1,0,3,5,7,2]
    elif ferret == 'F339':
        groups = [1,4,3,5,0,2,6]
    else:
        groups = [0,1,2,3,4,5,6]

    if ngroups < len(groups):
        groups = groups[0:ngroups]

    if ngroups > len(groups):
        for i in range(max(groups)+1, ngroups):
            groups.append(i)
    print("Template groups:")
    print(groups)

    return groups


if __name__=="__main__":
    main()