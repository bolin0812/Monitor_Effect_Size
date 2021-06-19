#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Find the optimal number of repeatition that generates stable CI range. 

Version | Last Modified |    Author     | Commment
---------------------------------------------------------
1.0     | 2021-06-16    |   Bolin Li    | initial version
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def nums_repetitive_2subsample_metric(a_sample, b_sample, subsample_size, func,
                                      resample_nums_list = [10, 20], alpha=0.05,
                                      one_ref_test=True):
                                      
    """
    One pipeline to compute key statistics and all comparison scores. 

    Parameters:
        a_sample: simulation sample data a
        b_sample: simulation sample data b
        subsample_size: the size of one subsample for both data a and data b
        func: one defined function that returns one metric score based on subsampling
        resample_nums_list: one list of repeated numbers used to compute CIs
        one_ref_test: True, Flase or String, subsampling methods 
        alpha: float, one significant level used to compute percentile values
        
    Returns:
        NA 
    """  
    # two subsampling methods
    fig, axs = plt.subplots(2,2, figsize=(17, 14))
    score_mean_drift = []
    score_mean_nodrift = []
    score_percentile1_drift = []
    score_percentile1_nodrift = []
    score_percentile2_drift = []
    score_percentile2_nodrift = []
    resample_repeat_drift = []
    resample_repeat_nodrift = []
    # avoid information leakage for performance drift so we keep one set as constant
    # we can change another set for repetitive subsampling
    
    for repeat in tqdm(resample_nums_list):
        if one_ref_test == True:
            a_dist = np.random.choice(a_sample, size=subsample_size, replace=False)
            for i in range(repeat):
                a_dist_ = np.random.choice(a_sample, size=subsample_size, replace=False)
                b_dist = np.random.choice(b_sample, size=subsample_size, replace=False)
                score_drift = func(a_dist, b_dist)
                score_nodrift = func(a_dist, a_dist_)
                resample_repeat_drift.append(score_drift)
                resample_repeat_nodrift.append(score_nodrift)  
        else:# create 2 different test sets in each iteration 
            for i in range(repeat):
                a_dist = np.random.choice(a_sample, size=subsample_size, replace=False)
                a_dist_ = np.random.choice(a_sample, size=subsample_size, replace=False)
                b_dist = np.random.choice(b_sample, size=subsample_size, replace=False)
                score_drift = func(a_dist, b_dist)
                score_nodrift = func(a_dist, a_dist_)
                resample_repeat_drift.append(score_drift)
                resample_repeat_nodrift.append(score_nodrift)            
        score_mean_drift.append(np.mean(resample_repeat_drift))
        score_mean_nodrift.append(np.mean(resample_repeat_nodrift))
        score_percentile1_drift.append(np.percentile(resample_repeat_drift, 100*alpha/2))
        score_percentile1_nodrift.append(np.percentile(resample_repeat_nodrift, 100*alpha/2))
        score_percentile2_drift.append(np.percentile(resample_repeat_drift,  100*(1-(alpha/2))))
        score_percentile2_nodrift.append(np.percentile(resample_repeat_nodrift, 100*(1-(alpha/2))))
        
    axs[0,0].plot(resample_nums_list, score_mean_nodrift, linewidth=2, label=f'Mean {func.__name__}', marker='o')
    axs[0,0].plot(resample_nums_list, score_percentile1_nodrift, linewidth=2, label='CI Lower Limit', linestyle='--')
    axs[0,0].plot(resample_nums_list, score_percentile2_nodrift, linewidth=2, label='CI Upper Limit', linestyle='--')
    axs[0,0].legend()
    axs[0,0].set_ylabel(f'{func.__name__} Score')
    axs[0,0].set_xlabel(f'Number of Repetitions (one_ref_test is {one_ref_test})')
    axs[0,0].set_title(f'[No Drift] The relationship between {func.__name__} and repetition')
    axs[0,1].plot(resample_nums_list, 
                  np.subtract(score_percentile2_nodrift, score_percentile1_nodrift), 
                  color='dimgray', linewidth=2, 
                  label=f'Range of CI with {alpha} alpha and {repeat} repetitions')
    axs[0,1].axhline(y=0.03, color='orange', linestyle=':', alpha=0.9,  label='y axis = 0.03')
    axs[0,1].set_ylabel('Upper Limit - Lower Limit')
    axs[0,1].set_xlabel(f'Number of Repetitions (one_ref_test is {one_ref_test})')
    axs[0,1].set_title('[No Drift] Range of CI')
    axs[0,1].legend()

    axs[1,0].plot(resample_nums_list, score_mean_drift, linewidth=2, label=f'Mean {func.__name__}', marker='o')
    axs[1,0].plot(resample_nums_list, score_percentile1_drift, linewidth=2, label='CI Lower Limit', linestyle='--')
    axs[1,0].plot(resample_nums_list, score_percentile2_drift, linewidth=2, label='CI Upper Limit', linestyle='--')
    axs[1,0].legend()
    axs[1,0].set_ylabel(f'{func.__name__} Score')
    axs[1,0].set_xlabel(f'Number of Repetitions (one_ref_test is {one_ref_test})')
    axs[1,0].set_title(f'[Drift] The relationship between {func.__name__} and repetition')
    axs[1,1].plot(resample_nums_list, 
                  np.subtract(score_percentile2_drift, score_percentile1_drift), 
                  color='dimgray', linewidth=2, 
                  label=f'Range of CI with {alpha} alpha and {repeat} repetitions')
    axs[1,1].axhline(y=0.03, color='orange', linestyle=':', alpha=0.9,  label='y axis = 0.03')
    axs[1,1].set_ylabel('Upper Limit - Lower Limit')
    axs[1,1].set_xlabel(f'Number of Repetitions (one_ref_test is {one_ref_test})')
    axs[1,1].set_title('[Drift] Range of CI')
    axs[1,1].legend()
    print(func.__name__)
    return fig