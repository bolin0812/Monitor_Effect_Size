#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build one pipeline for users to monitor scores of selected metrics. 

Version | Last Modified |    Author     | Commment
---------------------------------------------------------
1.0     | 2021-06-14    |   Bolin Li    | initial version
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt


def subsample_stratify(df, sub_num, target, feature, random_state=42, stratify=True):
    """
    Stratify-based subsampling based on one feature in one pandas dataframe 

    Input:
        df: one dataframe, 
        sub_num: integer, size of subsample
        target: string, target name used for stratification
        feature: string, selected feature name 
        random_state: integer, control randomness
        stratify: boolean, whether conduct stratification or not

    Output: 
        1D numpy array, the subsample of one relevant feature 
    """
    if stratify == True:
        _, df_sub = train_test_split(df[feature], test_size=sub_num, 
                                     random_state=random_state, 
                                     stratify=df[target])
    else:
        _, df_sub = train_test_split(df[feature], test_size=sub_num, 
                                     random_state=random_state, 
                                     stratify=None)
    return df_sub.to_numpy()



def repeated_func_scores_behavior(data_root_path, data_a, data_b, features_filename, stratify, func, size=192433,
                                  repeat_num=100, random_state=42,one_ref_test=True,
                                  target='default',feature='pct_solde_limit_months'):
    """
    Compute  the mean & CI of input metric(s) based on one specific sample size

    Input:
        data_root_path: root path 
        data_a: string name, upload one monthly folder used as the reference set
        data_b: string name, upload one incoming folder to compare based on one metric
        features_filename: file name of one monthly dataset
        stratify: boolean, whether or not implement stratified subsampling
        func: one defined function that returns one metric score
        size: integer, the number of selected observations for monitoring
        repeat_num: integer, the number of repetitions for one selected sample size
        random_state: integer, control randomness
        one_ref_test: 4 types of subsampling 
                    - True: only use one subset of data_a to compare with different subsets of data_a
                    - False: changing both subsets of data_a and data_b to make comparisons
                    - boostrap_a: baseline computed by boostraping data_a
                    - boostrap_b: baseline computed by boostraping data_b
        target: string, stratify based on this column name
        feature: string, only return subsamples of data_a and data_b based on this feature

    Output:
        CI_ranges_func_df: one dataframe, 
                        index is the sample size, 
                        the df contains metric scores computed by input functions
    """
    
    use_feats = []
    use_feats.append(feature)
    use_feats.append(target)
    repeated_scores = []
    if data_a == data_b:
        change_random = 1
        df_a = pd.read_csv(data_root_path + '//' + data_a + '//' + features_filename, 
                            low_memory=True, usecols=use_feats)
        df_b = df_a 
    else:
        change_random = 0
        df_a = pd.read_csv(data_root_path + '//' + data_a + '//' + features_filename, 
                            low_memory=True, usecols=use_feats)
        df_b = pd.read_csv(data_root_path + '//' + data_b + '//' + features_filename, 
                            low_memory=True, usecols=use_feats)
    # one reference test 
    if one_ref_test == True:
        a_dist = subsample_stratify(df_a, sub_num=size, random_state=random_state, 
                                    stratify=stratify, target=target, feature=feature)
        for i in range(repeat_num):  
            b_dist_i = subsample_stratify(df_b, sub_num=size, random_state=random_state+i+change_random, 
                                          stratify=stratify, target=target, feature=feature)
            score_ab = func(a_dist, b_dist_i)
            repeated_scores.append(score_ab)
    
    # change reference test 
    elif one_ref_test == False:
        for i in range(repeat_num):
                
            a_dist_i = subsample_stratify(df_a, sub_num=size, random_state=random_state+i, 
                                          stratify=stratify, target=target, feature=feature)
            b_dist_i = subsample_stratify(df_b, sub_num=size, random_state=random_state+i+change_random, 
                                          stratify=stratify, target=target, feature=feature)
            score_ab = func(a_dist_i, b_dist_i)
            repeated_scores.append(score_ab)
            
    # Bootstrapping only used to compute baseline
    else:
        if one_ref_test == 'boostrap_a':
            ref_data = subsample_stratify(df_a, sub_num=size, random_state=random_state, 
                                          stratify=stratify, target=target, feature=feature)
        elif one_ref_test == 'boostrap_b':
            ref_data = subsample_stratify(df_b, sub_num=size, random_state=random_state, 
                                          stratify=stratify, target=target, feature=feature)           
        for i in range(repeat_num):
            bootstrap_rows = np.random.choice(np.arange(size), size)
            ref_bstrap_i = ref_data[bootstrap_rows]
            score_ref_bstrap = func(ref_data, ref_bstrap_i)
            repeated_scores.append(score_ref_bstrap)
            
    return repeated_scores
    

def multi_compare_ab_scores(data_root_path, data_a, data_b_list, features_filename, stratify, func,
                            size, repeat_num, random_state, 
                            target, feature, alpha):
    """
    Apply one repeated subsampling method, 
    compare reference data with multiple incoming datasets to compute lists of CI and mean.
    By default, one_ref_test is False when calculating the baseline (data_a), 
    and one_ref_test is True when comparing data_a with data_b.

    Input:
        data_root_path: root path 
        data_a: string name, upload one monthly folder used as the reference set
        data_b_list:a list string names, they are names of folders that contain saved datasets
        features_filename: file name of one monthly dataset
        stratify: boolean, whether or not implement stratified subsampling
        func: one defined function that returns one metric score
        size: integer, the number of selected observations for monitoring
        repeat_num: integer, the number of repetitions for one selected sample size
        random_state: integer, control randomness
        feature: string, only return subsamples of data_a and data_b based on this feature
        alpha: float, one significant level used to compute percentile values
        
    Output:
        score_dict: dictionary, CI & mean score for each comparison between data_a and data_b
        all_scores: all scores computed with func and subsampling by comparing data_a and data_b 
    """  
    # data a is reference and data b list contains all monthly data
    score_dict = {}
    # save mean, upper and lower limits
    score_mean = []
    score_percentile1 = []
    score_percentile2 = []
    all_scores = {}
    for data_b in tqdm(data_b_list):
        # one list of a_b scores 
        if data_b == data_a:
            repeat_scores = repeated_func_scores_behavior(data_root_path, data_a, data_b, features_filename,
                                                          stratify, func, 
                                                          size, repeat_num, random_state,False, 
                                                          target, feature)
        else:
            repeat_scores = repeated_func_scores_behavior(data_root_path, data_a, data_b, features_filename,
                                                          stratify, func, 
                                                          size, repeat_num, random_state,True, 
                                                          target, feature)
        score_mean.append(np.mean(repeat_scores))
        score_percentile1.append(np.percentile(repeat_scores, 100*alpha/2))
        score_percentile2.append(np.percentile(repeat_scores, 100*(1-(alpha/2))))
        all_scores[data_b] = repeat_scores
        
    score_dict['mean'] = score_mean
    score_dict['percentile1'] = score_percentile1
    score_dict['percentile2'] = score_percentile2
    return score_dict, all_scores


def visual_monitor_behavior_list(data_b_list, score_dict, func, alpha=0.05,
                                feature='pct_solde_limit_months',repeat_num=50, one_ref_test=False):
    """ 
    Visualize CI and mean scores.

    Input:
        data_b_list: list, a list of string names representing incoming datasets for comparisions
        score_dict: dictionary, CI & mean score for each comparison between data_a and data_b
        func: one defined function that returns one metric score
        alpha: float, one significant level used to compute percentile values
        feature: string, only return subsamples of data_a and data_b based on this feature
        repeat_num: integer, the number of repetitions for one selected sample size
        one_ref_test: True, Flase or String, subsampling methods 
        
    Output: 
        NA
    """
    fig, axs = plt.subplots(1,2, figsize=(16, 5.5))
    axs[0].plot(data_b_list, score_dict['mean'], linewidth=2, marker='o', label=f'Mean {func.__name__}')
    axs[0].plot(data_b_list, score_dict['percentile1'], linewidth=2,marker='o', label='CI Lower Limit')
    axs[0].plot(data_b_list, score_dict['percentile2'], linewidth=2,marker='o', label='CI Upper Limit')
    axs[0].legend()
    axs[0].axhline(score_dict['mean'][0], color='k')
    axs[0].axhline(score_dict['percentile1'][0], color='k', linestyle='--')
    axs[0].axhline(score_dict['percentile2'][0], color='k', linestyle='--')
    axs[0].fill_between(data_b_list, score_dict['percentile1'],score_dict['percentile2'], 
                        facecolor='grey', alpha=0.25)
    axs[0].set_title(f'Monitor new data with {func.__name__}\nFeature {feature}')
    axs[1].plot(data_b_list, np.subtract(score_dict['percentile2'],score_dict['percentile1']), color='dimgray',
                marker='o', linewidth=2, label=f'The Range of CI')
    axs[1].set_ylabel('Upper Limit - Lower Limit')
    if one_ref_test == True:
        ref_test_type = 'constant refer test set'
    elif one_ref_test == False:
        ref_test_type = 'changing refer test set'
    else:
        ref_test_type = one_ref_test
    axs[1].set_title(f'Range of CI ({alpha} alpha)\nCompute {repeat_num} times with {ref_test_type}')
    axs[1].legend()
    return 


def monitor_behavior_func_pipeline(data_root_path, data_a, data_b_list, features_filename,
                                   stratify, func, size, repeat_num, random_state, one_ref_test, 
                                   target='default', feature='pct_solde_limit_months', alpha=0.05):
    """
    One pipeline to compute key statistics and all comparison scores. 

    Input:
        data_root_path: root path 
        data_a: string name, upload one monthly folder used as the reference set
        data_b_list:a list string names, they are names of folders that contain saved datasets
        features_filename: file name of one monthly dataset
        stratify: boolean, whether or not implement stratified subsampling
        func: one defined function that returns one metric score
        size: integer, the number of selected observations for monitoring
        repeat_num: integer, the number of repetitions for one selected sample size
        random_state: integer, control randomness
        one_ref_test: True, Flase or String, subsampling methods 
        target: string, stratify based on this column name
        feature: string, only return subsamples of data_a and data_b based on this feature
        alpha: float, one significant level used to compute percentile values
        
    Output:
        score_dict: dictionary, CI & mean score for each comparison between data_a and data_b
        all_scores: all scores computed with func and subsampling by comparing data_a and data_b 
    """  
    # compute key statistics and all scores
    score_dict, all_scores = multi_compare_ab_scores(data_root_path, data_a, data_b_list,features_filename,
                                                     stratify,func,size,repeat_num, random_state,
                                                     target, feature, alpha)
    # visualization
    visual_monitor_behavior_list(data_b_list, score_dict, func, alpha, feature, repeat_num, one_ref_test)
    
    return score_dict, all_scores