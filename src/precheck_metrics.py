#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Precheck the stability & the effectiveness of metrics with simulation amd behavior datasets

Version | Last Modified |    Author     | Commment
---------------------------------------------------------
1.0     | 2021-06-14    |   Bolin Li    | initial version
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly
import yaml
from src.modify_functions import *


def simulate_datasets(variable_type, size=10000):
    
    if variable_type == 'Continuous Data - Normal Distribution':
        # mean =0, std = 1
        data_0 = np.random.normal(loc=0.0, scale=1.0, size=size)
        data_0_ = np.random.normal(loc=0.0, scale=1.0, size=size)
        # mean = 2, std = 1
        data_1 = np.random.normal(loc=2, scale=1.0, size=size)
        
    elif variable_type == 'Discrete Data - Normal Distribution':
        # mean =0, std = 1
        data_0 = np.random.normal(loc=0.0, scale=2.0, size=size)//1
        data_0_ = np.random.normal(loc=0.0, scale=2.0, size=size)//1
        # mean = 2, std = 1
        data_1 = np.random.normal(loc=2, scale=2.0, size=size)//1

    elif variable_type == 'Continuous Data - Gamma Distribution':
        data_0 = np.random.gamma(shape=2, scale=2.0, size=size)
        data_0_ = np.random.gamma(shape=2, scale=2.0, size=size)
        data_1 = np.random.gamma(shape=4, scale=2.0, size=size)

    elif variable_type == 'Discrete Data - Gamma Distribution':
        data_0 = np.random.gamma(shape=2, scale=2.0, size=10000)//1
        data_0_ = np.random.gamma(shape=2, scale=2.0, size=10000)//1
        data_1 = np.random.gamma(shape=4, scale=2.0, size=10000)//1
        
    elif variable_type == 'Categorical Data - Three Levels':
        data_0 = np.random.choice(a=[1,2,3], size=size, p=[0.3, 0.3, 0.4])
        data_0_ = np.random.choice(a=[1,2,3], size=size, p=[0.3, 0.3, 0.4])
        data_1 = np.random.choice(a=[1,2,3], size=size, p=[0.4, 0.4, 0.2])
    
    elif variable_type == 'Categorical Data - Seven Levels':
        data_0 = np.random.choice(a=[1,2,3,4,5,6,7,8], size=size, p=[0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1])
        data_0_ = np.random.choice(a=[1,2,3,4,5,6,7,8], size=size, p=[0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1])
        data_1 = np.random.choice(a=[1,2,3,4,5,6,7,8], size=size, p=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2])
    
    return data_0, data_0_, data_1

def actual_func_score(func, data1, data2):
    """
    Print the actual score with with data1 and data2
    Parameters:
        func: one function used to compare two datasets
        data1: 2D numpy array
        data2: 2D numpy array 

    Returns: 
        name of func and the score of func computed by comparing data1 and data2 
    """
    return print(f"Actual {func.__name__}: {func(data1, data2)}")

def visual_two_simulate_dists(data1, data2, title='title of the plot'):
    """
    Visualize two distributions with data1 and data2
    Parameters:
        data1: 2D numpy array
        data2: 2D numpy array 
        title: string, name of the plot

    Returns: 
        one plot contains two distributions 
    """
    plt.figure(figsize=(7,4))
    plt.hist(data1, alpha=0.7, bins=50, color='dodgerblue', edgecolor='black', 
         label='Large Data Set 1')
    plt.hist(data2, alpha=0.4, bins=50,color='slategrey', edgecolor='black', 
             label='Large Data Set 2')
    
    plt.title(title)
    plt.ylabel('Count')
    plt.xlabel('Value')
    plt.legend()
#     plt.savefig("save_plot")
    plt.show()
    return  

def two_resample_multifunc_visual(a_sample, b_sample, ab_compare_type, true_score,
                                  sample_sizes_list, resample_nums=100, alpha=0.05,*func):
    """
    Evaluate metric stability & effectiveness with repeated subsampling method and simulated datasets

    Parameters:
        func: one function used to compare two datasets
        a_sample: 2D numpy array, the reference set
        b_sample: 2D numpy array, the incoming dataset
        ab_compare_type: one string name for the 1st plot
        true_score: one float value, computed with actual two datasets
        sample_sizes_list: one list of integers, a list of sample sizes
        resample_nums: one integer, number of resampling times for CI
        alpha: one float, the significant level
        func: the predefined function(s), can be used to measure the difference between 2 distributions
    Returns: 
        all_CI: one dict, it contains func name, lists of upper limits, lower limits and mean values
    """
    fig, axs = plt.subplots(len(func),2, figsize=(20, 8*(len(func))))
    all_CI = {}
    for a, f in enumerate(func):   
        
        # all metric scores, percentile values and mean score
        score_mean = []
        score_percentile1 = []
        score_percentile2 = []
        score_all = []
        if f.__name__ == 'CLE_Effect_Size':
            resample_nums = 150
        for size in tqdm(sample_sizes_list):
            resample_repeat = []
            for i in range(resample_nums):
                # keep chaning a and b datasets
                a_dist = np.random.choice(a_sample, size=size, replace=False)
                b_dist = np.random.choice(b_sample, size=size, replace=False)
                score = f(a_dist, b_dist)
                resample_repeat.append(score)
            score_mean.append(np.mean(resample_repeat))
            score_percentile1.append(np.percentile(resample_repeat, 100*alpha/2))
            score_percentile2.append(np.percentile(resample_repeat, 100*(1-(alpha/2))))
            score_all.append(resample_repeat)
            # return lists of upper limits, lower limits and mean values
            all_CI[f.__name__]=[score_percentile1, score_mean, score_percentile2]
            
        # visualization
        if len(func)==1:
            axs1 = axs[0]
            axs2 = axs[1]
        else:
            axs1 = axs[a,0]
            axs2 = axs[a,1]
        #first plot
        axs1.plot(sample_sizes_list, score_mean, linewidth=2, label=f'Mean {f.__name__}')
        axs1.plot(sample_sizes_list, score_percentile1, linewidth=2, label='CI Lower Limit', linestyle='--')
        axs1.plot(sample_sizes_list, score_percentile2, linewidth=2, label='CI Upper Limit', linestyle='--')
        if true_score == 'No':
            pass
        else:
            axs1.axhline(y=true_score, color='black', linestyle=':', alpha=0.8,  label='True Metric Score')
        axs1.legend()
        axs1.set_ylabel(f'{f.__name__} Score')
        axs1.set_xlabel('Subsample Size')
        axs1.set_title(f'{ab_compare_type}')
        #second plot 
        axs2.plot(sample_sizes_list, np.subtract(score_percentile2,score_percentile1), color='dimgray',
                      linewidth=2, label=f'The Range of CI with {alpha} alpha and {resample_nums} repetitions')
        axs2.axhline(y=0.03, color='orange', linestyle=':', alpha=0.9,  label='y axis = 0.03')
        axs2.set_ylabel('Upper Limit - Lower Limit')
        axs2.set_xlabel('Sample Size')
        axs2.set_title('Range of CI')
        axs2.legend()
        #plt.savefig("save_plot")
        
    return all_CI, fig

def st_two_resample_multifunc_visual(a_sample, b_sample, ab_compare_type, true_score,
                                    sample_sizes_list, resample_nums=100, alpha=0.05,*func):
    """
    Evaluate metric stability & effectiveness with repeated subsampling method and simulated datasets

    Parameters:
        func: one function used to compare two datasets
        a_sample: 2D numpy array, the reference set
        b_sample: 2D numpy array, the incoming dataset
        ab_compare_type: one string name for the 1st plot
        true_score: one float value, computed with actual two datasets
        sample_sizes_list: one list of integers, a list of sample sizes
        resample_nums: one integer, number of resampling times for CI
        alpha: one float, the significant level
        func: the predefined function(s), can be used to measure the difference between 2 distributions
    Returns: 
        all_CI: one dict, it contains func name, lists of upper limits, lower limits and mean values
    """
    fig, axs = plt.subplots(len(func),2, figsize=(20, 8*(len(func))))
    all_CI = {}
    for a, f in enumerate(func):   
        
        # all metric scores, percentile values and mean score
        score_mean = []
        score_percentile1 = []
        score_percentile2 = []
        score_all = []
        if f.__name__ == 'CLE_Effect_Size':
            resample_nums = 150
        for size in tqdm(sample_sizes_list):
            resample_repeat = []
            for i in range(resample_nums):
                # keep chaning a and b datasets
                a_dist = np.random.choice(a_sample, size=size, replace=False)
                b_dist = np.random.choice(b_sample, size=size, replace=False)
                score = f(a_dist, b_dist)
                resample_repeat.append(score)
            score_mean.append(np.mean(resample_repeat))
            score_percentile1.append(np.percentile(resample_repeat, 100*alpha/2))
            score_percentile2.append(np.percentile(resample_repeat, 100*(1-(alpha/2))))
            score_all.append(resample_repeat)
            # return lists of upper limits, lower limits and mean values
            all_CI[f.__name__]=[score_percentile1, score_mean, score_percentile2]
            
        # visualization
        if len(func)==1:
            axs1 = axs[0]
            axs2 = axs[1]
        else:
            axs1 = axs[a,0]
            axs2 = axs[a,1]
        #first plot
        axs1.plot(sample_sizes_list, score_mean, linewidth=2, label=f'Mean {f.__name__}')
        axs1.plot(sample_sizes_list, score_percentile1, linewidth=2, label='CI Lower Limit', linestyle='--')
        axs1.plot(sample_sizes_list, score_percentile2, linewidth=2, label='CI Upper Limit', linestyle='--')
        if true_score == 'No':
            pass
        else:
            axs1.axhline(y=true_score, color='black', linestyle=':', alpha=0.8,  label='True Metric Score')
        axs1.legend()
        axs1.set_ylabel(f'{f.__name__} Score')
        axs1.set_xlabel('Subsample Size')
        axs1.set_title(f'{ab_compare_type}')
        #second plot 
        axs2.plot(sample_sizes_list, np.subtract(score_percentile2,score_percentile1), color='dimgray',
                      linewidth=2, label=f'The Range of CI with {alpha} alpha and {resample_nums} repetitions')
        axs2.axhline(y=0.03, color='orange', linestyle=':', alpha=0.9,  label='y axis = 0.03')
        axs2.set_ylabel('Upper Limit - Lower Limit')
        axs2.set_xlabel('Sample Size')
        axs2.set_title('Range of CI')
        axs2.legend()
        
    return all_CI, fig

def stable_multifunc_data(data_root_path, data_a, data_b, features_filename,
                          target='default', feature='pct_solde_limit_months'):
    """
    Upload behavior monthly datasets
    Parameters:
        data_root_path: root path
        data_a: sting, upload one monthly folder used as the reference set
        data_b: sting, upload one incoming folder to compare based on one metric
        features_filename: file name of one monthly dataset
        target: string, stratify based on this column name
        feature: only return subsamples of data_a and data_b based on this feature
    Returns:
        df_a: dataframe, subsample of data_a
        df_b: dataframe, subsample of data_b
    """
    use_feats = []
    use_feats.append(feature)
    use_feats.append(target)

    if data_a == data_b:
        df_a = pd.read_csv(data_root_path + '//' + data_a + '//' + features_filename, 
                            low_memory=True, usecols=use_feats)
        df_b = df_a 
    else:
        df_a = pd.read_csv(data_root_path + '//' + data_a + '//' + features_filename, 
                            low_memory=True, usecols=use_feats)
        df_b = pd.read_csv(data_root_path + '//' + data_b + '//' + features_filename, 
                            low_memory=True, usecols=use_feats)

        return df_a, df_b


def subsample_stratify(df, sub_num, target, feature, random_state=42, stratify=True):
    """
    Stratify-based subsampling based on one feature in one pandas dataframe 

    Parameters:
        df: one dataframe, 
        sub_num: integer, size of subsample
        target: string, target name used for stratification
        feature: string, selected feature name 
        random_state: integer, control randomness
        stratify: boolean, whether conduct stratification or not

    Returns: 
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

def visual_multihist_distributions(data_root_path, data_tests, features_filename,
                                    feature,  target, barmode='group', random_state=42, 
                                    stratify=True, subsample_size=10000):
    
    """
    Visualize multiple monthly datasets with 

    Parameters:
        data_root_path: root path   
        data_tests: list, a list of string names of folders saved in data_to_monitor
        features_filename: file name of one monthly dataset
        target: string, target name used for stratification
        feature: string, selected feature name 
        barmode: types of  plotly histgrams, default is 'group'
        random_state: integer, control randomness
        stratify: boolean, whether conduct stratification or not
        subsample_size: integer, number of observations to select

    Returns:
        NA
    """
    use_feats = []
    use_feats.append(feature)
    use_feats.append(target)
    fig = go.Figure()

    for data_month in tqdm(data_tests):
        df = pd.read_csv(data_root_path + '//' + data_month + '//' + features_filename, 
                         low_memory=True, usecols=use_feats)
        test_sub = subsample_stratify(df, sub_num=subsample_size, random_state=random_state, stratify=stratify, target=target, feature=feature)
        fig.add_trace(go.Histogram(x=test_sub, name=data_month))
        
    fig.update_layout(barmode=barmode, #'overlay', 
                      title_text=f'Distribtuions of {feature} ({subsample_size} obs & {stratify} stratify)')
    fig.update_traces(opacity=0.7)
    fig.show()
    return 



def stable_multifunc_visual_behavior(data_root_path, data_a, data_b, features_filename,
                                      stratify, sample_sizes_list,
                                      resample_nums=100, alpha=0.05, random_state=42,
                                      target='default', feature='pct_solde_limit_months',
                                      *func):
    """
    Check the mean & CI of input metric(s) 

    Parameters:
        data_root_path: root path 
        data_a: string, upload one monthly folder used as the reference set
        data_b: string, upload one incoming folder to compare based on one metric
        features_filename: file name of one monthly dataset
        stratify: boolean, whether or not implement stratified subsampling
        sample_sizes_list: a list of integers, test changes with different sample sizes
        resample_nums: integer, the number of repetitions for one selected sample size
        alpha: float, significant level
        random_state: integer, control randomness
        target: string, stratify based on this column name
        feature: only return subsamples of data_a and data_b based on this feature
        func*: defined function(s) that returns one metric score
        
    Returns:
        CI_ranges_func_df: one dataframe, 
                        index is the sample size, 
                        the df contains metric scores computed by input functions
    """ 
    use_feats = []
    use_feats.append(feature)
    use_feats.append(target)

    if data_a == data_b:
        df_a = pd.read_csv(data_root_path + '//' + data_a + '//' + features_filename, 
                            low_memory=True, usecols=use_feats)
        df_b = df_a 
    else:
        df_a = pd.read_csv(data_root_path + '//' + data_a + '//' + features_filename, 
                            low_memory=True, usecols=use_feats)
        df_b = pd.read_csv(data_root_path + '//' + data_b + '//' + features_filename, 
                            low_memory=True, usecols=use_feats)
    
    fig, axs = plt.subplots(len(func),2, figsize=(16, 5.5*(len(func))))
    CI_ranges_func = []
    for a, f in enumerate(func):        
        score_mean = []
        score_percentile1 = []
        score_percentile2 = []
        score_all = []
        for size in tqdm(sample_sizes_list):
            resample_repeat = []
            a_dist = subsample_stratify(df_a, sub_num=size, random_state=random_state, 
                                        stratify=stratify, target=target, feature=feature)
            for i in range(resample_nums):
                random_num_b = random_state + i
                b_dist = subsample_stratify(df_b, sub_num=size, random_state=random_num_b, 
                                    stratify=stratify, target=target, feature=feature)
                score = f(a_dist, b_dist)
                resample_repeat.append(score)
            score_mean.append(np.mean(resample_repeat))
            score_percentile1.append(np.percentile(resample_repeat, 100*alpha/2))
            score_percentile2.append(np.percentile(resample_repeat, 100*(1-(alpha/2))))
            score_all.append(resample_repeat)
        if len(func)==1:
            axs1 = axs[0]
            axs2 = axs[1]
        else:
            axs1 = axs[a,0]
            axs2 = axs[a,1]
#     first plot
        axs1.plot(sample_sizes_list, score_mean, linewidth=2, label=f'Mean {f.__name__}')
        axs1.plot(sample_sizes_list, score_percentile1, linewidth=2, label='CI Lower Limit', linestyle='--')
        axs1.plot(sample_sizes_list, score_percentile2, linewidth=2, label='CI Upper Limit', linestyle='--')
        axs1.legend()
        axs1.set_ylabel(f'{f.__name__} Score')
        axs1.set_xlabel('Subsample Size')
        axs1.set_title(f'                The relationship between {f.__name__} and sample size')
#     second plot 
        CI_range = np.subtract(score_percentile2,score_percentile1).reshape(-1,1)
        axs2.plot(sample_sizes_list, np.subtract(score_percentile2,score_percentile1), color='dimgray',
                      linewidth=2, label=f'The Range of CI with {alpha} alpha and {resample_nums} repetitions')
        axs2.axhline(y=0.03, color='orange', linestyle=':', alpha=0.9,  label='y axis = 0.03')
        axs2.set_ylabel('Upper Limit - Lower Limit')
        axs2.set_xlabel('Sample Size')
        axs2.set_title('Range of CI')
        axs2.legend()
        CI_ranges_func.append(CI_range)
        
    CI_ranges_func_df = pd.DataFrame(np.concatenate(CI_ranges_func, axis=1), index=sample_sizes_list)
    CI_ranges_func_df.columns = [i.__name__ for i in func]
    return CI_ranges_func_df


def effect1size_baseline_multifunc_visual_behavior(data_root_path, data_a, features_filename, stratify, test_set_size,
                                                  resample_nums=100, alpha=0.05, random_state=42,
                                                  target='default', feature='pct_solde_limit_months',
                                                  *func):
    """
    Visualize baseline scores (mean & CI) of input metric(s) with one specific sample size

    Parameters:
        data_root_path: root path 
        data_a: string, upload one monthly folder used as the reference set
        features_filename: file name of one monthly dataset
        stratify: boolean, whether or not implement stratified subsampling
        test_set_size: integer, one sample size used for repeated subsampling
        resample_nums: integer, the number of repetitions for one selected sample size
        alpha: float, the significant level
        random_state: integer, control randomness
        target: string, stratify based on this column name
        feature: only return subsamples of data_a and data_b based on this feature
        func*: defined function(s) that returns one metric score
        
    Returns:
        NA
    """
    use_feats = []
    use_feats.append(feature)
    use_feats.append(target)
    df_a = pd.read_csv(data_root_path + '//' + data_a + '//' + features_filename, 
                       low_memory=True, usecols=use_feats)
    
    fig, axs = plt.subplots(len(func),2, figsize=(16, 5.5*(len(func))))
    CI_ranges_func = []
    for a, f in enumerate(func): 
        score_mean = []
        score_percentile1 = []
        score_percentile2 = []
        
        resample_repeat = []
        a_dist = subsample_stratify(df_a, sub_num=test_set_size, random_state=random_state, 
                                    stratify=stratify, target=target, feature=feature)
        for i in tqdm(range(resample_nums)):
            random_num2 = random_state + i
            a_dist2 = subsample_stratify(df_a, sub_num=test_set_size, random_state=random_num2, 
                                stratify=stratify, target=target, feature=feature)
            score = f(a_dist, a_dist2)
            resample_repeat.append(score)
        score_mean.append(np.mean(resample_repeat))
        score_percentile1.append(np.percentile(resample_repeat, 100*alpha/2))
        score_percentile2.append(np.percentile(resample_repeat, 100*(1-(alpha/2))))
        if len(func)==1:
            axs1 = axs[0]
            axs2 = axs[1]
        else:
            axs1 = axs[a,0]
            axs2 = axs[a,1]        
        axs1.hist(resample_repeat,bins=50, label=f'Baseline {f.__name__}')
        axs1.axvline(score_mean[0], color='k')
        axs1.axvline(score_percentile1[0], color='k', linestyle='--')
        axs1.axvline(score_percentile2[0], color='k', linestyle='--')
        axs1.set_xlabel(f'{f.__name__} Score')
        axs1.set_ylabel(f'Count in {resample_nums} Resample')
        axs1.set_title(f'Baseline of {f.__name__} for {feature} with {resample_nums} repetitions')
        axs1.legend()
    return 