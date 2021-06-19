#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Modify functions (metrics) so they can be used as inputs of precheck functions or monitoring pipeline.

Version | Last Modified |    Author     | Commment
---------------------------------------------------------
1.0     | 2021-06-14    |   Bolin Li    | initial version
"""

import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg

"""
Parameters:

These functions are mainly used to compare two data sets: a and b.

    a: 2D numpy array, reference dataset
    b: 2D numpy array, incoming dataset 

Returns: 
    score of one metric
    score can be one statistical score, p-value or effect size score. 
"""
# Statistic Tests
def ks_2samp_statistic(a,b):
    ks_stat, p_value = stats.ks_2samp(a, b, alternative='two-sided', mode='auto')
    return ks_stat

def ks_2samp_pvalue(a,b):
    ks_stat, p_value = stats.ks_2samp(a, b, alternative='two-sided', mode='auto')
    return p_value

def ad_2sample_statistic(a,b):
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    statistic, _, significance_level = stats.anderson_ksamp([a, b])
    return statistic

def ad_2sample_significance_level(a,b):
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    statistic, _, significance_level = stats.anderson_ksamp([a, b])
    return significance_level

def mwu_2sample_statistic(a,b):
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    statistic, pvalue = stats.mannwhitneyu(a, b)
    return statistic

def mwu_2sample_pvalue(a,b):
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    statistic, p_value = stats.mannwhitneyu(a, b)
    return p_value

def chisquare_statistic(a,b):
    a_cat, a_counts = np.unique(a, return_counts=True)
    b_cat, b_counts = np.unique(b, return_counts=True)
    statistic, pvalue = stats.chisquare(a_counts, b_counts)
    return statistic

def chisquare_pvalue(a,b):
    a_cat, a_counts = np.unique(a, return_counts=True)
    b_cat, b_counts = np.unique(b, return_counts=True)
    statistic, pvalue = stats.chisquare(a_counts, b_counts)
    return pvalue

# normal distribution
def CohenD_Effect_Size(a,b):
    return pg.compute_effsize(a, b, paired=False, eftype='cohen')

def HedgesG_Effect_Size(a,b):
    return pg.compute_effsize(a, b, paired=False, eftype='hedges')

# commmon effect size 
def CLE_Effect_Size(a,b):
    return pg.compute_effsize(a, b, paired=False, eftype='CLES')

# cliff d designed for categorical ordinal data & MWU test
def CliffD_Effect_Size(a,b):
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    statistic, p_value = stats.mannwhitneyu(a, b)
    d = statistic*2/(len(a)*len(b)) -1
    return d

# KS statistics can be viewed one effect size -- max distance betweem 2 distributions
def ks_2samp_statistic(a,b):
    ks_stat, p_value = stats.ks_2samp(a, b, alternative='two-sided', mode='auto')
    return ks_stat

# 2 types of cramer's V  effect size built for 
def chi2_gf_effsize(a,b):
    n = len(a)+len(b)
    a_cat, a_counts = np.unique(a, return_counts=True)
    b_cat, b_counts = np.unique(b, return_counts=True)
    v_star = min(len(a_cat), len(b_cat)) -1 
    statistic, pvalue = stats.chisquare(b_counts, a_counts)
    V = np.sqrt(statistic/(n*v_star))
    return V

def chi2_gf_effsize2(a,b):
    n = len(a)+len(b)
    a_cat, a_counts = np.unique(a, return_counts=True)
    b_cat, b_counts = np.unique(b, return_counts=True)
    v = (len(a_cat)-1)*(len(b_cat)-1)
    statistic, pvalue = stats.chisquare(b_counts, a_counts)
    V = np.sqrt(statistic/(n*v))
    return V
