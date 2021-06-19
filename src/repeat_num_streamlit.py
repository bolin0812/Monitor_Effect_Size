#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Find optimal number of repetition for CI computation with simulated datasets (streamlit app).

Version | Last Modified |    Author     | Commment
---------------------------------------------------------
1.0     | 2021-06-17    |   Bolin Li    | initial version
"""

import streamlit as st
from src.repeat_times import *
from src.modify_functions import *
from src.precheck_metrics import *

def app():

    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.header('Number of Repetition')
    st.markdown('\n\nFind the optimal number of repetitions used to compute confidence interval.\n')

    sample_size = st.number_input(
                                label = 'Input one value used as the size of simulation datasets.',
                                min_value = 10000, 
                                max_value = 200000,
                                key='sample_size'
    )

    sample_selectbox = st.selectbox(
        label = 'Select simulated datasets with predefined distributions',
        options= ('Continuous Data - Normal Distribution', 
                'Discrete Data - Normal Distribution',
                'Continuous Data - Gamma Distribution',
                'Discrete Data - Gamma Distribution',
                'Categorical Data - Three Levels',  
                'Categorical Data - Seven Levels'), 
        key='repeat_num',
    )

    data_3sets = simulate_datasets(sample_selectbox, sample_size)

    subsample_size = st.slider(label = 'Select subsample size to compute confidence interval with different repetitions.',
                                min_value = 1000,
                                max_value = 200000,
                                step=1000,
                                key = 'subsample_size'
                                )

    metric_name = st.selectbox(
        label = 'Select one metric to quantify data drift.',
        options= ("KS Statistic", 
                "Cramer's V",
                "Cliff's Delta",
                "Common Language Effect Size",
                "Cohen's D"), 
        key='effect size',
    )
    metric_dict = {"KS Statistic": ks_2samp_statistic,
                "Cramer's V": chi2_gf_effsize,
                "Cliff's Delta":CliffD_Effect_Size,
                "Common Language Effect Size": CLE_Effect_Size,
                "Cohen's D":CohenD_Effect_Size}
    st.write(f'Apply {metric_dict[metric_name].__name__} to measure the data drift.')
    st.markdown('**Click the button blow to visualize distributions and evaluate ranges of confidence interval with different number of repetition.**')
    if st.button(label=f"Visualize changes of mean & CI with different numbers of repetition", key='sub_size'):
        
        st.write(f'\nPlot two different large datasets distributions.')
        st.pyplot(visual_two_simulate_dists(data_3sets[0], data_3sets[2], 
                                            title=f'Each large dataset contains {sample_size} Observations'))
        
        st.write(f'\nVisualize how CI changes with different number of repetition.')
        st.write(f'Subsample contains  is {subsample_size} observations.\nFollowing scores are computed by {metric_dict[metric_name].__name__}')
        opt_repeat_num = nums_repetitive_2subsample_metric(data_3sets[0], data_3sets[2],
                                                            subsample_size,
                                                            metric_dict[metric_name],
                                                            [20, 50, 100, 150, 200, 250, 300],
                                                            0.05, 
                                                            one_ref_test=True)
        st.pyplot(opt_repeat_num)
    return