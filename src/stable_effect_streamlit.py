#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Evaluate the stability and effectiveness of one metric with simulated datasets (streamlit app).

Version | Last Modified |    Author     | Commment
---------------------------------------------------------
1.0     | 2021-06-17    |   Bolin Li    | initial version
"""

import streamlit as st
import numpy as np
from src.repeat_times import *
from src.modify_functions import *
from src.precheck_metrics import *
st.set_option('deprecation.showPyplotGlobalUse', False)

def app():
    st.header('Stability & Effectiveness')
    
    st.markdown('Evaluate the stability and the effectiveness of one metric by changing the size for subsampling.\n')

    sample_size = st.number_input(
                                label = 'Input one value as the size of simulation datasets.',
                                min_value = 10000, 
                                max_value = 20000000,
                                key='sample_size'
    )

    subsample_sizes = st.slider('Select a range of sizes for subsampling', 
                         sample_size//10, sample_size, (sample_size//10, sample_size//2),
                         step=100,
                         key='subsampling_sizes')
    step_size = int((subsample_sizes[1] - subsample_sizes[0])/6//100*100)
    subsample_sizes_list = np.arange(subsample_sizes[0], subsample_sizes[1], step=step_size).tolist()
    # st.write(subsample_sizes_list,'List of Subsample Sizes:', subsample_sizes[0], subsample_sizes[1])

    sample_selectbox = st.selectbox(
        label = 'Select simulated datasets based on predefined distributions',
        options= ('Continuous Data - Normal Distribution', 
                'Discrete Data - Normal Distribution',
                'Continuous Data - Gamma Distribution',
                'Discrete Data - Gamma Distribution',
                'Categorical Data - Three Levels',  
                'Categorical Data - Seven Levels'), 
        key='repeat_num',
    )

    data_3sets = simulate_datasets(sample_selectbox, sample_size)

    num_repeat = st.number_input(label = 'Input one number of repetition to compute confidence intervals.',
                                min_value = 50,
                                max_value = 300,
                                key = 'num_repeat'
                                )

    metric_name = st.selectbox(
        label = 'Select one metric to measure data drifts.',
        options= ("KS Statistic", 
                "KS Test P-value",
                "AD Statistic",
                "AD Significance Level",
                "MWU Statistic",
                "MWU P-value",
                "Cliff's Delta",
                "Common Language Effect Size",
                "Chi2 Statistic",
                "Chi2 P-value",
                "Cramer's V",
                "Cohen's D",
                "Hedges' G"), 
        key='effect size',
    )
    metric_dict={"KS Statistic": ks_2samp_statistic,
                "KS Test P-value":ks_2samp_pvalue,
                "AD Statistic": ad_2sample_statistic,
                "AD Significance Level": ad_2sample_significance_level,
                "MWU Statistic":mwu_2sample_statistic,
                "MWU P-value":mwu_2sample_pvalue,
                "Cliff's Delta":CliffD_Effect_Size,
                "Common Language Effect Size": CLE_Effect_Size,
                "Chi2 Statistic":chisquare_statistic,
                "Chi2 P-value":chisquare_pvalue,
                "Cramer's V": chi2_gf_effsize,
                "Cohen's D":CohenD_Effect_Size,
                "Hedges' G": HedgesG_Effect_Size }
    st.write(f'Apply {metric_dict[metric_name].__name__} to measure the data drift.')

    st.markdown('**Click the button blow to visualize distributions and evaluate the stability & the effectiveness of one metric with different subsample size.**')

    if st.button(label=f"Visualize calculated mean & CI with different subsampling sizes", key='sub_size'):
        
        st.write(f'**[Drift]** Plot distributions of two different large datasets.')
        st.pyplot(visual_two_simulate_dists(data_3sets[0], data_3sets[2], 
                                            title=f'Each dataset contains {sample_size} observations'))
        
        st.write(f'**Visualize metric scores calculated with above datasets by changing subsample size.**')
        st.write(f'Repeatedly compute comparison scores with {metric_dict[metric_name].__name__}.')
        CIs_drift, fig_drift= st_two_resample_multifunc_visual(data_3sets[0], data_3sets[2],
                                                        f'Evaluate the Stability and the Effectiveness for {metric_dict[metric_name].__name__}',
                                                        metric_dict[metric_name](data_3sets[0], data_3sets[2]),
                                                        subsample_sizes_list,
                                                        num_repeat,
                                                        0.05, 
                                                        metric_dict[metric_name])
        st.pyplot(fig_drift)

        st.write(f'**[No Drift]** Plot two different large datasets distributions.')
        st.pyplot(visual_two_simulate_dists(data_3sets[0], data_3sets[1], 
                                            title=f'Each large dataset contains {sample_size} observations'))
        
        st.write(f'**Visualize how metric scores calculated with above datasets by changing subsample size.**')
        st.write(f'Repeatedly compute comparison scores with {metric_dict[metric_name].__name__}.')
        CIs_nodrift, fig_nodrift= st_two_resample_multifunc_visual(data_3sets[0], data_3sets[1],
                                                        f'Evaluate the Stability and the Effectiveness for {metric_dict[metric_name].__name__}',
                                                        metric_dict[metric_name](data_3sets[0], data_3sets[1]),
                                                        subsample_sizes_list,
                                                        num_repeat,
                                                        0.05, 
                                                        metric_dict[metric_name])
        st.pyplot(fig_nodrift)
    
    return

