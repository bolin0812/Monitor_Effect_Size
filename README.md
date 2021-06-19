
# Monitoring Precheck
The main objective of this project is to evaluate effect size in the context of measuring data drifts.
<br/><br/>

**Run Streamlit App**
Train models for all possible Matcher ID AI, you can input following command:
`streamlit run precheck_app.py`

## Project Organization
------------
    ├── README.md                      <-- The top-level README for developers using this project.
    │
    │
    ├── requirements.txt               <-- The requirements file for reproducing the analysis environment.
    │            
    │
    ├── src                   
    │   │
    │   ├── modify_functions.py        <-- Modify functions (metrics) so they can be used as inputs of precheck functions or monitoring pipeline.
    │   │
    │   ├── repeat_times               <-- Find the optimal number of repeatition that generates stable CI range. 
    │   │
    │   ├── precheck_metrics           <-- Precheck the stability & effectiveness of metrics with simulation amd behavior datasets
    │   │
    │   ├── monitor_scores             <-- Build one pipeline for users to monitor scores of selected metrics.              
    │   │
    │   ├── repeat_num_streamlit       <-- Interactively search the optimal number of repeatition based on one user's inputs
    │   │
    │   └── stable_effect_streamlit    <-- Interactively check the  the stability & effectiveness of metrics based on one user's inputs
    │
    │
    │
    ├── config.yaml                    <-- Configuration parameters used to upload datasets.
    │
    │
    └── prechec_app.py                 <-- Scripts to visualize with multiple streamlit pages.  
--------
