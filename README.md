
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
    │   ├── repeat_times.py            <-- Find the optimal number of repeatition that generates stable CI range. 
    │   │
    │   ├── precheck_metrics.py        <-- Precheck the stability & effectiveness of metrics with simulation or behavior datasets
    │   │
    │   ├── monitor_scores.py          <-- Build one pipeline for users to monitor scores of selected metrics.              
    │   │
    │   ├── repeat_num_streamlit.py    <-- Interactively search the optimal number of repeatition based on one user's inputs
    │   │
    │   └── stable_effect_streamlit.py <-- Interactively check the  the stability & effectiveness of metrics based on one user's inputs
    │
    │
    │
    ├── config.yaml                    <-- Configuration parameters used to upload datasets.
    │
    │
    └── precheck_app.py                <-- Scripts to visualize with multiple streamlit pages.  
--------


## Test monitoring functions
** One app is deployed to heroku (https://precheckmonitor.herokuapp.com/). Users are able to test functions in repeat_times.py and precheck_metrics.py with simulated datasets. **
