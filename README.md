
# Monitoring Precheck
Evaluate statistical metrics and effect sizes in the context of measuring data drifts.
<br/><br/>

**Run Streamlit App**
To find optimal number of repetitions and appropriate size of subsamples with simulated datasets, users can input following command and interactively test with one app:
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
    │   ├── modify_functions.py        <-- Modify functions or metrics, so they can be used as inputs of precheck functions or monitoring pipeline.
    │   │
    │   ├── repeat_times.py            <-- Find the optimal number of repeatition that generates stable CI range. 
    │   │
    │   ├── precheck_metrics.py        <-- Precheck the stability & the effectiveness of metrics with simulations or behavior datasets
    │   │
    │   ├── monitor_scores.py          <-- Build one pipeline for users to monitor scores of selected metrics.              
    │   │
    │   ├── repeat_num_streamlit.py    <-- Interactively search to find the optimal number of repeatition based on one user's inputs
    │   │
    │   └── stable_effect_streamlit.py <-- Interactively check and evaluate the stability & the ffectiveness of metrics based on one user's inputs
    │
    │
    │
    ├── config.yaml                    <-- Configuration parameters used to upload datasets.
    │
    │
    └── precheck_app.py                <-- Scripts to visualize with multiple pages in one streamlit app.  
--------

## Precheck Effect Size
**One app is deployed to heroku (https://precheckmonitor.herokuapp.com/). Users are able to test several effect size functions with simulated datasets.**
