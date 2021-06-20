
# Monitoring with Effect Size
Evaluate and apply various effect sizes in the context of measuring data drifts.
<br/><br/>

## Precheck App
**One app is deployed to heroku (https://precheckmonitor.herokuapp.com/). Users are able to test several effect size functions with simulated datasets.**

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

## Understand Effect Size

|    Effect Size Studies    |  Key Points  |
|  :---------  | :------:  |
|  [What does effect size tell you? ](https://www.simplypsychology.org/effect-size.html)  |Introduce Cohen's d and Pearson r correlation. Why report effect sizes?| 
|  [It's the Effect Size, Stupid](https://dradamvolungis.files.wordpress.com/2012/01/its-the-effect-size-stupid-what-effect-size-is-why-it-is-important-coe-2002.pdf)  | How to interprete effect sizes, and What is the margin for error in estimating effect sizes |
|  [Effect size, confidence interval and statistical significance: a practical guide for biologists](https://pubmed.ncbi.nlm.nih.gov/17944619/) |Combined use of an effect size and its CIs enables one to assess the relationships within data more effectively than the use of p values, regardless of statistical significance. |
|  [Using Effect Size—or Why the P Value Is Not Enough](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3444174/)  | Effect size is the magnitude of the difference between groups. Differentiate absolute and standardized Effect Size. Why do we report effect size and why p-value is not enough. |
|  [The importance of effect sizes](https://www.tandfonline.com/doi/full/10.3109/13814788.2013.818655)  | Introduce effect size calculated with relative risk and odds ratio. |
|  [Understanding Confidence Intervals (CIs) and Effect Size Estimation ](https://www.psychologicalscience.org/observer/understanding-confidence-intervals-cis-and-effect-size-estimation)  |  |
|  [Effect Size](https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118625392.wbecp048)  |  |
|  [The simple difference formula: an approach to teaching nonparametric correlatio](https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118625392.wbecp048)  | Introduce the Common Language Effect Size, Rank-Biserial Correlation, the Simple Difference Formula and the Wilcoxon Signed-rank Test. |
|  [Multi-sample comparison of detrital age distributions](https://www.ucl.ac.uk/~ucfbpve/papers/VermeeschChemGeol2013.pdf)  | The KS statistic is most sensitive to the region near the modes of the sample distributions, and less sensitive to their tails. The KS effect size can be used as a dissimilar measure since it fulfills the triangle inequality. This paper also explains why p-values are unsuitable as a measure of dissimilarity. |
