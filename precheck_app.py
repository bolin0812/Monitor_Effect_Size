# -*- coding: utf-8 -*-


"""
Precheck monitoring app for metrics or functions in modify_functions.py.

Version | Last Modified |    Author     | Commment
---------------------------------------------------------
1.0     | 2021-06-17    |   Bolin Li    | initial version
"""

import streamlit as st
from src.repeat_times import *
from src.modify_functions import *
from src.precheck_metrics import *
from src import repeat_num_streamlit, stable_effect_streamlit

PAGES={
    "Number of Repetition": repeat_num_streamlit,
    "Stability & Effectiveness": stable_effect_streamlit
}

st.sidebar.title('Monitoring Precheck')  
st.sidebar.subheader("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()

