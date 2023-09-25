#!/usr/bin/env python
# -- coding: utf-8 --
# @Time : 5/31/2023 1:52 PM
# @Author : Zohaib Salahuddin
# @File : Main_page.py

# General Imports
import os
import time
import pandas as pd
from PIL import Image
import numpy as np
from datetime import datetime
import streamlit as st
from firebase_admin import firestore

# Local Imports
from config import *
from source.information_page import *
from source.trial_sel_page import *
from source.usability_eval_page import *
from source.exp_usability_page import *
from source.exp_trial_page import *
from source.ai_exp_eval_page import *
from source.ai_only_eval_page import *
from source.final_page_exp import *
from source.final_page_usability import *
from source.final_page_ai import *
from source.end_questionaire import *
from source.utils import *

if __name__ == "__main__":

    # Initializing Firestore database
    st.session_state["db"] = firestore.Client.from_service_account_json("firestore-key.json")

    # Setting Streamlit configuration
    st.set_page_config(page_title='In silico Trial', layout="wide")
    add_bg_from_local('img.jpg')    

    # Load prediction model data
    df = pd.read_csv(path_model_prediction, delimiter=";")

    # Hide Streamlit's default hamburger menu and footer
    st.markdown("""
    <style>
    .css-nqowgj.e1ewe7hr3,
    .css-164nlkn.e1g8pov61
    {
        visibility: hidden;
    }
    </style>
    """, unsafe_allow_html=True)
        
    # Session state initialization
    # If any session variable is not already present, initialize it
    st.session_state.setdefault("u_name", list(np.loadtxt(r"./trial/usability_ID.txt")))
    temp = [int(x) for x in st.session_state.u_name]
    st.session_state.setdefault("u_id", [df[df['ID'] == x].index[0] for x in temp])
    st.session_state.setdefault("page_no", -1)
    st.session_state.setdefault("id", 0)
    st.session_state.setdefault("pred_mod_page1", False)
    st.session_state.setdefault("image_names", os.listdir(path_orig_images))
    st.session_state.setdefault("counterfactual_slider", 0.1)
    st.session_state.setdefault("usability_questionaire", {})
    st.session_state.setdefault("usability_pred", {})
    st.session_state.setdefault("causability_questionaire", {})
    st.session_state.setdefault("ai_trial", {})
    st.session_state.setdefault("ai_exp_trial", {})
    st.session_state.setdefault("time_taken", {})
    st.session_state.setdefault("required_flag", False)
    st.session_state.setdefault("name_user", "")
    st.session_state.setdefault("nationality_user", "")
    st.session_state.setdefault("department_user", "")
    st.session_state.setdefault("years_of_experience_user", "")
    st.session_state.setdefault("speciality_user", "")
    st.session_state.setdefault("end_questionnaire_ai", "")
    st.session_state.setdefault("end_questionnaire_ai_exp", "")

    # Page navigation
    # Depending on the value of 'page_no', a different function (page) is called
    pages = {
        -1: info_page,
        0: landing_page,
        1: usability_page,
        2: ai_trial,
        3: ai_trial_explanations,
        4: explainability_page,
        5: usability_questionaire,
        6: explainability_page_trial,
        7: final_page_u,
        8: final_page_a,
        9: final_page_ex,
        10: system_causability_scale,
        11: end_questionaire_ai_only,
        12: end_questionaire_ai_exp,
    }

    # Calling the corresponding function for the current page
    pages[st.session_state.page_no]()
