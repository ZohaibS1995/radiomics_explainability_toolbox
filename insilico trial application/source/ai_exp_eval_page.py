import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from datetime import datetime 

from config import *
from source.utils import * 

def plus_one_ai_exp():
    """
    Increments the image ID and handles the timer for each image.
    """
    # If the current ID is less than 80, record the time taken to view the current image
    if st.session_state.id < 80:
        st.session_state.time_taken[st.session_state.image_names[st.session_state.id]] = datetime.now() - st.session_state.start_time
        st.session_state.start_time = datetime.now()
    
    # Increment image ID unless it's already at the maximum
    if st.session_state.id < len(st.session_state.image_names):
        st.session_state.id += 1
        st.session_state.pred_mod_page1 = False

    # If the image ID reaches the maximum, update the page number
    if st.session_state.id == len(st.session_state.image_names):
        st.session_state.page_no = 12
    return

def explainability_disp_t():
    """
    Set the session state page number for the explainability display.
    """
    st.session_state.page_no = 6

def pred_mod():
    """
        Button Callback to display the AI model prediction
    """
    st.session_state.pred_mod_page1 = True
    
def ai_trial_explanations():
    """
    Displays the AI trial explanations page in the Streamlit app.
    """

    # Apply CSS styles to specific components
    st.markdown(
        """
        <style>
        div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
            font-size: 24px;
        }
        div[class*="stRadio"] {
            background-color: #FEA09A;
            border: 3px solid;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # Set the page number and display the header
    st.session_state.page_no = 3
    st.write("## AI + Clinicians + Explanations Trial: " + str(st.session_state.id+1) + "/" + str(len(st.session_state.image_names)))
    st.write("---")

    # Display images
    col1, col2 = st.columns([4, 5])
    with col1:
        st.write("**Top: SWE Image, Bottom: B-mode Ultrasound**")
        image = Image.open(path_orig_images + st.session_state.image_names[st.session_state.id])
        st.image(image)

    # Display clinical data alongside images
    with col2:
        df = pd.read_csv(path_clinical_data)
        # Iterate over columns and convert if all values are floats
        for col in df.columns:
            # Check if all values in the column can be converted to float
            if df[col].apply(is_float).all():
                df[col] = df[col].astype(float).round(2)
                
        keys_n = list(df.columns)[2:]
        val_w_k = list(df.iloc[st.session_state.id, 2:].to_numpy())
        normal_range = list(pd.read_csv(path_clinical_normal).to_numpy().transpose())

        data = np.vstack((keys_n, val_w_k, normal_range)).transpose()
        v_df = pd.DataFrame(data, columns=["Clinical Variable", "Value", "Normal Range"])
        hyperlinks = list(pd.read_csv(path_clinical_hyperlinks, delimiter=";")["Hyperlinks"].to_numpy().transpose())
        v_df['Clinical Variable'] = [apply_hyperlink(item, link) for item, link in zip(v_df['Clinical Variable'], hyperlinks)]

    
        styled_df = v_df.style.apply(color_rows, axis=1)

        st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)
        st.write(f"##### For explanation of clinical variables, [click here]({URL_variable_explanation})")

    # Extract predictions for the current image
    df_pred = pd.read_csv(path_model_prediction, delimiter=";")
    val_dict = {df_pred["ID"][i]: [df_pred["prediction"][i], df_pred["probability"][i]] for i in range(len(df_pred["ID"]))}

    # Handle model prediction results
    if not st.session_state.pred_mod_page1:
        st.markdown('<p class="big-font"> Press the button to reveal the model prediction</p>', unsafe_allow_html=True)
    
    st.button("1- Model Prediction Results of PHLF Risk", on_click=pred_mod, key="disp_pred_mod")

    # Display model predictions when available
    if st.session_state.pred_mod_page1:
        val_w_k = val_dict[int(st.session_state.image_names[st.session_state.id].split(".")[0])]
        st.markdown("#### The AI model prediction is :red["+ str(val_w_k[0])+ "] with a probability of  :red["+ str(np.round(float(val_w_k[1].replace(",", ".")), 3))+ "]")

    # Explainability page button
    explainability_page = st.button("2- Explainability", on_click=explainability_disp_t, key="exp_page")

    # UI elements for clinician's prediction
    col1, col2, col3 = st.columns([4,4,4])
    with col1:
        st.radio("3- Select your prediction of PHLF Risk", ("High risk of PHLF", "Low risk of PHLF"), key="ai_exp_pred")

    st.session_state.ai_exp_trial[st.session_state.image_names[st.session_state.id]] = st.session_state.ai_exp_pred

    # Navigation buttons
    col1, col2, col3 = st.columns([1,7,1])
    with col3:
        button_next = st.button("Next", on_click=plus_one_ai_exp, key="plus_one_ai_exp")
    
    return
