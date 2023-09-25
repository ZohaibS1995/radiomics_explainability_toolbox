import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from datetime import datetime

from config import *
from source.utils import * 

def plus_one_ai():
    """
    Increments the session state ID to move to the next image in the trial.
    
    This function also captures the time taken by the user to evaluate each image and stores it.
    If all images have been evaluated, the function updates the session state page number to 8.
    """

    # If not the last image, record the time taken and reset the start time
    if st.session_state.id < 80:  
        _record_time_taken()

    # Move to the next image if there are remaining images
    if st.session_state.id < len(st.session_state.image_names) - 1:
        st.session_state.id += 1
        st.session_state.pred_mod_page1 = False

    # If all images have been evaluated, set the page number to 8
    elif st.session_state.id == len(st.session_state.image_names) - 1:
        st.session_state.page_no = 11

def _record_time_taken():
    """
    Helper function to record the time taken to evaluate an image.
    """
    elapsed_time = datetime.now() - st.session_state.start_time
    st.session_state.time_taken[st.session_state.image_names[st.session_state.id]] = elapsed_time
    st.session_state.start_time = datetime.now()

def pred_mod():
    """
        Button Callback to display the AI model prediction
    """
    st.session_state.pred_mod_page1 = True

def ai_trial():
    """
    Displays the AI + Clinicians Trial page in a Streamlit app.
    
    This function provides a side-by-side view of SWE Images with B-mode Ultrasound, and shows
    clinical data along with model predictions for PHLF Risk. It allows users to make predictions
    based on the displayed images and data.
    """

    # Custom styles for radio buttons
    _inject_style("""
        div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
            font-size: 24px;
        }
        div[class*="stRadio"] {
            background-color: #FEA09A;
            border: 3px solid;
        }
    """)

    # Set current page number and display trial progress
    st.session_state.page_no = 2
    st.write(f"## AI + Clinicians Trial: {st.session_state.id+1}/{len(st.session_state.image_names)}")
    st.write("---")

    col1, col2 = st.columns([4, 5])

    # Display SWE Image and B-mode Ultrasound
    with col1:
        st.write("**Top: SWE Image, Bottom: B-mode Ultrasound**")
        image = Image.open(path_orig_images + st.session_state.image_names[st.session_state.id])
        st.image(image)

    # Display clinical data
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

    # Display model prediction
    model_prediction = _get_model_prediction(path_model_prediction)
    _display_model_prediction(model_prediction)

    # User prediction input
    with st.columns([4,4,4])[0]:
        st.radio("2- Select your prediction of PHLF Risk", ("High risk of PHLF", "Low risk of PHLF"), key="ai_pred")

    # Store user prediction in session state
    st.session_state.ai_trial[st.session_state.image_names[st.session_state.id]] = st.session_state.ai_pred

    # Next button
    with st.columns([1,7,1])[2]:
        button_next = st.button("Next", on_click=plus_one_ai, key="plus_one_ai")
    return

def _inject_style(style_str):
    """
    Helper function to inject custom CSS styles into the Streamlit app.
    """
    st.markdown(f"""<style>{style_str}</style>""", unsafe_allow_html=True)


def _get_model_prediction(path):
    """
    Retrieve model prediction data from a CSV file.
    """
    df = pd.read_csv(path, delimiter=";")
    val_dict = {}
    for i in range(len(df["ID"])):
        val_dict[df["ID"][i]] = [df["prediction"][i], df["probability"][i]]
    return val_dict

def _display_model_prediction(predictions):
    """
    Display the AI model's prediction based on the image.
    """
    if not st.session_state.pred_mod_page1:
        st.markdown('<p class="big-font"> Press the button to reveal the model prediction</p>', unsafe_allow_html=True)

    st.button("1- Model Prediction Results of PHLF Risk", on_click=pred_mod, key="disp_pred_mod")

    if st.session_state.pred_mod_page1:
        val_w_k = predictions[int(st.session_state.image_names[st.session_state.id].split(".")[0])]
        st.markdown(f"#### The AI model prediction is :red[{val_w_k[0]}] with a probability of :red[{np.round(float(val_w_k[1].replace(',', '.')), 3)}]")
        st.write("The cut-off value for PHLF prediction is set at 0.35 based on Youden's index.")