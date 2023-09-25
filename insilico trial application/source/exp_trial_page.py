import os
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from config import *
from source.utils import *


# Function to decrease the value of the counterfactual slider
def decrease_slider():
    st.session_state.counterfactual_slider = max(0.1, st.session_state.counterfactual_slider - 0.1)

# Function to increase the value of the counterfactual slider
def increase_slider():
    st.session_state.counterfactual_slider = min(0.9, st.session_state.counterfactual_slider + 0.1)

def back_return_t():
    """
    Set the session state page number to return to the previous page.
    """
    st.session_state.page_no = 3

def explainability_page_trial():
    """
    Renders an explainability page for the AI model's trial results.
    Displays various images related to the model's prediction and counterfactual explanations.
    """

    # Set the page number in session state for navigation purposes
    st.session_state.page_no = 6

    # Loading model predictions from the CSV file
    df_pred = pd.read_csv(path_model_prediction, delimiter=";")
    val_dict = {
        df_pred["ID"][i]: [df_pred["prediction"][i], df_pred["probability"][i]]
        for i in range(len(df_pred["ID"]))
    }

    # Extracting model prediction for the current image
    val_w_k = val_dict[int(st.session_state.image_names[st.session_state.id].split(".")[0])]
    prob = np.round(float(val_w_k[1].replace(",", ".")), 3)

    # Display counterfactual explanations section
    st.markdown('#### :red[Counterfactual Explanations]')
    st.write("##### For explanation of counterfactual explanation, click [here](LINK)")
    st.write(f"##### Model Predicted Probability: :red[{prob}]")

    # Display preprocessed, reconstructed, and counterfactual images side by side
    col1, col2, col3 = st.columns([5, 5, 5])

    # Preprocessed Image
    with col1:
        st.write("**Preprocessed Image**")
        st.write("SWE image is preprocessed to remove marking and b-mode ultrasound.")
        image = Image.open(path_preprocessed + st.session_state.image_names[st.session_state.id])
        st.image(image.resize((224, 224)))

    # Reconstructed Image
    with col2:
        st.write("**Reconstructed Image**")
        st.write("This reconstructed image is used for generating counterfactual explanations.")
        image = Image.open(path_reconstructed + st.session_state.image_names[st.session_state.id])
        st.image(image.resize((224, 224)))

    # Counterfactual Image
    col1, col2, col3 = st.columns([5, 3, 3])
    with col1:
        st.write("**Counterfactual Image**")
        st.write("This image is produced by applying minimal perturbation to the original image so that the model's prediction matches the selected probability.")
        counterfactual_path = os.path.join(path_counterfactual + st.session_state.image_names[st.session_state.id].split(".")[0],
                                           f"0_{int(st.session_state.counterfactual_slider * 10)}.png")
        st.image(Image.open(counterfactual_path).resize((224, 224)))

    # Counterfactual probability slider
    col1, col2, col3, col4 = st.columns([0.4, 4, 0.4, 14])
    with col1:
        st.button(r"\-", on_click=decrease_slider)
    with col2:
        val = st.slider('Select the Counterfactual Probability:', min_value=0.1, max_value=0.9, step=0.1, key="counterfactual_slider")
    with col3:
        st.button(r"\+", on_click=increase_slider)

    # Display Layerwise Relevance Propagation (LRP) Explanations
    st.markdown('#### :red[Layerwise Relevance Propagation (LRP) Explanations]')
    st.write("##### For explanation of LRP, click [here](LINK)")

   # Displaying global and local explanations
    col1, col2 = st.columns([6, 4])

    with col1:
        st.write("**Global Explanation**")
        st.write("The contributions of variables for the model in general. This explanation is the same for all examples.")
        st.image(Image.open(path_global_explanation))


    st.write("**Local Explanation**")
    st.write("The contributions of variables for this specific patient. This explanation is specific to the patient.")
    
    col1, col2, col3 = st.columns([6, 1, 3])

    with col1:
        image_path = path_lrp_local + st.session_state.image_names[st.session_state.id]
        st.image(Image.open(image_path))
    with col3:
        # Load data and display
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
        html_content = styled_df.to_html(escape=False)


        # Define your desired table dimensions
        width = "800px"
        height = "400px"

        # Create a wrapper for the table with specific styles
        table_wrapper = f""" <div style="width: {width}; height: {height}; overflow: auto;"> {html_content} </div>"""

        # Pass the wrapper to Streamlit
        st.markdown(table_wrapper, unsafe_allow_html=True)

    # Link to the explanation of clinical variables
    st.write("##### For explanation of clinical variables, click [here](LINK)")
    
    # Back button to navigate to the previous page
    st.button("Back", on_click=back_return_t, key="r_back_t")

    return