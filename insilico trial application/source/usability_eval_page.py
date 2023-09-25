import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime 

import streamlit as st
from config import *
from source.utils import *

def pred_mod():
    """Handles the event when the 'Model Prediction Results of PHLF Risk' button is clicked."""
    st.session_state.pred_mod_page1 = True


def explainability_disp():
    """Handles the event when the 'Explainability' button is clicked."""
    st.session_state.page_no = 4


def question_page():
    """Handles the event when the 'Next' button is clicked."""
    st.session_state.page_no = 5
    

# Increment user state and manage timing
def plus_one_u():
    """Handles the event when the 'Next' button is clicked."""

    # Store the time taken for the current user ID
    st.session_state.time_taken["{:03d}".format(int(st.session_state.u_name[st.session_state.id])) + ".png"] = datetime.now() - st.session_state.start_time

    # Reset the start time for the next user ID
    st.session_state.start_time = datetime.now()

    # Initialize page number
    st.session_state.page_no = 1

    # Increment user ID if within bounds
    if st.session_state.id < 6:  # Consider using len(st.session_state.image_names) - 1 for dynamic checking
        st.session_state.id += 1
        st.session_state.pred_mod_page1 = False

    # Check if last ID reached to set page number
    if st.session_state.id == 6:
        st.session_state.page_no = 10

    return

def plus_one_cu():
    """
    This function is used as a callback, for taking the application view to a new use case.
    """
    st.session_state.page_no = 7
    
    return


def usability_page():
    """Renders the usability page."""
    
    # Styling
    st.markdown("""<style>
        div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
            font-size: 24px;
        }
        div[class*="stRadio"] {
            background-color: #FEA09A;
            border: 3px solid;
        }
    </style>""", unsafe_allow_html=True)

    st.session_state.page_no = 1
    st.write(f"## Usability Trial: {st.session_state.id+1}/6")
    st.write("---")

    # Rendering columns
    col1, col2 = st.columns([4, 5])

    with col1:
        st.write("**Top: SWE Image, Bottom: B-mode Ultrasound**")
        image_path = f"{path_orig_images}{int(st.session_state.u_name[st.session_state.id]):03}.png"
        image = Image.open(image_path)
        st.image(image)

    with col2:
        # Load data and display
        df = pd.read_csv(path_clinical_data)
        # Iterate over columns and convert if all values are floats
        for col in df.columns:
            # Check if all values in the column can be converted to float
            if df[col].apply(is_float).all():
                df[col] = df[col].astype(float).round(2)
                
        keys_n = list(df.columns)[2:]
        val_w_k = list(df.iloc[st.session_state.u_id[st.session_state.id], 2:].to_numpy())

        normal_range = list(pd.read_csv(path_clinical_normal).to_numpy().transpose())

        data = np.vstack((keys_n, val_w_k, normal_range)).transpose()
        v_df = pd.DataFrame(data, columns=["Clinical Variable", "Value", "Normal Range"])
        hyperlinks = list(pd.read_csv(path_clinical_hyperlinks, delimiter=";")["Hyperlinks"].to_numpy().transpose())
        v_df['Clinical Variable'] = [apply_hyperlink(item, link) for item, link in zip(v_df['Clinical Variable'], hyperlinks)]

    
        styled_df = v_df.style.apply(color_rows, axis=1)
        st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)

        st.write(f"##### For explanation of clinical variables, [click here]({URL_variable_explanation})")

    # Load predictions
    df_pred = pd.read_csv(path_model_prediction, delimiter=";")
    val_dict = {df_pred["ID"][i]: [df_pred["prediction"][i], df_pred["probability"][i]] for i in range(len(df_pred["ID"]))}

    if not st.session_state.pred_mod_page1:
        st.markdown('<p class="big-font"> Press the button to reveal the model prediction</p>', unsafe_allow_html=True)

    st.button("1- Model Prediction Results of PHLF Risk", on_click=pred_mod, key="disp_pred_mod")

    if st.session_state.pred_mod_page1:
        val_w_k = val_dict[int(st.session_state.u_name[st.session_state.id])]
        pred_text = f"#### The AI model prediction is :red[{val_w_k[0]}] with a probability of :red[{np.round(float(val_w_k[1].replace(',', '.')), 3)}]"
        st.markdown(pred_text)
        st.write("The cut-off value for PHLF prediction is set at 0.35 based on Youden's index.")

    explainability_page = st.button("2- Explainability", on_click=explainability_disp, key="exp_page")

    col1, col2, col3 = st.columns([4, 4, 4])
    with col1:
        st.radio("3- Select your prediction of PHLF Risk", ("High risk of PHLF", "Low risk of PHLF"), key="u_pred")

    st.session_state.usability_pred[f"{int(st.session_state.u_name[st.session_state.id]):03}.png"] = st.session_state.u_pred

    col1, col2, col3 = st.columns([1, 7, 1])
    with col3:
        button_next = st.button("Next", on_click=question_page, key="q_page")

    return



def usability_questionaire():
    """
    Displays a usability questionnaire on counterfactual explanations and LayerWise Relevance Propagation.
    This function uses Streamlit to generate a series of radio buttons that gather user feedback 
    regarding AI explanations.
    """
    
    # Initialize the title and set the current page number
    st.title("")
    st.session_state.page_no = 5
    
    # Start of Counterfactual Explanation questionnaire
    st.markdown("### Questionnaire: Counterfactual Explanation")
    
    # Injecting custom styles for radio buttons
    st.markdown("""
        <style>
        .radio-group > * {
            display: inline-block;
            margin-right: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Rating Scale description
    st.markdown('**1. Strongly disagree 2. Disagree 3. Neither agree nor disagree 4. Agree 5. Strongly agree**')
    
    # Questions for Counterfactual Explanation
    _generate_question_block("1. Understandability:", "I understand how the AI system made the above assessment for PHLF", "radio1")
    _generate_question_block("2. Classifier’s decision justification:", "The changes in the SWE image are related to PHLF", "radio2")
    _generate_question_block("3. Visual quality:", "The generated counterfactual images look like SWE images", "radio3")
    _generate_question_block("4. Helpfulness:", "The explanation helped me understand the assessment made by the AI system", "radio4")
    _generate_question_block("5. Confidence:", "I feel more confident on the model with the explanation", "radio5")
    
    # Start of LayerWise Relevance Propagation questionnaire
    st.markdown("### Questionnaire: LayerWise Relevance Propagation")
    
    # Re-using the same Rating Scale description (consider making this a function if reused multiple times)
    st.markdown('**1. Strongly disagree 2. Disagree 3. Neither agree nor disagree 4. Agree 5. Strongly agree**')
    
    # Questions for LayerWise Relevance Propagation
    _generate_question_block("1. Understandability:", "I understand which features influence the prediction and how they influence", "radiol1")
    _generate_question_block("2. Classifier’s decision justification:", "The feature's contribution are reasonably related to PHLF", "radiol2")
    _generate_question_block("3. Helpfulness:", "The explanation helped me understand the assessment made by the AI system", "radiol3")
    _generate_question_block("4. Confidence:", "I feel more confident on the model with the explanation", "radiol4")
    
    # Layout columns and save responses
    col1, col2, col3 = st.columns([1, 7, 1])
    
    # Store user responses in the session state
    st.session_state.usability_questionaire[st.session_state.image_names[st.session_state.id]] = [
        st.session_state.radio1, st.session_state.radio2, st.session_state.radio3, 
        st.session_state.radio4, st.session_state.radio5,
        st.session_state.radiol1, st.session_state.radiol2, st.session_state.radiol3, 
        st.session_state.radiol4
    ]

    # "Next" button
    with col3:
        button_next = st.button("Next", on_click=plus_one_u, key="add_one_u")

    return 

def _generate_question_block(title, question, key):
    """
    Helper function to generate a block of questionnaire.
    Each block contains a markdown title and a radio button group for the user to provide feedback.
    """
    with st.container():
        st.markdown(f"**{title}**")
        st.radio(question, ("1", "2", "3", "4", "5"), key=key, horizontal=True)

def system_causability_scale():
    # Title of the page
    st.title("")
    
    # Set the page number in the session state
    st.session_state.page_no = 10

    # Displaying a questionnaire title
    st.markdown("## Questionnaire about the Interpretability System")

    # Inserting styling for radio button groups
    st.markdown(
        """
        <style>
        .radio-group > * {
            display: inline-block;
            margin-right: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Instructions for the questionnaire
    st.markdown('**1. Strongly disagree 2. Disagree 3. Neither agree nor disagree 4. Agree 5. Strongly agree**')
    
    # List of questions to display in the questionnaire
    questions = [
        "1) I found that the data included all relevant known causal factors with sufficient precision and granularity",
        "2) I understood the explanations within the context of my work.",
        "3) I could change the level of detail on demand.",
        "4) I did not need support to understand the explanations.",
        "5) I found the explanations helped me to understand causality",
        "6) I was able to use the explanations with my knowledge base.",
        "7) I did not find inconsistencies between explanations",
        "8) I think that most people would learn to understand the explanations very quickly",
        "9) I did not need more references in the explanations: e.g., medical guidelines, regulations.",
        "10) I received the explanations in a timely and efficient manner."
    ]

    # Display each question with its radio button options
    for idx, question in enumerate(questions, 1):
        key = f"radio{idx}"
        with st.container():
            st.radio(question, ("1", "2", "3", "4", "5"), key=key, horizontal=True)

    # Define column layout
    col1, col2, col3 = st.columns([1, 7, 1])

    # Update session state with user's responses to the questionnaire
    st.session_state.causability_questionaire[st.session_state.image_names[st.session_state.id]] = [
        st.session_state.radio1, st.session_state.radio2, st.session_state.radio3, 
        st.session_state.radio4, st.session_state.radio5,
        st.session_state.radiol1, st.session_state.radiol2, st.session_state.radiol3, 
        st.session_state.radiol4, st.session_state.radiol5
    ]

    # Display "Next" button
    with col3:
        button_next = st.button("Next", on_click=plus_one_cu, key="add_one_cu")

    return
