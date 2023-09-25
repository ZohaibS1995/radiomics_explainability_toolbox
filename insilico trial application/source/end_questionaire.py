import streamlit as st

def plus_one_quest_ai():
    st.session_state.page_no = 8

def plus_one_quest_ai_exp():
    st.session_state.page_no = 9

def end_questionaire_ai_only():

    st.session_state.page_no = 11

    # List of variables
    variables = [
        "Shear wave elastography (SWE) image",
        "Albumin",
        "Total bilirubin",
        "Gamma-glutamyl transferase",
        "Prothrombin time",
        "International normalized ratio",
        "Albumin-Bilirubin",
        "Child-Pugh score",
        "Child-Pugh grade",
        "Model for end-stage liver disease",
        "Major hepatectomy",
        "Splenomegaly",
        "Cirrhosis",
        "Clinically significant portal hypertension",
        "Ascites",
        "Tumor size",
        "Milan criteria",
        "BCLC",
        "Total liver volume",
        "Resected liver volume",
        "Future liver remnant volume",
        "Future liver remnant volume ratio"
    ]

    # Streamlit multi-select widget
    st.session_state.end_questionnaire_ai = st.multiselect(
        "Which variables do you focus on when making the prediction? (You can choose one or more options)", 
        variables
    )

    col1, col2, col3 = st.columns([1, 7, 1])

    # "Next" button
    with col3:
        button_next = st.button("Next", on_click=plus_one_quest_ai, key="add_one_ques_ai")

def end_questionaire_ai_exp():

    st.session_state.page_no = 12

    # List of variables
    variables = [
        "Shear wave elastography (SWE) image",
        "Albumin",
        "Total bilirubin",
        "Gamma-glutamyl transferase",
        "Prothrombin time",
        "International normalized ratio",
        "Albumin-Bilirubin",
        "Child-Pugh score",
        "Child-Pugh grade",
        "Model for end-stage liver disease",
        "Major hepatectomy",
        "Splenomegaly",
        "Cirrhosis",
        "Clinically significant portal hypertension",
        "Ascites",
        "Tumor size",
        "Milan criteria",
        "BCLC",
        "Total liver volume",
        "Resected liver volume",
        "Future liver remnant volume",
        "Future liver remnant volume ratio"
    ]

    # Streamlit multi-select widget
    st.session_state.end_questionnaire_ai_exp = st.multiselect(
        "Which variables do you focus on when making the prediction? (You can choose one or more options)", 
        variables
    )

    col1, col2, col3 = st.columns([1, 7, 1])

    # "Next" button
    with col3:
        button_next = st.button("Next", on_click=plus_one_quest_ai_exp, key="add_one_ques_ai_exp")