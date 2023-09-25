import streamlit as st
from datetime import datetime 

def initialize_trial(page_no):
    """
    Initializes the trial by recording the start time and setting the session's page number.
    """
    st.session_state.start_time = datetime.now() 
    st.session_state.page_no = page_no

def sel_usability():
    """ Handles the usability test button click. """
    initialize_trial(1)

def sel_ai_trial():
    """ Handles the AI trial button click. """
    initialize_trial(2)

def sel_ai_exp_trial():
    """ Handles the AI + explanation trial button click. """
    initialize_trial(3)

def landing_page():
    """ Renders the landing page with title, description, and buttons for different trials. """
    
    # Displaying title
    st.markdown("<h1 style='text-align: center;'>"
                "Post-Hepatectomy Liver Failure Prediction Based on 2D-SWE images and Clinical Variables with"
                " an Interpretable Deep Learning Framework"
                "</h1>", unsafe_allow_html=True)

    # Information on timer start
    st.markdown("<h4 style='text-align: center;color: red;'>"
                    "The Timer will start when you click the button."
                    "</h4>", unsafe_allow_html=True)

    # Styling for the buttons
    st.markdown("""
        <style>
            div[class*="stButton"] > label > div[data-testid="stMarkdownContainer"] > p {
                font-size: 32px;
            }
            div.stButton > button:first-child {
                font-size:28px !important;
                margin: auto;
                display: block;
                height: 2em; 
                width: 20em;
            }
            div[class*="stButton"] > button > div[data-testid="stMarkdownContainer"] > p {
                font-size: 28px;
                margin: auto;
                display: block;
                height: 2em; 
                width: 20em;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Creating buttons for different trials
    st.button("Usability test", on_click=sel_usability, key="s_u")
    st.button("Clinical trial: only AI", on_click=sel_ai_trial, key="s_a")
    st.button("Clinical trial: AI+explanation", on_click=sel_ai_exp_trial, key="s_ex_a")

    # Setting up columns
    col1, col2, col3 = st.columns([2, 4, 2])

    with col2:   
        # Displaying video instructions
        with open('./video_instructions.mp4', 'rb') as video_file:
            video_bytes = video_file.read()

        st.markdown("<h2 style='text-align: center;'>"
                    "Please go through the video before you begin the trial"
                    "</h2>", unsafe_allow_html=True)
        st.video(video_bytes)

    return