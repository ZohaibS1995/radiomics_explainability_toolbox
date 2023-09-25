import streamlit as st 


def go_home():
    """
    Navigate to the application's home page by resetting certain session state values.
    """
    # Reset the current ID and page number to their initial values
    st.session_state.id = 0
    st.session_state.page_no = 0
    # No explicit return is needed; the function will return None by default

def _save_to_db_q_ai_exp(data, document_name):
    """
    Save the given data to the database under the specified document name.
    """

    data_dict = {"response_questionnaire_ai_exp": str(data)}
    doc_ref = st.session_state["db"].collection(st.session_state["name_user"]).document(document_name)
    doc_ref.set(data_dict)

def final_page_ex():
    """
    Displays the final page after the user completes the AI experiment trial.
    """
    # Display conclusion messages
    st.markdown(
        """
        <h1 style='text-align: center;'>
            You finished the Trial!
            Press the home button to go the home page.
        </h1>
        <h1 style='text-align: center;'>
            Thank you for the participation!
        </h1>
        <h1 style='text-align: center;'>
            Press the home button to go the home page.
        </h1>
        """,
        unsafe_allow_html=True
    )
    
    # Save AI Exp Prediction results to database
    dict_ai_exp_trial = {str(key): str(val) for key, val in zip(st.session_state.image_names, st.session_state.ai_exp_trial.values())}
    doc_ref = st.session_state["db"].collection(st.session_state["name_user"]).document("ai_exp_trial_results")
    doc_ref.set(dict_ai_exp_trial)

    # Save time taken data to database
    dict_time = {str(key): str(val) for key, val in st.session_state.time_taken.items()}
    doc_ref = st.session_state["db"].collection(st.session_state["name_user"]).document("ai_exp_trial_time")
    doc_ref.set(dict_time)

    # Save user profile information to database
    profile_values = [
        st.session_state.name_user,
        st.session_state.nationality_user,
        st.session_state.hospital_user,
        st.session_state.department_user, 
        st.session_state.years_of_experience_user,
        st.session_state.speciality_user
    ]
    profile_keys = [
        "user_name",
        "nationality_user",
        "hospital_user",
        "department_user",
        "years of experience user",
        "user speciality"
    ]
    dict_profile = {str(key): str(val) for key, val in zip(profile_keys, profile_values)}
    doc_ref = st.session_state["db"].collection(st.session_state["name_user"]).document("profile")
    doc_ref.set(dict_profile)

    _save_to_db_q_ai_exp(
        data=st.session_state.end_questionnaire_ai_exp,
        document_name="end_questionnaire_ai_exp"
    )    

    # Display Home button for navigation
    _, _, _, col4, _, _ = st.columns([1,1,1,1,1,1])  # Using unpacking for readability
    with col4:
        button_home = st.button("Home", on_click=go_home, key="go_h")