import streamlit as st 


def go_home():
    """
    Navigate to the application's home page by resetting certain session state values.
    """
    # Reset the current ID and page number to their initial values
    st.session_state.id = 0
    st.session_state.page_no = 0
    # No explicit return is needed; the function will return None by default

def _save_to_db_q_ai(data, document_name):
    """
    Save the given data to the database under the specified document name.
    """

    data_dict = {"response_questionnaire_ai_only": str(data)}
    doc_ref = st.session_state["db"].collection(st.session_state["name_user"]).document(document_name)
    doc_ref.set(data_dict)

def final_page_a():
    # Display completion messages
    st.markdown("<h1 style='text-align: center;'>"
                "You finished the Trial! Press the home button to go the home page."
                "</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>"
                "Thank you for the participation!"
                "</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>"
                "Press the home button to go the home page."
                "</h1>", unsafe_allow_html=True)
    
    # Store AI Prediction
    keys_t = list(st.session_state.image_names)
    values_t = list(st.session_state.ai_trial.values())
    dict_ai_trial_pred = {str(key): str(value) for key, value in zip(keys_t, values_t)}
    doc_ref = st.session_state["db"].collection(st.session_state["name_user"]).document("ai_trial_results")
    doc_ref.set(dict_ai_trial_pred)

    # Store Time taken data
    keys_t = list(st.session_state.time_taken.keys())
    values_t = list(st.session_state.time_taken.values())
    dict_time = {str(key): str(value) for key, value in zip(keys_t, values_t)}
    doc_ref = st.session_state["db"].collection(st.session_state["name_user"]).document("ai_trial_time")
    doc_ref.set(dict_time)

    # Saving profile information
    keys_t = [
        "user_name",
        "nationality_user",
        "hospital_user",
        "department_user",
        "years of experience user",
        "user speciality"
    ]
    values_t = [
        st.session_state.name_user,
        st.session_state.nationality_user,
        st.session_state.hospital_user,
        st.session_state.department_user, 
        st.session_state.years_of_experience_user,
        st.session_state.speciality_user
    ]
    dict_profile = {str(key): str(value) for key, value in zip(keys_t, values_t)}
    doc_ref = st.session_state["db"].collection(st.session_state["name_user"]).document("profile")
    doc_ref.set(dict_profile)

    _save_to_db_q_ai(
        data=st.session_state.end_questionnaire_ai,
        document_name="end_questionnaire_ai"
    )    

    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
    with col4:
        button_home = st.button("Home", on_click=go_home, key="go_h")
