import streamlit as st 


def go_home():
    """
    Navigate to the application's home page by resetting certain session state values.
    """
    # Reset the current ID and page number to their initial values
    st.session_state.id = 0
    st.session_state.page_no = 0
    # No explicit return is needed; the function will return None by default

def final_page_u():
    """
    Display the final page after the trial is completed and save all the results to the database.
    """
    # Display completion messages to the user
    _display_completion_messages()

    # Save usability prediction results to the database
    _save_to_db(
        data=st.session_state.usability_pred,
        document_name="usability_prediction_results"
    )

    # Save usability questionnaire results to the database
    _save_to_db(
        data=st.session_state.usability_questionaire,
        document_name="usability_trial_questionaire"
    )

    # Save causability questionnaire results to the database
    _save_to_db(
        data=st.session_state.causability_questionaire,
        document_name="causability_questionaire",
        use_sequential_keys=True
    )

    # Save time taken for the trial to the database
    _save_to_db(
        data=st.session_state.time_taken,
        document_name="usability_trial_time"
    )

    # Save profile information to the database
    _save_profile_to_db()

    # Display the home button
    _display_home_button()

def _display_completion_messages():
    """
    Display completion and thank you messages to the user.
    """
    messages = [
        "You finished the Trial!",
        "Thank you for the participation!",
        "Press the home button to go the home page."
    ]
    for message in messages:
        st.markdown(
            f"<h1 style='text-align: center;'>{message}</h1>",
            unsafe_allow_html=True
        )

def _save_to_db(data, document_name, use_sequential_keys=False):
    """
    Save the given data to the database under the specified document name.
    """
    keys = list(data.keys())
    values = list(data.values())

    if use_sequential_keys:
        keys = [str(x) for x in range(len(values))]

    data_dict = {str(k): str(v) for k, v in zip(keys, values)}
    doc_ref = st.session_state["db"].collection(st.session_state["name_user"]).document(document_name)
    doc_ref.set(data_dict)

def _save_profile_to_db():
    """
    Save user profile data to the database.
    """
    values = [
        st.session_state.name_user,
        st.session_state.nationality_user,
        st.session_state.hospital_user,
        st.session_state.department_user, 
        st.session_state.years_of_experience_user,
        st.session_state.speciality_user
    ]
    keys = [
        "user_name",
        "nationality_user",
        "hospital_user",
        "department_user",
        "years of experience user",
        "user speciality"
    ]
    
    profile_dict = {str(k): str(v) for k, v in zip(keys, values)}
    doc_ref = st.session_state["db"].collection(st.session_state["name_user"]).document("profile")
    doc_ref.set(profile_dict)

def _display_home_button():
    """
    Display a button to navigate to the home page.
    """
    _, _, _, col4, _, _ = st.columns([1, 1, 1, 1, 1, 1])
    with col4:
        button_home = st.button("Home", on_click=go_home, key="go_h")