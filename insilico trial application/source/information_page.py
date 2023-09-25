import streamlit as st


# This button leads to the landing page which shows all the types of available trials
def go_to_landing_page():
    """
    Updates session state based on user inputs from the form in the info page. 
    If all fields are filled out, it sets the state to move to the landing page.
    If not, it updates the state to show a requirement message on the info page.
    """
    # Check if all fields are filled out
    required_fields = ['name', 'nationality', 'hospital', 'department', 'years_of_experience', 'speciality']
    if all(st.session_state.get(field, '') != '' for field in required_fields):
        st.session_state.page_no = 0
        # Copying input data to user specific session state variables
        st.session_state.name_user = st.session_state.name
        st.session_state.nationality_user = st.session_state.nationality
        st.session_state.hospital_user = st.session_state.hospital
        st.session_state.department_user = st.session_state.department
        st.session_state.years_of_experience_user = st.session_state.years_of_experience
        st.session_state.speciality_user = st.session_state.speciality
        st.session_state.required_flag = False
    else:
        st.session_state.page_no = -1
        st.session_state.required_flag = True
    return


def info_page():
    """
    Displays an information page with a form to get user details.
    """
    # Reset to the info page state
    st.session_state.page_no = -1

    # Defining layout columns
    col1, col2, col3 = st.columns([2, 4, 2])

    with col2:
        # Title for the page
        st.markdown("<h1 style='text-align: center;'>"
                    "Post-Hepatectomy Liver Failure Prediction Based on 2D-SWE images and Clinical Variables with"
                    " an Interpretable Deep Learning Framework"
                    "</h1>", unsafe_allow_html=True)

        # Subheader for the trial details
        st.markdown("<h2 style='text-align: center; color: red'>"
                    "In silico Clinical and Usability Trial"
                    "</h2>", unsafe_allow_html=True)
                   
        # User input form
        st.subheader("Enter Details Below")
        with st.form("form1"):
            st.text_input("Full Name", key="name")
            st.text_input("Nationality", key="nationality")
            st.text_input("Hospital", key="hospital")
            st.text_input("Department", key="department")
            st.text_input("Speciality", key="speciality")
            st.text_input("Years of Working Experience in the Speciality", key="years_of_experience")
            st.form_submit_button("Submit", on_click=go_to_landing_page)
    
        # Check and display message if all fields are not filled
        if st.session_state.get('required_flag', False):
            st.write("**:red[Please fill all the fields]**")