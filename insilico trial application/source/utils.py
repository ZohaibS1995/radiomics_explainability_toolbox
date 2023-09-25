import base64
import pandas as pd
import streamlit as st

def color_rows(row):
    """Applies color to rows based on whether the value is within the normal range."""
    value, normal_range = row["Value"], row["Normal Range"]
    in_range = check_in_range(value, normal_range)
    
    color_style = "color: red"
    
    if in_range is None or in_range:
        return ["", "", ""]
    else:
        # Check if it's a hyperlink
        if '<a' in row["Clinical Variable"]:
            # Applying color to the anchor tag directly
            return [f'a:link, a:visited, a:hover, a:active {{{color_style}}}', color_style, color_style]
        else:
            return [color_style, color_style, color_style]

def check_in_range(value, range_str):
    if range_str == "-":
        return None  # No coloring
    
    elif len(range_str.split("-")) > 1:
        # Handling cases where the range might use the en dash (–) instead of the hyphen (-)
        low = range_str.split("-")[0]
        high = range_str.split("-")[1].split("(")[0].split("s")[0]
    else:
        # It's a single value
        low = high = range_str

    # Check if the value lies within the range
    try:
        value = float(value)
        low, high = float(low), float(high)
        return value >= low and value <= high
    except:
        # Handle non-numeric value comparisons
        return value == low
    

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

# Apply hyperlinks to the first column of DataFrame
def apply_hyperlink(item, link):
    if pd.isna(link):
        return item
    else:
        return f'<a href="{link}" target="_blank">{item}</a>'
    
def _record_time_taken():
    """
    Helper function to record the time taken to evaluate an image.
    """
    elapsed_time = datetime.now() - st.session_state.start_time
    st.session_state.time_taken[st.session_state.image_names[st.session_state.id]] = elapsed_time
    st.session_state.start_time = datetime.now()
    

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
    
# not used    
def minus_one():
    if st.session_state.id > 0:
        st.session_state.id -= 1
        st.session_state.pred_mod_page1 = False
    return

