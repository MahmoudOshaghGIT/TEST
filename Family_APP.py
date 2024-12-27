import streamlit as st
import pandas as pd
import urllib
import cv2
import math
import numpy as np
from PIL import Image

st.set_page_config(layout="wide")

# Function to load and scale down gallery images
def load_gallery_image_scaled(ref):
    try:
        # Load image bytes from URL
        image_bytes = urllib.request.urlopen(f"https://m.atcdn.co.uk/a/media/w1024/{ref}.jpg").read()
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        # Resize image to 600x400
        image = cv2.resize(image, (600, 400))
        return image
    except Exception as e:
        # Log the error for debugging
        print(f"Failed to load image for ref {ref}: {e}")
        # Return error placeholder image if loading fails
        placeholder_image = Image.new("RGB", (600, 400), color=(255, 0, 0))
        return cv2.cvtColor(np.array(placeholder_image), cv2.COLOR_RGB2BGR)

# Streamlit UI for displaying images with Approve/Reject options
def display_images_with_actions(
    metadata_search_id,
    df,
    image_column='imageId',
    make_column='make',
    model_column='model',
    max_columns=3
):
    # Create a dictionary to store the decisions
    if "decisions" not in st.session_state:
        st.session_state.decisions = {}

    # Calculate number of rows and columns
    num_images = len(metadata_search_id)
    num_columns = min(max_columns, num_images)
    num_rows = math.ceil(num_images / num_columns)

    # Create layout using Streamlit columns
    for row in range(num_rows):
        cols = st.columns(num_columns)
        for col_idx in range(num_columns):
            idx = row * num_columns + col_idx
            if idx >= num_images:
                break
            
            ref = metadata_search_id[idx]
            try:
                # Get the image ID, make, and model from the DataFrame
                image_id = df.loc[df['metadata_search_id'] == ref, image_column].iloc[0]
                make_value = df.loc[df['metadata_search_id'] == ref, make_column].iloc[0]
                model_value = df.loc[df['metadata_search_id'] == ref, model_column].iloc[0]
                
                # Load and scale down the image
                image = load_gallery_image_scaled(image_id)
                
                # Convert the OpenCV image to a format suitable for Streamlit
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st_image = Image.fromarray(rgb_image)
                
                # Display the image and metadata in the column
                with cols[col_idx]:
                    st.image(st_image, use_container_width=True)
                    st.caption(f"{ref}\n{make_value} {model_value}")
                    # Add Approve/Reject radio buttons
                    decision = st.radio(
                        f"Action for {ref}",
                        ('Approve', 'Reject'),
                        key=f"radio_{ref}"
                    )
                    st.session_state.decisions[ref] = decision  # Save the decision for each image
            except Exception as e:
                # Handle errors
                with cols[col_idx]:
                    st.error(f"Error loading image for ref {ref}: {e}")

    return st.session_state.decisions  # Return the decisions dictionary

# Streamlit App
st.title("OMG Category Approve/Reject")

# Load `df_final_omg` and `omg_list` locally
df_final_omg = pd.read_csv("df_final_family.csv")  # Replace with the actual path to your DataFrame CSV file
omg_list = df_final_omg['metadata_search_id'].tolist()  # Example: Get the list of public references

# User input for customizing the gallery
max_columns = st.sidebar.slider("Max Columns", min_value=1, max_value=10, value=3)

# Display the gallery with Approve/Reject options
decisions = display_images_with_actions(omg_list, df_final_omg, max_columns=max_columns)

# Function to save decisions to CSV
def save_decisions_to_file(decisions, file_path="decisions.csv"):
    # Convert decisions dictionary to DataFrame
    decisions_df = pd.DataFrame(list(decisions.items()), columns=["metadata_search_id", "decision"])
    # Save the DataFrame as a CSV
    decisions_df.to_csv(file_path, index=False)
    st.success(f"Decisions saved to {file_path}")

# Save the file when the button is clicked
if st.button("Save Decisions to File"):
    save_decisions_to_file(decisions)

    # Allow the user to download the file
    with open("decisions.csv", "r") as f:
        st.download_button(
            label="Download Decisions File",
            data=f,
            file_name="decisions.csv",
            mime="text/csv"
        )