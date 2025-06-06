import streamlit as st
import pandas as pd
import urllib
import mmcv
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
        image = mmcv.imfrombytes(image_bytes)
        # Rescale image to 600x400
        return mmcv.imrescale(image, (600, 400))
    except Exception as e:
        # Return error placeholder image if loading fails
        placeholder_image = Image.new("RGB", (600, 400), color=(255, 0, 0))
        return cv2.cvtColor(np.array(placeholder_image), cv2.COLOR_RGB2BGR)

# Streamlit UI for displaying images with Approve/Reject options
def display_images_with_actions(
    public_references,
    df,
    image_column='imageId',
    method_column='method',
    make_column='vehicle_standard_make',
    model_column='vehicle_standard_model',
    max_columns=3
):
    # Create a dictionary to store the decisions (initialized outside to retain decisions)
    decisions = {}

    # Calculate number of rows and columns
    num_images = len(public_references)
    num_columns = min(max_columns, num_images)
    num_rows = math.ceil(num_images / num_columns)

    # Create layout using Streamlit columns
    for row in range(num_rows):
        cols = st.columns(num_columns)
        for col_idx in range(num_columns):
            idx = row * num_columns + col_idx
            if idx >= num_images:
                break
            
            ref = public_references[idx]
            try:
                # Get the image ID, method, make, and model from the DataFrame
                image_id = df.loc[df['public_reference'] == ref, image_column].iloc[0]
                method_value = df.loc[df['public_reference'] == ref, method_column].iloc[0]
                make_value = df.loc[df['public_reference'] == ref, make_column].iloc[0]
                model_value = df.loc[df['public_reference'] == ref, model_column].iloc[0]
                
                # Load and scale down the image
                image = load_gallery_image_scaled(image_id)
                
                # Convert the OpenCV image to a format suitable for Streamlit
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st_image = Image.fromarray(rgb_image)
                
                # Display the image and metadata in the column
                with cols[col_idx]:
                    st.image(st_image, use_container_width=True)
                    st.caption(f"{ref}\nMethod: {method_value}\n{make_value} {model_value}")
                    # Add Approve/Reject radio buttons
                    decision = st.radio(
                        f"Action for {ref}",
                        ('Approve', 'Reject'),
                        key=f"radio_{ref}"
                    )
                    decisions[ref] = decision  # Save the decision for each image
            except Exception as e:
                # Handle errors
                with cols[col_idx]:
                    st.error(f"Error: {e}")
                    st.text("Failed to load image.")

    return decisions  # Return the decisions dictionary to save to a file

# Streamlit App
st.title("Luxury Category Approve/Reject")

df_final_sport = pd.read_csv("df_final_luxury.csv")  # Replace with the actual path to your DataFrame CSV file
sport_list = df_final_sport['public_reference'].tolist()  # Example: Get the list of public references

# User input for customizing the gallery
max_columns = st.sidebar.slider("Max Columns", min_value=1, max_value=10, value=5)

# Display the gallery with Approve/Reject options
decisions = display_images_with_actions(sport_list, df_final_sport, max_columns=max_columns)

# Function to save decisions to CSV
def save_decisions_to_file(decisions, file_path="decisions.csv"):
    # Convert decisions dictionary to DataFrame
    decisions_df = pd.DataFrame(list(decisions.items()), columns=["public_reference", "decision"])
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
