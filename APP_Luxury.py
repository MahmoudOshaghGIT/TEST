import streamlit as st
import pandas as pd
import urllib
import cv2
import math
import numpy as np
from PIL import Image

st.set_page_config(layout="wide")

# Function to load and scale down gallery images using OpenCV (not mmcv)
def load_gallery_image_scaled(ref, max_width=600, max_height=400):
    try:
        url = f"https://m.atcdn.co.uk/a/media/w1024/{ref}.jpg"
        image_bytes = urllib.request.urlopen(url).read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Get original dimensions
        h, w = image.shape[:2]
        scale = min(max_width / w, max_height / h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        return cv2.resize(image, (new_w, new_h))
    except Exception as e:
        placeholder = Image.new("RGB", (max_width, max_height), color=(255, 0, 0))
        return cv2.cvtColor(np.array(placeholder), cv2.COLOR_RGB2BGR)

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
    decisions = {}
    num_images = len(public_references)
    num_columns = min(max_columns, num_images)
    num_rows = math.ceil(num_images / num_columns)

    for row in range(num_rows):
        cols = st.columns(num_columns)
        for col_idx in range(num_columns):
            idx = row * num_columns + col_idx
            if idx >= num_images:
                break

            ref = public_references[idx]
            try:
                image_id = df.loc[df['public_reference'] == ref, image_column].iloc[0]
                method_value = df.loc[df['public_reference'] == ref, method_column].iloc[0]
                make_value = df.loc[df['public_reference'] == ref, make_column].iloc[0]
                model_value = df.loc[df['public_reference'] == ref, model_column].iloc[0]

                image = load_gallery_image_scaled(image_id)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st_image = Image.fromarray(rgb_image)

                with cols[col_idx]:
                    st.image(st_image, use_container_width=True)
                    st.caption(f"{ref}\nMethod: {method_value}\n{make_value} {model_value}")
                    decision = st.radio(
                        f"Action for {ref}",
                        ('Approve', 'Reject'),
                        key=f"radio_{ref}"
                    )
                    decisions[ref] = decision
            except Exception as e:
                with cols[col_idx]:
                    st.error(f"Error: {e}")
                    st.text("Failed to load image.")

    return decisions

# App title
st.title("Luxury Category Approve/Reject")

# Load the luxury CSV (must be in your repo)
df_final_luxury = pd.read_csv("df_final_luxury.csv")

# Get the list of image references
luxury_list = df_final_luxury['public_reference'].tolist()

# Sidebar for layout control
max_columns = st.sidebar.slider("Max Columns", min_value=1, max_value=10, value=5)

# Display gallery with controls
decisions = display_images_with_actions(
    luxury_list,
    df_final_luxury,
    make_column='vehicle_standard_make',
    model_column='vehicle_standard_model',
    max_columns=max_columns
)

# Save to CSV and enable download
def save_decisions_to_file(decisions, file_path="decisions.csv"):
    df = pd.DataFrame(list(decisions.items()), columns=["public_reference", "decision"])
    df.to_csv(file_path, index=False)
    st.success(f"Decisions saved to {file_path}")

if st.button("Save Decisions to File"):
    save_decisions_to_file(decisions)

    with open("decisions.csv", "r") as f:
        st.download_button(
            label="Download Decisions File",
            data=f,
            file_name="decisions.csv",
            mime="text/csv"
        )
