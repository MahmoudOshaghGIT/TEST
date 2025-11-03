import streamlit as st
import pandas as pd
import urllib

from io import BytesIO
import cv2
import math
import numpy as np
from PIL import Image

st.set_page_config(layout="wide")

# ---- Image Loader ----
def load_gallery_image_scaled(image_id):
    try:
        url = f"https://m.atcdn.co.uk/a/media/w1024/{image_id}.jpg"
        image_bytes = urllib.request.urlopen(url).read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        image = image.resize((600, 400))  # resize like your original
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except:
        placeholder = Image.new("RGB", (600, 400), color=(255, 0, 0))
        return cv2.cvtColor(np.array(placeholder), cv2.COLOR_RGB2BGR)

# ---- Display UI ----
def display_images(df, max_columns=3):
    decisions = {}
    num_images = len(df)
    num_columns = min(max_columns, num_images)
    num_rows = math.ceil(num_images / num_columns)

    for row in range(num_rows):
        cols = st.columns(num_columns)
        for col_idx in range(num_columns):
            idx = row * num_columns + col_idx
            if idx >= num_images:
                break
            
            row_data = df.iloc[idx]

            image_id = row_data["imageId"]
            make = row_data["vehicle_standard_make"]
            model = row_data["vehicle_standard_model"]
            derivative_id = row_data["derivative_id"]

            image = load_gallery_image_scaled(image_id)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st_image = Image.fromarray(rgb_image)

            with cols[col_idx]:
                st.image(st_image, use_container_width=True)
                st.caption(f"{make} {model}\nDerivative: {derivative_id}")

                decision = st.radio(
                    f"Action for {image_id}",
                    ('Approve', 'Reject'),
                    key=f"radio_{image_id}"
                )
                decisions[image_id] = decision

    return decisions

# ---- App ----
st.title("Vehicle Image Approval Tool")

df = pd.read_csv("vehicle_image_map.csv")  # your new CSV

max_columns = st.sidebar.slider("Columns", 1, 10, 5)

decisions = display_images(df, max_columns=max_columns)

# ---- Save ----
def save_decisions(decisions, file="decisions.csv"):
    out = pd.DataFrame(list(decisions.items()), columns=["imageId", "decision"])
    out.to_csv(file, index=False)
    st.success(f"Saved to {file}")

if st.button("Save Decisions"):
    save_decisions(decisions)

    with open("decisions.csv", "r") as f:
        st.download_button("Download Decisions File", f, "decisions.csv", "text/csv")
