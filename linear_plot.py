import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Title
st.title("Interactive Linear Line Plot")

# Sidebar sliders to change slope and intercept
slope = st.sidebar.slider("Slope (m)", min_value=-10.0, max_value=10.0, value=1.0, step=0.1)
intercept = st.sidebar.slider("Intercept (b)", min_value=-20.0, max_value=20.0, value=0.0, step=0.5)

# Generate x and y values
x = np.linspace(-10, 10, 400)
y = slope * x + intercept

# Plot
fig, ax = plt.subplots()
ax.plot(x, y, label=f"y = {slope}x + {intercept}")
ax.axhline(0, color='gray', lw=1)
ax.axvline(0, color='gray', lw=1)
ax.set_title("Line Plot")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid(True)

# Show plot
st.pyplot(fig)
