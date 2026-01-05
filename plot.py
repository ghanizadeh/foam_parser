import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# Page config
# ==============================
st.set_page_config(
    page_title="3-Variable Scatter Plot",
    layout="wide"
)

st.title("📊 Interactive Cross-Plot with Color Scale")

# ==============================
# Upload data
# ==============================
uploaded_file = st.file_uploader(
    "📂 Upload CSV file",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

# ==============================
# Column selection
# ==============================
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

st.sidebar.header("🔧 Plot Settings")

x_col = st.sidebar.selectbox("X-axis", numeric_cols)
y_col = st.sidebar.selectbox("Y-axis", numeric_cols)
c_col = st.sidebar.selectbox("Color-coded variable", numeric_cols)

log_color = st.sidebar.checkbox("Apply log10 to color variable")

# ==============================
# Data preparation
# ==============================
x = df[x_col]
y = df[y_col]

if log_color:
    c = np.log10(df[c_col])
    c_label = f"log10({c_col})"
else:
    c = df[c_col]
    c_label = c_col

# ==============================
# Plot
# ==============================
fig, ax = plt.subplots(figsize=(8, 6))

sc = ax.scatter(
    x,
    y,
    c=c,
    cmap="viridis",
    edgecolor="k",
    alpha=0.85
)

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label(c_label)

ax.set_xlabel(x_col)
ax.set_ylabel(y_col)

ax.set_title(
    f"{y_col} vs {x_col}\n(Color-coded by {c_label})",
    fontsize=12
)

ax.grid(True, linestyle="--", alpha=0.4)

st.pyplot(fig)
