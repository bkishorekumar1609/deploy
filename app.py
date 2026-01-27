import streamlit as st
import pandas as pd

# Main project title
st.title("🛒 Supermarket Analytics App")

# File uploader
file = st.file_uploader("Upload your sales data (CSV)", type=["csv"])

if file is not None:
    df = pd.read_csv(file)
    st.success("File uploaded successfully!")

    st.subheader("🔍 Data Preview")
    st.write(df.head())


# Second project title (below uploader)
st.header("📊 Automated Sales & Inventory Analysis Project")

# File uploader
file = st.file_uploader("Upload file you want (CSV)", type=["csv"])