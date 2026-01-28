import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

st.title("🔧 Machine Quality Comparison Web App")

file = st.file_uploader("Upload CSV file", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("Data Preview")
    st.dataframe(df)

    machines = df['Machine'].unique()

    st.subheader("Descriptive Statistics")
    stats_df = df.groupby("Machine")['Measurement'].agg(['mean', 'std', 'count'])
    st.write(stats_df)

    st.subheader("One-Way ANOVA")
    data = [df[df['Machine']==m]['Measurement'] for m in machines]
    f, p = stats.f_oneway(*data)

    st.write("F-statistic:", f)
    st.write("P-value:", p)

    if p < 0.05:
        st.error("Significant difference between machines")
    else:
        st.success("No significant difference between machines")

    st.subheader("Control Chart (X̄)")
    for m in machines:
        values = df[df['Machine']==m]['Measurement']
        mean = values.mean()
        std = values.std()

        plt.figure()
        plt.plot(values.values, marker='o')
        plt.axhline(mean)
        plt.axhline(mean + 3*std)
        plt.axhline(mean - 3*std)
        plt.title(f"Control Chart - Machine {m}")
        st.pyplot(plt)