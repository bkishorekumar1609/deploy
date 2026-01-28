import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
from scipy import stats
import matplotlib.pyplot as plt


st.set_page_config(page_title="Machine Quality Control App", layout="wide")
st.title("KA 🔧 Machine Quality Analysis Web App")

# -------------------------------
# USER INPUT: FIXED MEASUREMENT
# -------------------------------
st.subheader("⚙ Enter Quality Specifications")

col1, col2 = st.columns(2)
with col1:
    target = st.number_input("Target Measurement (Fixed Value)", min_value=0.0, value=50.0)
with col2:
    tolerance = st.number_input("Tolerance (±)", min_value=0.0, value=0.5)

LSL = target - tolerance
USL = target + tolerance

st.info(f"📌 Specification Limits → LSL: {LSL} | USL: {USL}")

# -------------------------------
# Upload Dataset
# -------------------------------
file = st.file_uploader("📂 Upload Machine Dataset (CSV)", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df)

    machines = df['Machine'].unique()

    # -------------------------------
    # Descriptive Statistics
    # -------------------------------
    st.subheader("📊 Descriptive Statistics")
    desc = df.groupby("Machine")['Measurement'].agg(['mean', 'std', 'count'])
    st.dataframe(desc)

    # -------------------------------
    # ANOVA
    # -------------------------------
    st.subheader("📈 One-Way ANOVA Result")

    data = [df[df['Machine'] == m]['Measurement'] for m in machines]
    f_stat, p_value = stats.f_oneway(*data)

    col1, col2 = st.columns(2)
    col1.metric("F-Statistic", round(f_stat, 4))
    col2.metric("P-Value", round(p_value, 4))

    if p_value < 0.05:
        st.error("❌ Significant difference between machines")
    else:
        st.success("✅ No significant difference between machines")

    # -------------------------------
    # Control Charts + Spec Limits
    # -------------------------------
    st.subheader("📉 Control Charts with User-Defined Limits")

    results = []

    for m in machines:
        values = df[df['Machine'] == m]['Measurement']
        mean = values.mean()
        std = values.std()

        UCL = mean + 3 * std
        LCL = mean - 3 * std

        spec_violations = ((values < LSL) | (values > USL)).sum()
        control_violations = ((values < LCL) | (values > UCL)).sum()

        results.append({
            "Machine": m,
            "Mean": round(mean, 3),
            "Std Dev": round(std, 3),
            "Spec Violations": spec_violations,
            "Control Violations": control_violations
        })

        plt.figure()
        plt.plot(values.values, marker='o', label="Measurement")
        plt.axhline(mean, linestyle='--', label='Process Mean')
        plt.axhline(UCL, linestyle='--', label='UCL')
        plt.axhline(LCL, linestyle='--', label='LCL')
        plt.axhline(USL, linestyle=':', label='USL (Spec)')
        plt.axhline(LSL, linestyle=':', label='LSL (Spec)')
        plt.title(f"Control Chart - Machine {m}")
        plt.legend()
        st.pyplot(plt)

    # -------------------------------
    # GOOD vs BAD MACHINE
    # -------------------------------
    st.subheader("✅ Machine Quality Decision")

    result_df = pd.DataFrame(results)

    result_df["Status"] = np.where(
        (result_df["Spec Violations"] == 0) & (result_df["Control Violations"] == 0),
        "GOOD",
        "BAD"
    )

    st.dataframe(result_df)

    st.success("✔ Good Machines: " + ", ".join(result_df[result_df["Status"]=="GOOD"]["Machine"]))
    st.error("❌ Bad Machines: " + ", ".join(result_df[result_df["Status"]=="BAD"]["Machine"]))

    # -------------------------------
    # Prediction
    # -------------------------------
    st.subheader("🔮 Prediction (Next Output)")

    selected_machine = st.selectbox("Select Machine for Prediction", machines)

    mean_val = desc.loc[selected_machine, 'mean']
    std_val = desc.loc[selected_machine, 'std']

    predicted = np.random.normal(mean_val, std_val)
    status = "OK" if LSL <= predicted <= USL else "OUT OF SPEC"

    st.info(f"📌 Predicted Value: **{round(predicted, 3)}**")
    st.write(f"📊 Prediction Status: **{status}**")
