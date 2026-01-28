import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

st.set_page_config(page_title="Machine Quality Control App", layout="wide")
st.title("🔧 Machine Quality Analysis Web App")

# Upload CSV
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
    # Control Charts
    # -------------------------------
    st.subheader("📉 Control Charts (X̄ Chart)")

    results = []

    for m in machines:
        values = df[df['Machine'] == m]['Measurement']
        mean = values.mean()
        std = values.std()

        UCL = mean + 3 * std
        LCL = mean - 3 * std

        out_of_control = ((values > UCL) | (values < LCL)).sum()

        # Store result
        results.append({
            "Machine": m,
            "Mean": round(mean, 3),
            "Std Dev": round(std, 3),
            "Out of Control Points": out_of_control
        })

        # Plot
        plt.figure()
        plt.plot(values.values, marker='o')
        plt.axhline(mean, linestyle='--', label='Mean')
        plt.axhline(UCL, linestyle='--', label='UCL')
        plt.axhline(LCL, linestyle='--', label='LCL')
        plt.title(f"Control Chart - Machine {m}")
        plt.legend()
        st.pyplot(plt)

    # -------------------------------
    # Good vs Bad Machine
    # -------------------------------
    st.subheader("✅ Machine Performance Decision")

    result_df = pd.DataFrame(results)

    result_df["Status"] = np.where(
        (result_df["Out of Control Points"] == 0) & (result_df["Std Dev"] == result_df["Std Dev"].min()),
        "GOOD",
        "NEEDS ATTENTION"
    )

    st.dataframe(result_df)

    best_machine = result_df.sort_values("Std Dev").iloc[0]["Machine"]
    worst_machine = result_df.sort_values("Std Dev").iloc[-1]["Machine"]

    st.success(f"🏆 Best Machine: {best_machine}")
    st.error(f"⚠ Worst Machine: {worst_machine}")

    # -------------------------------
    # Prediction
    # -------------------------------
    st.subheader("🔮 Prediction (Next Output Value)")

    selected_machine = st.selectbox("Select Machine for Prediction", machines)

    mean_val = desc.loc[selected_machine, 'mean']
    std_val = desc.loc[selected_machine, 'std']

    prediction = np.random.normal(mean_val, std_val)

    st.info(f"📌 Predicted next output for Machine {selected_machine}: **{round(prediction, 3)}**")