import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
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

# Page Config
st.set_page_config(page_title="Supermarket Insights Pro", layout="wide")
st.title("🛒 Supermarket Analytics Dashboard")
st.markdown("Upload your sales data to generate instant business intelligence.")

# 1. File Uploader
uploaded_file = st.sidebar.file_uploader("Upload Sales CSV/Excel", type=['csv', 'xlsx'])

if uploaded_file:
    # Load Data
    df = pd.read_csv(uploaded_file) # Use pd.read_excel for xlsx
    
    # Simple KPI Row
    total_sales = df['Total'].sum()
    total_items = df['Quantity'].sum()
    avg_rating = df['Rating'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"${total_sales:,.2f}")
    col2.metric("Units Sold", f"{total_items:,}")
    col3.metric("Avg Customer Rating", f"{avg_rating:.1f} / 5")

    # 2. Sales Trend Analysis
    st.subheader("📈 Sales Trends")
    df['Date'] = pd.to_datetime(df['Date'])
    daily_sales = df.groupby('Date')['Total'].sum().reset_index()
    fig_line = px.line(daily_sales, x='Date', y='Total', title="Daily Revenue Growth")
    st.plotly_chart(fig_line, use_container_width=True)

    # 3. Product Performance (Fast vs. Slow Moving)
    st.subheader("📦 Inventory Intelligence")
    prod_col1, prod_col2 = st.columns(2)
    
    product_sales = df.groupby('Product line')['Quantity'].sum().sort_values(ascending=False).reset_index()
    
    with prod_col1:
        fig_fast = px.bar(product_sales.head(5), x='Quantity', y='Product line', orientation='h', 
                          title="Fast-Moving Products", color_discrete_sequence=['#00CC96'])
        st.plotly_chart(fig_fast)

    with prod_col2:
        fig_slow = px.bar(product_sales.tail(5), x='Quantity', y='Product line', orientation='h', 
                          title="Slow-Moving Products", color_discrete_sequence=['#EF553B'])
        st.plotly_chart(fig_slow)

else:
    st.info("Please upload a CSV file in the sidebar to begin.")
