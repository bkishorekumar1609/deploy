import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import io
from fpdf import FPDF
from xlsxwriter import Workbook


st.set_page_config(page_title="SK Analytics App", layout="wide")
st.title("🔧 Machine Quality Analysis Web App")

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

# ---------------- Page Config ----------------

st.set_page_config(page_title="Supermarket Analytics Dashboard", layout="wide")
st.title("🛒 Supermarket Analytics Dashboard")

# ---------------- File Upload ----------------
file = st.file_uploader("Upload Supermarket Sales Data (CSV)", type=["csv"])

if file:
    df = pd.read_csv(file)
    
    # Preprocessing (Safeguards added)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Sales'] = df['Quantity'] * df['Price']
    df['Profit'] = df['Sales'] - (df['Quantity'] * df['Cost'])
    # Prevent division by zero errors
    df['Profit_Margin'] = np.where(df['Sales'] != 0, df['Profit'] / df['Sales'], 0)

    # ---------------- Sidebar Filters ----------------
    st.sidebar.header("🔍 Filters")

    # Date Range with error handling
    try:
        start_date, end_date = st.sidebar.date_input(
            "Select Date Range",
            [df['Date'].min(), df['Date'].max()]
        )
    except ValueError:
        st.error("Please select a valid start and end date.")
        st.stop()

    category_filter = st.sidebar.multiselect(
        "Select Category", df['Category'].unique(), default=df['Category'].unique()
    )

    filtered_df = df[
        (df['Date'].between(pd.to_datetime(start_date), pd.to_datetime(end_date))) &
        (df['Category'].isin(category_filter))
    ]

    # ---------------- KPIs ----------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Total Sales", f"₹ {filtered_df['Sales'].sum():,.0f}")
    col2.metric("📦 Quantity Sold", f"{filtered_df['Quantity'].sum():,}")
    col3.metric("🧾 Total Profit", f"₹ {filtered_df['Profit'].sum():,.0f}")
    col4.metric("🛍️ Products", filtered_df['Product'].nunique())

    st.divider()

    # ---------------- Sales Trend ----------------
    st.subheader("📈 Sales Trend")
    daily_sales = filtered_df.groupby('Date')['Sales'].sum()
    st.line_chart(daily_sales)

    # ---------------- Product & Category Analysis ----------------
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("🏆 Top 5 Products")
        product_sales = filtered_df.groupby('Product')['Sales'].sum().sort_values(ascending=False)
        st.bar_chart(product_sales.head(5))

    with col_b:
        st.subheader("🗂️ Sales by Category")
        category_sales = filtered_df.groupby('Category')['Sales'].sum()
        st.bar_chart(category_sales)

    st.divider()

    # ---------------- ABC Inventory Analysis ----------------
    st.subheader("📦 ABC Inventory Classification")
    # 
    
    abc_df = (
        filtered_df.groupby('Product')['Sales']
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    abc_df['Cumulative %'] = abc_df['Sales'].cumsum() / abc_df['Sales'].sum() * 100
    abc_df['Class'] = abc_df['Cumulative %'].apply(lambda x: 'A' if x <= 70 else ('B' if x <= 90 else 'C'))
    
    # Displaying with color highlighting
    def color_abc(val):
        color = 'lightgreen' if val == 'A' else 'orange' if val == 'B' else 'pink'
        return f'background-color: {color}'
    
    st.dataframe(abc_df.style.applymap(color_abc, subset=['Class']))

    # ---------------- Profit Margin Heatmap ----------------
    st.subheader("🔥 Profit Margin Heatmap")
    heatmap_df = filtered_df.pivot_table(
        values='Profit_Margin', index='Category', columns='Product', aggfunc='mean'
    ).fillna(0)
    st.dataframe(heatmap_df.style.background_gradient(cmap="RdYlGn"))

    # ---------------- Download PDF (Improved Logic) ----------------
    st.subheader("⬇️ Download Reports")
    
    if st.button("📄 Generate PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "Supermarket Sales Summary Report", ln=True, align='C')
        pdf.ln(10)
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, f"Total Sales: INR {filtered_df['Sales'].sum():,.2f}", ln=True)
        pdf.cell(200, 10, f"Total Profit: INR {filtered_df['Profit'].sum():,.2f}", ln=True)
        
        # Use Output as a string/bytes buffer for Streamlit compatibility
        pdf_output = pdf.output(dest='S').encode('latin-1')
        st.download_button(
            label="📥 Download PDF",
            data=pdf_output,
            file_name="Supermarket_Report.pdf",
            mime="application/pdf"
        )

else:
    st.info("👆 Please upload a CSV file to begin.")

