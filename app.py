import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
from scipy import stats
import matplotlib.pyplot as plt
import io
from fpdf import FPDF
from xlsxwriter import Workbook


st.set_page_config(page_title="SK Analytics App", layout="wide")
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

# ---------------- Page Config ----------------
st.set_page_config(page_title="Supermarket Analytics Dashboard", layout="wide")
st.title("🛒 Supermarket Analytics Dashboard")

# ---------------- File Upload ----------------
file = st.file_uploader("Upload Supermarket Sales Data (CSV)", type=["csv"])

if file:
    df = pd.read_csv(file)

    # ---------------- Preprocessing ----------------
    df['Date'] = pd.to_datetime(df['Date'])
    df['Sales'] = df['Quantity'] * df['Price']
    df['Profit'] = df['Sales'] - (df['Quantity'] * df['Cost'])
    df['Profit_Margin'] = df['Profit'] / df['Sales']

    # ---------------- Sidebar Filters ----------------
    st.sidebar.header("🔍 Filters")

    date_range = st.sidebar.date_input(
        "Select Date Range",
        [df['Date'].min(), df['Date'].max()]
    )

    category_filter = st.sidebar.multiselect(
        "Select Category",
        df['Category'].unique(),
        default=df['Category'].unique()
    )

    product_filter = st.sidebar.multiselect(
        "Select Product",
        df['Product'].unique(),
        default=df['Product'].unique()
    )

    filtered_df = df[
        (df['Date'].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))) &
        (df['Category'].isin(category_filter)) &
        (df['Product'].isin(product_filter))
    ]

    # ---------------- KPIs ----------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Total Sales", f"₹ {filtered_df['Sales'].sum():,.0f}")
    col2.metric("📦 Quantity Sold", filtered_df['Quantity'].sum())
    col3.metric("🧾 Total Profit", f"₹ {filtered_df['Profit'].sum():,.0f}")
    col4.metric("🛍️ Products", filtered_df['Product'].nunique())

    st.divider()

    # ---------------- Sales Trend ----------------
    st.subheader("📈 Sales Trend")
    daily_sales = filtered_df.groupby('Date')['Sales'].sum()
    st.line_chart(daily_sales)

    st.divider()

    # ---------------- Product Performance ----------------
    st.subheader("🏆 Product Performance")
    product_sales = filtered_df.groupby('Product')['Sales'].sum().sort_values(ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Top 5 Products")
        st.bar_chart(product_sales.head(5))
    with col2:
        st.write("Least 5 Products")
        st.bar_chart(product_sales.tail(5))

    st.divider()

    # ---------------- Inventory Analysis ----------------
    st.subheader("📦 Inventory Analysis")
    inventory = filtered_df.groupby('Product')['Stock'].mean()
    low_stock = inventory[inventory < 20]

    st.write("⚠️ Low Stock Products")
    st.dataframe(low_stock)
    st.bar_chart(inventory)

    st.divider()

    # ---------------- Category Analysis ----------------
    st.subheader("🗂️ Category Analysis")
    category_sales = filtered_df.groupby('Category')['Sales'].sum()
    category_profit = filtered_df.groupby('Category')['Profit'].sum()

    col1, col2 = st.columns(2)
    with col1:
        st.write("Sales by Category")
        st.bar_chart(category_sales)
    with col2:
        st.write("Profit by Category")
        st.bar_chart(category_profit)

    st.divider()

    # ---------------- Sales Forecast ----------------
    st.subheader("🔮 Sales Forecast (Next 7 Days)")
    daily_sales_df = filtered_df.groupby('Date')['Sales'].sum().reset_index()
    daily_sales_df['MA_7'] = daily_sales_df['Sales'].rolling(7).mean()

    last_ma = daily_sales_df['MA_7'].iloc[-1]
    future_dates = pd.date_range(
        start=daily_sales_df['Date'].max() + pd.Timedelta(days=1),
        periods=7
    )

    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecasted Sales': [last_ma] * 7
    })

    st.line_chart(daily_sales_df.set_index('Date')[['Sales', 'MA_7']])
    st.dataframe(forecast_df)

    st.divider()

    # ---------------- ABC Inventory Analysis ----------------
    st.subheader("📦 ABC Inventory Classification")

    abc_df = (
        filtered_df.groupby('Product')['Sales']
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    abc_df['Cumulative %'] = abc_df['Sales'].cumsum() / abc_df['Sales'].sum() * 100

    def abc_class(x):
        if x <= 70:
            return 'A'
        elif x <= 90:
            return 'B'
        else:
            return 'C'

    abc_df['Class'] = abc_df['Cumulative %'].apply(abc_class)
    st.dataframe(abc_df)

    st.divider()

    # ---------------- Profit Margin Heatmap ----------------
    st.subheader("🔥 Profit Margin Heatmap")

    heatmap_df = filtered_df.pivot_table(
        values='Profit_Margin',
        index='Category',
        columns='Product',
        aggfunc='mean'
    )

    st.dataframe(heatmap_df.style.background_gradient(cmap="RdYlGn"))

    st.divider()

    # ---------------- Automatic Insights ----------------
    st.subheader("🧠 Automatic Insights")

    st.success(f"""
    🔹 Top Product: {product_sales.idxmax()}  
    🔹 Lowest Product: {product_sales.idxmin()}  
    🔹 Best Category: {category_sales.idxmax()}  
    🔹 Total Profit: ₹ {filtered_df['Profit'].sum():,.0f}
    """)

    st.divider()

    # ---------------- Download Excel ----------------
    st.subheader("⬇️ Download Report")

    report_df = filtered_df.groupby('Category').agg(
        Total_Sales=('Sales', 'sum'),
        Total_Profit=('Profit', 'sum'),
        Quantity_Sold=('Quantity', 'sum')
    ).reset_index()

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        report_df.to_excel(writer, index=False, sheet_name='Summary')

    st.download_button(
        "📥 Download Excel Report",
        data=output.getvalue(),
        file_name="Supermarket_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # ---------------- Download PDF ----------------
    if st.button("📄 Download PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, "Supermarket Sales Summary Report", ln=True)
        pdf.ln(5)
        pdf.cell(200, 10, f"Total Sales: ₹ {filtered_df['Sales'].sum():,.0f}", ln=True)
        pdf.cell(200, 10, f"Total Profit: ₹ {filtered_df['Profit'].sum():,.0f}", ln=True)
        pdf.cell(200, 10, f"Total Quantity: {filtered_df['Quantity'].sum()}", ln=True)

        pdf.output("report.pdf")

        with open("report.pdf", "rb") as f:
            st.download_button(
                "📥 Download PDF",
                data=f,
                file_name="Supermarket_Report.pdf",
                mime="application/pdf"
            )

    # ---------------- Data Preview ----------------
    with st.expander("📄 View Data"):
        st.dataframe(filtered_df)

else:
    st.info("👆 Upload a CSV file to start analysis")
