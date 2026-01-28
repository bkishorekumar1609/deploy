import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
from scipy import stats
import matplotlib.pyplot as plt


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

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Supermarket Analytics App",
    layout="wide"
)

st.title("🛒 Automated Supermarket Analytics Dashboard")
st.markdown(
    """
    Upload supermarket sales data (CSV or Excel) and get **instant insights**
    on sales, inventory, customers, and discounts.
    """
)

# -------------------- FILE UPLOAD --------------------
file = st.file_uploader(
    "📂 Upload Sales Data (CSV / Excel)",
    type=["csv", "xlsx"]
)

if file is not None:

    # Read file
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.success("✅ File uploaded successfully!")

    # -------------------- DATA PREPARATION --------------------
    df['Date'] = pd.to_datetime(df['Date'])
    df['Sales'] = (df['Quantity'] * df['Price']) - df['Discount']

    # -------------------- SIDEBAR FILTERS --------------------
    st.sidebar.header("🔍 Filters")

    category_filter = st.sidebar.multiselect(
        "Select Category",
        options=df['Category'].unique(),
        default=df['Category'].unique()
    )

    df = df[df['Category'].isin(category_filter)]

    # -------------------- KPI METRICS --------------------
    total_sales = df['Sales'].sum()
    total_orders = df['Bill_ID'].nunique()
    avg_bill = total_sales / total_orders if total_orders != 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Total Sales", f"₹ {total_sales:,.0f}")
    col2.metric("🧾 Total Bills", total_orders)
    col3.metric("📊 Avg Bill Value", f"₹ {avg_bill:,.0f}")

    # -------------------- DATA PREVIEW --------------------
    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # -------------------- SALES TRENDS --------------------
    st.subheader("📈 Sales Trends")

    daily_sales = df.groupby('Date')['Sales'].sum()
    monthly_sales = df.groupby(df['Date'].dt.to_period('M'))['Sales'].sum()

    col1, col2 = st.columns(2)
    col1.write("### Daily Sales")
    col1.line_chart(daily_sales)

    col2.write("### Monthly Sales")
    col2.bar_chart(monthly_sales)

    # -------------------- PRODUCT PERFORMANCE --------------------
    st.subheader("📦 Product Performance")

    product_sales = (
        df.groupby('Product')['Quantity']
        .sum()
        .sort_values(ascending=False)
    )

    col1, col2 = st.columns(2)

    col1.write("### 🔥 Fast-Moving Products")
    col1.dataframe(product_sales.head(10))

    col2.write("### 🐌 Slow-Moving Products")
    col2.dataframe(product_sales.tail(10))

    # -------------------- INVENTORY MANAGEMENT --------------------
    st.subheader("🏪 Inventory Alerts")

    low_stock = df[df['Current_Stock'] < 10][
        ['Product', 'Current_Stock']
    ].drop_duplicates()

    if len(low_stock) > 0:
        st.warning("⚠️ Low Stock Products")
        st.dataframe(low_stock)
    else:
        st.success("✅ No stock-out risk detected")

    # -------------------- CUSTOMER BEHAVIOR --------------------
    st.subheader("🛍️ Customer Purchase Behavior")

    items_per_bill = df.groupby('Bill_ID')['Product'].count()

    col1, col2 = st.columns(2)

    col1.write("### Items per Bill")
    col1.bar_chart(items_per_bill.value_counts())

    col2.write("### Top Customers")
    top_customers = (
        df.groupby('Customer_ID')['Sales']
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    col2.dataframe(top_customers)

    # -------------------- DISCOUNT ANALYSIS --------------------
    st.subheader("💸 Discount Impact Analysis")

    discount_sales = df.groupby('Discount')['Sales'].sum()

    st.bar_chart(discount_sales)

    # -------------------- DOWNLOAD REPORT --------------------
    st.subheader("📥 Download Processed Data")

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="supermarket_analysis_output.csv",
        mime="text/csv"
    )

else:
    st.info("👆 Please upload a CSV or Excel file to start analysis.")
