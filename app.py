import streamlit as st
import pandas as pd

st.set_page_config(page_title="Supermarket Analytics", layout="wide")

st.title("🛒 Supermarket Analytics Dashboard")

file = st.file_uploader("Upload Supermarket Sales Data (CSV)", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    # --- Data Preparation ---
    df["Date"] = pd.to_datetime(df["Date"])
    df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M").dt.hour
    df["Total"] = df["Quantity"] * df["Price"] * (1 - df["Discount"] / 100)

    # --- Filters ---
    st.sidebar.header("🔎 Filters")

    category_filter = st.sidebar.multiselect(
        "Select Category",
        options=df["Category"].unique(),
        default=df["Category"].unique()
    )

    product_filter = st.sidebar.multiselect(
        "Select Product",
        options=df["Product"].unique(),
        default=df["Product"].unique()
    )

    filtered_df = df[
        (df["Category"].isin(category_filter)) &
        (df["Product"].isin(product_filter))
    ]

    # --- KPIs ---
    st.subheader("📊 Executive KPIs")
    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Total Sales (₹)", round(filtered_df["Total"].sum(), 2))
    k2.metric("Total Quantity", filtered_df["Quantity"].sum())
    k3.metric("Avg Bill Value (₹)", round(filtered_df.groupby("Bill_ID")["Total"].sum().mean(), 2))
    k4.metric("Total Transactions", filtered_df["Bill_ID"].nunique())

    # --- Charts ---
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📅 Daily Sales Trend")
        daily_sales = filtered_df.groupby("Date")["Total"].sum()
        st.line_chart(daily_sales)

    with c2:
        st.subheader("⏰ Peak Shopping Hours")
        hourly_sales = filtered_df.groupby("Hour")["Total"].sum()
        st.bar_chart(hourly_sales)

    c3, c4 = st.columns(2)

    with c3:
        st.subheader("🥇 Top Products")
        top_products = filtered_df.groupby("Product")["Quantity"].sum().sort_values(ascending=False)
        st.bar_chart(top_products)

    with c4:
        st.subheader("💰 Discount Effectiveness")
        discount_sales = filtered_df.groupby("Discount")["Total"].sum()
        st.bar_chart(discount_sales)

    # --- Inventory ---
    st.subheader("📦 Inventory Alerts")

    inventory = filtered_df.groupby("Product")["Current_Stock"].mean().reset_index()
    avg_sales = filtered_df.groupby("Product")["Quantity"].mean().mean()

    inventory["Stock_Status"] = inventory["Current_Stock"].apply(
        lambda x: "⚠️ Reorder Required" if x < avg_sales else "✅ Stock OK"
    )

    st.dataframe(inventory)

    # --- Raw Data ---
    with st.expander("📄 View Raw Data"):
        st.dataframe(filtered_df)


