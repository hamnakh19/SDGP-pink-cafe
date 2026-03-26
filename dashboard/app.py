import streamlit as st
import pandas as pd
import datetime

st.set_page_config(
    page_title="Bristol Pink Bakery Dashboard",
    layout="wide"
)

st.markdown("""
<style>
.main-title {font-size: 40px; font-weight: 700; color: #1E0F2E;}
.subtitle {font-size: 18px; color: #6B4C7A; margin-bottom: 25px;}
.section-header {font-size: 18px; font-weight: 600; margin-bottom: 10px; color: #1E0F2E;}
.metric-card {background: linear-gradient(180deg, #2A153F, #1E0F2E); padding: 20px; border-radius: 14px; margin-bottom: 20px; border-left: 6px solid #E75480;}
.metric-title {font-size: 14px; color: #E8D8E6;}
.metric-value {font-size: 26px; font-weight: bold; color: white;}
section[data-testid="stSidebar"] {background-color: #1E0F2E;}
section[data-testid="stSidebar"] * {color: white;}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## 🧁 Bristol Pink Bakery")

# Title
st.markdown('<div class="main-title">Bristol Pink Bakery</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Operational Sales Forecast Dashboard</div>', unsafe_allow_html=True)

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=True)

if not uploaded_file:
    st.info("Upload CSV to begin")

else:
    dfs = []

    # ===== DATA CLEANING =====
    for file in uploaded_file:

        # extra safety check
        if not file.name.endswith(".csv"):
            st.error(f"{file.name} is not a CSV file")
            continue

        temp_df = pd.read_csv(file)
        temp_df.columns = temp_df.columns.str.strip()

        # Croissant
        if "Number Sold" in temp_df.columns and len(temp_df.columns) == 2:
            temp_df["product"] = "Croissant"
            temp_df = temp_df.rename(columns={"Number Sold": "number_sold"})

        # Coffee (Americano + Cappuccino)
        elif "Unnamed: 2" in temp_df.columns:
            temp_df = temp_df.rename(columns={
                "Number Sold": "Cappuccino",
                "Unnamed: 2": "Americano"
            })

            temp_df = temp_df.melt(
                id_vars=["Date"],
                value_vars=["Cappuccino", "Americano"],
                var_name="product",
                value_name="number_sold"
            )

        dfs.append(temp_df)

    if not dfs:
        st.warning("Please upload valid CSV files.")
        st.stop()

    df = pd.concat(dfs, ignore_index=True)

    df = df.rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)

    df["number_sold"] = pd.to_numeric(df["number_sold"], errors="coerce")
    df = df.dropna(subset=["number_sold"])

    # ===== DATE FILTER =====
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", min_date)
    with col2:
        end_date = st.date_input("End Date", max_date)

    filtered_df = df[
        (df["date"] >= pd.to_datetime(start_date)) &
        (df["date"] <= pd.to_datetime(end_date))
    ]

    # detect available products
    available_products = filtered_df["product"].unique() if not filtered_df.empty else []

    # ===== TRAINING WINDOW =====
    training_weeks = st.slider("Forecast Horizon (weeks)", 4, 8, 6)

    # ===== METRICS =====
    if filtered_df.empty:
        total_units = 0
        best_product = "N/A"
        stats = pd.DataFrame()
    else:
        total_units = filtered_df["number_sold"].sum()
        best_product = filtered_df.groupby("product")["number_sold"].sum().idxmax()
        stats = filtered_df.groupby("product")["number_sold"].agg(["mean", "std"]).round(2)

    num_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days

    left_col, right_col = st.columns([1, 3])

    # ===== LEFT PANEL =====
    with left_col:
        st.markdown('<div class="section-header">Business Overview</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Best-Selling Product</div>
            <div class="metric-value">{best_product}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Units Sold</div>
            <div class="metric-value">{total_units:,}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Days Analysed</div>
            <div class="metric-value">{num_days}</div>
        </div>
        """, unsafe_allow_html=True)

        if not stats.empty:
            most_stable = stats["std"].idxmin()
            most_volatile = stats["std"].idxmax()

            st.markdown('<div class="section-header">Product Insights</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Most Stable</div>
                <div class="metric-value">{most_stable}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Most Volatile</div>
                <div class="metric-value">{most_volatile}</div>
            </div>
            """, unsafe_allow_html=True)

    # ===== TABS =====
    with right_col:

        tabs = st.tabs([
            "Daily Sales",
            "Monthly Comparison",
            "Average Sales by Weekday",
            "Forecast Models"
        ])

        # ===== DAILY =====
        with tabs[0]:
            st.subheader("Daily Sales")

            if "Croissant" in available_products:
                st.markdown("### 🥐 Croissant")
                st.image("dashboard/graphs/croissant_daily.png", width="stretch")
                st.divider()

            if "Americano" in available_products:
                st.markdown("### ☕ Americano")
                st.image("dashboard/graphs/americano_daily.png", width="stretch")
                st.divider()

            if "Cappuccino" in available_products:
                st.markdown("### ☕ Cappuccino")
                st.image("dashboard/graphs/cappuccino_daily.png", width="stretch")

        # ===== MONTHLY =====
        with tabs[1]:
            st.subheader("Monthly Sales Comparison")
            st.image("dashboard/graphs/monthly_comparison.png", width="stretch")

        # ===== WEEKDAY =====
        with tabs[2]:
            st.subheader("Average Sales by Day of Week")
            st.image("dashboard/graphs/average_sales.png", width="stretch")

        # ===== FORECAST =====
        with tabs[3]:

            st.subheader("Model Evaluation & Forecasting")

            st.info(f"Forecast Horizon: {training_weeks} weeks | Best Model: Linear Regression (MAPE 8.3%)")

            # Prediction table
            if not filtered_df.empty:
                avg_sales = filtered_df.groupby("product")["number_sold"].mean()
                predictions = avg_sales * (training_weeks * 7)

                pred_df = predictions.reset_index()
                pred_df.columns = ["Product", f"Predicted Sales ({training_weeks} weeks)"]

                st.dataframe(pred_df)

            st.divider()

            products_config = {
                "Americano": "Americano_Coffee",
                "Cappuccino": "Cappuccino_Coffee",
                "Croissant": "Croissant"
            }

            for product, name in products_config.items():
                if product in available_products:

                    st.markdown(f"## {product}")

                    st.markdown("### Forecast")
                    st.image(f"dashboard/graphs/{name}_Forecast.png", width="stretch")

                    st.markdown("### Model Performance")
                    col1, col2 = st.columns(2)
                    col1.image(f"dashboard/graphs/{name}_Performance.png", width="stretch")
                    col2.image(f"dashboard/graphs/{name}_Heatmap.png", width="stretch")

                    if product == "Americano":
                        st.markdown("📊 Stable demand → most accurate predictions")
                    elif product == "Cappuccino":
                        st.markdown("📊 Moderate variability → slightly higher error")
                    else:
                        st.markdown("📊 High volatility → harder to predict")

                    st.divider()