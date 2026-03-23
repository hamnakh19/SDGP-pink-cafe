import streamlit as st
import pandas as pd
import datetime

st.set_page_config(
    page_title="Bristol Pink Bakery Dashboard",
    layout="wide"
)

st.markdown("""
<style>

/* Main title */
.main-title {
    font-size: 40px;
    font-weight: 700;
    color: #1E0F2E;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    font-size: 18px;
    color: #6B4C7A;
    margin-bottom: 25px;
}

/* Section headers */
.section-header {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 10px;
    color: #1E0F2E;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(180deg, #2A153F, #1E0F2E);
    padding: 20px;
    border-radius: 14px;
    margin-bottom: 20px;
    border-left: 6px solid #E75480;
}

.metric-title {
    font-size: 14px;
    color: #E8D8E6;
}

.metric-value {
    font-size: 28px;
    font-weight: bold;
    color: white;
}

/* Graph placeholder boxes */
.graph-box {
    background-color: white;
    border: 2px dashed #E75480;
    border-radius: 12px;
    height: 350px;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:16px;
    color:#6B4C7A;
}

/* Sidebar colour */
section[data-testid="stSidebar"] {
    background-color: #1E0F2E;
}

section[data-testid="stSidebar"] * {
    color: white;
}

</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## 🧁 Bristol Pink Bakery")
st.sidebar.markdown("### Dashboards")
st.sidebar.write("• Sales Forecasting")
st.sidebar.write("• Data Overview")

# Titles
st.markdown('<div class="main-title">Bristol Pink Bakery</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Operational Sales Forecast Dashboard</div>',
    unsafe_allow_html=True
)

# Upload CSV
uploaded_file = st.file_uploader(
    "Upload sales CSV file to begin analysis",
    type=["csv"]
)

if uploaded_file is None:

    st.info("Please upload a CSV file to begin analysing bakery sales.")

else:

    try:

        df = pd.read_csv(uploaded_file)

        st.success("Sales dataset loaded. Dashboard updated with analysis and forecasts.")

        # Date Range
        st.markdown("### Date Range")

        col1, col2 = st.columns(2)

        with col1:
            start_date = st.date_input(
                "Start Date",
                datetime.date(2024, 1, 1),
                format="DD/MM/YYYY"
            )

        with col2:
            end_date = st.date_input(
                "End Date",
                datetime.date(2024, 1, 28),
                format="DD/MM/YYYY"
            )

        st.markdown(
            f"<div style='color:#6B4C7A; font-size:14px; margin-bottom:20px;'>"
            f"Selected Date Range: {start_date.strftime('%d/%m/%Y')} – {end_date.strftime('%d/%m/%Y')}"
            f"</div>",
            unsafe_allow_html=True
        )

        left_col, right_col = st.columns([1, 3])

        # Metrics
        with left_col:

            st.markdown('<div class="section-header">Key Forecast Metrics</div>', unsafe_allow_html=True)

            st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Predicted Revenue (Next 4 Weeks)</div>
                <div class="metric-value">£4,120</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Training Window</div>
                <div class="metric-value">6 Weeks</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Model Accuracy (MAPE)</div>
                <div class="metric-value">8.3%</div>
            </div>
            """, unsafe_allow_html=True)

        # Graphs section
        with right_col:

            tabs = st.tabs([
                "Daily Sales",
                "Monthly Comparison",
                "Average Sales by Weekday",
                "Forecast Models"
            ])

            # Daily sales graphs
            with tabs[0]:

                st.subheader("Daily Sales by Item")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.image("graphs/croissant_daily.png", caption= "Croissant Daily Sales", use_container_width=True)

                with col2:
                    st.image("graphs/americano_daily.png", caption= "Americano Daily Sales", use_container_width=True)

                with col3:
                    st.image("graphs/cappuccino_daily.png", caption= "Cappuccino Daily Sales", use_container_width=True)

            # Monthly comparison
            with tabs[1]:

                st.subheader("Monthly Sales Comparison")

                st.image("graphs/monthly_comparison.png", caption= "Monthly Sales Comparison", use_container_width=True)

            # Average weekday sales
            with tabs[2]:

                st.subheader("Average Sales by Day of Week")

                st.image("graphs/average_sales.png", caption= "Weekday Sales Pattern", use_container_width=True)

            # Forecast models
            with tabs[3]:

                st.subheader("Forecast Models")

                model = st.selectbox(
                    "Select Model",
                    [
                        "Linear Regression",
                        "Random Forest",
                        "ARIMA",
                        "Moving Average",
                        "LSTM"
                    ]
                )

                st.markdown(
                    f'<div class="graph-box">{model} Forecast Graph</div>',
                    unsafe_allow_html=True
                )

    except Exception as e:

        st.error("Error reading the uploaded CSV file.")
        st.exception(e)