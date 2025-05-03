import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime

# Sample log data (in practice, load from DB or CSV)
log_data = pd.DataFrame([
    {"timestamp": "2025-05-01 10:00", "query": "Scope output mismatch", "category": "simulation", "success": True, "latency": 3.2},
    {"timestamp": "2025-05-01 10:05", "query": "Install toolbox error", "category": "installation", "success": False, "latency": 5.1},
    {"timestamp": "2025-05-01 10:10", "query": "MATLAB path issue", "category": "environment", "success": True, "latency": 2.3},
    {"timestamp": "2025-05-01 10:12", "query": "Simulation not starting", "category": "simulation", "success": True, "latency": 4.0},
    {"timestamp": "2025-05-01 10:18", "query": "Code generation error", "category": "codegen", "success": False, "latency": 6.4},
])
log_data["timestamp"] = pd.to_datetime(log_data["timestamp"])

st.set_page_config(page_title="AI Troubleshooting Admin Dashboard", layout="wide")
st.title("🔧 AI Troubleshooting Admin Dashboard")

# Filters
st.sidebar.header("Filters")
category_filter = st.sidebar.multiselect("Filter by Category", log_data["category"].unique(), default=log_data["category"].unique())
success_filter = st.sidebar.selectbox("Filter by Success", ["All", "Success", "Failure"])

filtered_data = log_data[log_data["category"].isin(category_filter)]
if success_filter == "Success":
    filtered_data = filtered_data[filtered_data["success"] == True]
elif success_filter == "Failure":
    filtered_data = filtered_data[filtered_data["success"] == False]

# Metrics
st.subheader("📊 Metrics Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Queries", len(filtered_data))
col2.metric("Success Rate", f"{filtered_data['success'].mean() * 100:.1f}%")
col3.metric("Avg. Latency (s)", f"{filtered_data['latency'].mean():.2f}")

# Charts
st.subheader("📈 Query Volume Over Time")
time_chart = alt.Chart(filtered_data).mark_line(point=True).encode(
    x='timestamp:T',
    y='count():Q',
    tooltip=['timestamp:T', 'count():Q']
).properties(height=300)
st.altair_chart(time_chart, use_container_width=True)

st.subheader("📌 Breakdown by Category")
category_chart = alt.Chart(filtered_data).mark_bar().encode(
    x='category:N',
    y='count():Q',
    color='category:N',
    tooltip=['category:N', 'count():Q']
).properties(height=300)
st.altair_chart(category_chart, use_container_width=True)

# Display Logs
st.subheader("📄 Detailed Logs")
st.dataframe(filtered_data.sort_values("timestamp", ascending=False), use_container_width=True)

# Export Option
st.download_button(
    label="📥 Export Logs to CSV",
    data=filtered_data.to_csv(index=False).encode('utf-8'),
    file_name='troubleshooting_logs.csv',
    mime='text/csv'
)

# Admin Action Placeholder
st.subheader("🛠 Admin Actions")
if st.button("🔁 Re-run Selected Queries with Updated Model"):
    st.info("[Simulated] Queries have been re-processed.")

if st.button("⚠️ Flag Inaccurate Responses"):
    st.warning("[Simulated] Selected responses flagged for review.")
