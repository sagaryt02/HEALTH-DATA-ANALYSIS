# HEALTH-DATA-ANALYSIS
# py_health_2.py
# One-click Streamlit app launcher + Health Data Analysis dashboard
# Save this file and double-click (or run `python py_health_2.py`) to open the Streamlit app.

import os
import sys
import subprocess
import traceback

# ---------- LAUNCHER: ensure double-click runs the app ----------
# When the script is double-clicked, this Python process will spawn:
#   python -m streamlit run py_health_2.py
# with an env var LAUNCHED_BY_SCRIPT=1 so the Streamlit-run instance knows not to spawn again.

if os.environ.get("LAUNCHED_BY_SCRIPT") != "1":
    # Not launched by streamlit-run wrapper yet -> spawn streamlit and exit
    try:
        python_exe = sys.executable or "python"
        # Build command: python -m streamlit run <thisfile>
        cmd = [python_exe, "-m", "streamlit", "run", __file__]
        # Pass through current environment and add marker
        env = os.environ.copy()
        env["LAUNCHED_BY_SCRIPT"] = "1"
        # Use subprocess.Popen so this launcher can exit immediately
        subprocess.Popen(cmd, env=env)
        # Optionally inform user in console (useful if they double-click and console shows)
        print("Starting Streamlit app... a browser window should open shortly.")
    except FileNotFoundError:
        print("ERROR: Python executable not found. Run the app with: python -m streamlit run", __file__)
    except Exception:
        print("ERROR launching Streamlit:")
        traceback.print_exc()
    # Exit the launcher process (the Streamlit process will run separately)
    sys.exit(0)

# ---------- If we reach here, we are inside the Streamlit-run process ----------
# Import dashboard libraries and build the app

import streamlit as st               # For interactive dashboard UI
import pandas as pd                  # Tabular data handling
import numpy as np                   # Numeric operations
import mysql.connector               # MySQL connector
import matplotlib.pyplot as plt      # Plots
from mysql.connector import Error

# --- Page config ---
st.set_page_config(page_title="Health Data Analysis", layout="wide")

# --- Helper: fetch data from MySQL safely ---
@st.cache_data(ttl=300)
def get_data_from_db(host="localhost", user="root", password="@K6543216s", database="health_db"):
    """
    Connects to MySQL and returns a pandas DataFrame with joined patient + vitals data.
    Cached for 5 minutes (adjust ttl as needed).
    """
    try:
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        query = """
            SELECT p.patient_id, p.name, p.age, p.gender, v.bp, v.heart_rate, v.cholesterol
            FROM patients p
            JOIN vitals v ON p.patient_id = v.patient_id
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Error as e:
        # Return None on error (caller will show message)
        st.error(f"Database connection or query failed: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None

# --- App layout ---
st.title("Health Data Analysis Dashboard")
st.markdown(
    "Interactive dashboard to explore patient vitals. "
)

# Sidebar: DB connection config (expandable)
with st.sidebar.expander("Database connection (change if needed)"):
    host = st.text_input("MySQL host", value="localhost")
    user = st.text_input("MySQL user", value="root")
    password = st.text_input("MySQL password", value="@K6543216s", type="password")
    database = st.text_input("Database name", value="health_db")
    if st.button("Reload data"):
        # clear cache and reload
        get_data_from_db.clear()
        st.experimental_rerun()

# Load data
df = get_data_from_db(host=host, user=user, password=password, database=database)

if df is None:
    st.stop()  # stop if DB failed (message already shown)

# Basic cleaning: ensure columns expected
expected_cols = {"patient_id", "name", "age", "gender", "bp", "heart_rate", "cholesterol"}
if not expected_cols.issubset(set(df.columns)):
    st.error(f"Dataframe missing expected columns. Found columns: {list(df.columns)}")
    st.stop()

# Convert types and handle missing values
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["heart_rate"] = pd.to_numeric(df["heart_rate"], errors="coerce")
df["cholesterol"] = pd.to_numeric(df["cholesterol"], errors="coerce")
df["gender"] = df["gender"].astype(str).str.strip()

# Sidebar filters
st.sidebar.markdown("### Filters")
min_age, max_age = int(np.nanmin(df["age"].dropna())), int(np.nanmax(df["age"].dropna()))
age_filter = st.sidebar.slider(
    "Select Age Range",
    min_value=min_age,
    max_value=max_age,
    value=(min_age, max_age)
)
gender_options = sorted(df["gender"].dropna().unique().tolist())
gender_filter = st.sidebar.multiselect(
    "Select Gender",
    options=gender_options,
    default=gender_options
)

# Apply filters
filtered_df = df[
    (df["age"].notna()) &
    (df["age"] >= age_filter[0]) &
    (df["age"] <= age_filter[1]) &
    (df["gender"].isin(gender_filter))
].copy()

# Main: show filtered data
st.header("Filtered Patient Data")
st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)

# Visualization area (two columns)
col1, col2 = st.columns(2)

with col1:
    st.subheader("Age Distribution")
    if filtered_df.empty:
        st.info("No data for selected filters.")
    else:
        fig1, ax1 = plt.subplots()
        ax1.hist(filtered_df["age"].dropna(), bins=10)
        ax1.set_xlabel("Age")
        ax1.set_ylabel("Number of Patients")
        ax1.set_title("Age Distribution")
        st.pyplot(fig1)

    st.subheader("Age vs. Cholesterol")
    if not filtered_df.empty:
        fig3, ax3 = plt.subplots()
        ax3.scatter(filtered_df["age"], filtered_df["cholesterol"])
        ax3.set_xlabel("Age")
        ax3.set_ylabel("Cholesterol Level")
        ax3.set_title("Age vs. Cholesterol")
        st.pyplot(fig3)

with col2:
    st.subheader("Gender Distribution")
    if filtered_df.empty:
        st.info("No data for selected filters.")
    else:
        gender_counts = filtered_df["gender"].value_counts()
        fig2, ax2 = plt.subplots()
        ax2.bar(gender_counts.index.astype(str), gender_counts.values)
        ax2.set_xlabel("Gender")
        ax2.set_ylabel("Count")
        ax2.set_title("Gender Distribution")
        st.pyplot(fig2)

    st.subheader("Summary Statistics")
    if filtered_df.empty:
        st.info("No data for selected filters.")
    else:
        stats = filtered_df[["age", "heart_rate", "cholesterol"]].describe().T
        st.table(stats)

# Optional: allow CSV download of filtered data
def convert_df_to_csv(df_in):
    return df_in.to_csv(index=False).encode("utf-8")

csv_bytes = convert_df_to_csv(filtered_df)
st.download_button("Download filtered data (CSV)", data=csv_bytes, file_name="health_filtered.csv", mime="text/csv")

# Footer / notes
st.markdown("---")
st.caption("Dashboard generated using Python, Streamlit, pandas, MySQL connector and matplotlib.")
