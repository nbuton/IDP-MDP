import streamlit as st
import plotly.express as px
import h5py
import os
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
BASE_PATH = "data/IDRome/IDRome_v4"
REFERENCE_ID = "Q5SYC1_270_327"

st.set_page_config(layout="wide", page_title="IDRome Global Statistics")
st.title("📊 IDRome Global Feature Distribution")


# --- HELPER: GET PATH FROM ID ---
def get_h5_path(prot_id):
    try:
        parts = prot_id.split("_")
        mid = parts[0]
        suffix = "_".join(parts[1:])
        # Sharding: A0/AU/Z9/...
        return os.path.join(
            BASE_PATH, mid[0:2], mid[2:4], mid[4:6], suffix, "properties.h5"
        )
    except:
        return None


# --- 1. DISCOVER KEYS (From Reference Protein) ---
@st.cache_data
def get_available_keys(ref_id):
    path = get_h5_path(ref_id)
    keys = []
    if path and os.path.exists(path):
        with h5py.File(path, "r") as h5:
            # Recursively find all dataset names
            h5.visititems(
                lambda name, node: (
                    keys.append(name) if isinstance(node, h5py.Dataset) else None
                )
            )
    return keys


# Run the discovery
all_keys = get_available_keys(REFERENCE_ID)

if not all_keys:
    st.error(
        f"Could not find reference protein at `{get_h5_path(REFERENCE_ID)}`. Please check your BASE_PATH."
    )
    st.stop()

# --- 2. USER SELECTION ---
selected_key = st.selectbox(
    "Select a Property to analyze across all proteins:", all_keys
)


# --- 3. GLOBAL DATA COLLECTION ---
@st.cache_data(show_spinner=f"Scanning files for {selected_key}...")
def collect_global_data(prop_name):
    values = []
    base = Path(BASE_PATH)
    # Recursively find all properties.h5 files
    file_list = list(base.rglob("properties.h5"))

    if not file_list:
        return np.array([]), "No H5 files found in the directory tree."

    progress_text = "Processing proteins..."
    my_bar = st.progress(0, text=progress_text)

    total = len(file_list)
    for i, f_path in enumerate(file_list):
        try:
            with h5py.File(f_path, "r") as h5:
                if prop_name in h5:
                    data = h5[prop_name][()]
                    # Handle scalars, 1D, and 2D by flattening
                    if isinstance(data, (np.ndarray, list)):
                        values.extend(np.array(data).flatten())
                    else:
                        values.append(data)
        except:
            continue  # Skip corrupted files

        if i % 500 == 0:
            my_bar.progress(i / total, text=f"{progress_text} ({i}/{total})")

    my_bar.empty()
    return np.array(values), None


# --- 4. EXECUTION AND VISUALIZATION ---
if selected_key:
    # Trigger the heavy lifting
    global_values, error_msg = collect_global_data(selected_key)

    if error_msg:
        st.error(error_msg)
    elif len(global_values) == 0:
        st.warning("No data found for this property.")
    else:
        # Show Statistics summary
        st.success(f"Analyzed {len(global_values)} data points.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Mean", f"{np.mean(global_values):.4f}")
        c2.metric("Median", f"{np.median(global_values):.4f}")
        c3.metric("Std Dev", f"{np.std(global_values):.4f}")

        # Histogram with marginal box plot
        fig = px.histogram(
            global_values,
            nbins=100,
            title=f"Global Distribution of {selected_key}",
            marginal="box",  # Adds a boxplot above the histogram
            color_discrete_sequence=["#1a73e8"],
        )
        # Update for 2026 Streamlit API
        st.plotly_chart(fig, width="stretch")
