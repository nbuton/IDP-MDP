import streamlit as st
import plotly.express as px
import h5py
import os
import numpy as np

# --- CONFIGURATION ---
BASE_PATH = "data/IDRome/IDRome_v4"

st.set_page_config(layout="wide", page_title="IDRome Explorer", page_icon="🧬")


# --- PARSING LOGIC ---
def get_h5_path(prot_id):
    try:
        parts = prot_id.split("_")
        main_id = parts[0]
        suffix = "_".join(parts[1:])
        return os.path.join(
            BASE_PATH, main_id[0:2], main_id[2:4], main_id[4:6], suffix, "properties.h5"
        )
    except:
        return None


# --- SIDEBAR ---
with st.sidebar:
    st.title("Search")
    user_input = st.text_input(
        "Enter Protein ID", value="", placeholder="Q5SYC1_270_327"
    ).strip()

if not user_input:
    st.info("👋 Enter a Protein ID in the sidebar to visualize properties.")
    st.stop()

full_path = get_h5_path(user_input)

if full_path and os.path.exists(full_path):
    try:
        with h5py.File(full_path, "r") as h5:
            scalars, feat_1d, feat_2d = {}, {}, {}

            def visitor_func(name, node):
                if isinstance(node, h5py.Dataset):
                    shape = node.shape
                    if len(shape) == 0 or shape == (1,):
                        scalars[name] = node[()]
                    elif len(shape) == 1:
                        feat_1d[name] = node[:]
                    elif len(shape) == 2:
                        feat_2d[name] = node[:]

            h5.visititems(visitor_func)

            st.header(f"Protein Data: `{user_input}`")

            # --- 1. SCALARS SECTION (Multi-line Grid) ---
            if scalars:
                st.subheader("Scalar Properties")
                st.markdown(
                    """
                    <style>
                    .scalar-container { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px; }
                    .scalar-card { 
                        background-color: #ffffff; border: 1px solid #e6e9ef; border-radius: 8px; 
                        padding: 10px; width: 160px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center;
                    }
                    .scalar-label { color: #5f6368; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; overflow: hidden; }
                    .scalar-value { color: #1a73e8; font-size: 1rem; font-weight: bold; }
                    </style>
                """,
                    unsafe_allow_html=True,
                )

                html_blocks = ['<div class="scalar-container">']
                for name, val in scalars.items():
                    clean_name = name.split("/")[-1].replace("_", " ").title()
                    d_val = (
                        val.decode()
                        if isinstance(val, bytes)
                        else (
                            f"{val:.4f}"
                            if isinstance(val, (float, np.float32))
                            else str(val)
                        )
                    )
                    html_blocks.append(
                        f'<div class="scalar-card"><div class="scalar-label">{clean_name}</div><div class="scalar-value">{d_val}</div></div>'
                    )
                html_blocks.append("</div>")
                st.markdown("".join(html_blocks), unsafe_allow_html=True)

            st.divider()

            # --- 2. 1D & 2D SECTION ---
            col_left, col_right = st.columns([1, 1], gap="large")

            with col_left:
                st.subheader("1D Sequence Features")
                if feat_1d:
                    selection_1d = st.multiselect(
                        "Overlay features",
                        list(feat_1d.keys()),
                        default=list(feat_1d.keys())[:1],
                    )
                    fig_1d = px.line()
                    for key in selection_1d:
                        fig_1d.add_scatter(y=feat_1d[key], name=key)
                    # UPDATED: width='stretch' replaces use_container_width=True
                    st.plotly_chart(fig_1d, width="stretch")
                else:
                    st.info("No 1D data found.")

            with col_right:
                st.subheader("2D Matrices")
                if feat_2d:
                    selected_2d = st.selectbox("Choose Matrix", list(feat_2d.keys()))
                    fig_2d = px.imshow(
                        feat_2d[selected_2d], color_continuous_scale="Viridis"
                    )
                    # UPDATED: width='stretch'
                    st.plotly_chart(fig_2d, width="stretch")
                else:
                    st.info("No 2D data found.")

    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.error(f"File not found: `{full_path}`")
