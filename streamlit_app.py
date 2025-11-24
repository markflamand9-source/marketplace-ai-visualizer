import streamlit as st
import pandas as pd

st.set_page_config(page_title="Market & Place Visualizer", layout="wide")

st.title("🛍️ Market & Place Product Explorer")
st.write("Search your Market & Place product file below.")

@st.cache_data
def load_data():
    return pd.read_excel("market_and_place_products.xlsx")

# Try loading the data
try:
    df = load_data()
except Exception:
    st.error("❌ Upload `market_and_place_products.xlsx` to your GitHub repo.")
    st.stop()

st.subheader("Search Products")

search = st.text_input("Search by product name or keyword")
color = st.text_input("Filter by color (optional)")

filtered = df.copy()

if search:
    filtered = filtered[
        filtered["Product name"].str.contains(search, case=False, na=False)
    ]

if color:
    filtered = filtered[
        filtered["Color"].str.contains(color, case=False, na=False)
    ]

st.write("### Results")
st.dataframe(filtered[["Product name", "Color", "Price", "raw_amazon"]])
