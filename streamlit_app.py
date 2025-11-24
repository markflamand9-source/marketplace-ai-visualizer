import os
import io
import requests

import streamlit as st
import pandas as pd
from PIL import Image

# ============= CONFIG =============

st.set_page_config(
    page_title="Market & Place AI Stylist (Free)",
    layout="wide",
)

st.title("🧠🛋️ Market & Place AI Stylist (Free)")
st.write(
    "Powered by free HuggingFace models. Ask for ideas, search the Market & Place "
    "catalog, and generate concept visualizations. "
    "Note: this uses small free models, so answers may be simpler than ChatGPT."
)

# HuggingFace setup
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")  # optional; public models work without it

# Text model: choose one that is definitely available on free HF Inference
HF_TEXT_MODEL = "gpt2"   # very small but reliably free/unguarded

# Image model: Stable Diffusion 2.1 (may require accepting license on HF once)
HF_IMAGE_MODEL = "stabilityai/stable-diffusion-2-1"


# ============= LOAD PRODUCT DATA =============

@st.cache_data
def load_data():
    return pd.read_excel("market_and_place_products.xlsx")

try:
    df = load_data()
except Exception as e:
    st.error(
        "❌ Could not load `market_and_place_products.xlsx`.\n\n"
        "Make sure the file is in the repo root and has columns at least:\n"
        "`Product name`, `Color`, `Price`, `raw_amazon`, optional `Image URL:` "
        "and optional `Description` or `Product description`."
    )
    st.stop()

# Normalize description column
if "Description" in df.columns:
    DESC_COL = "Description"
elif "Product description" in df.columns:
    DESC_COL = "Product description"
else:
    DESC_COL = None


# ============= HUGGINGFACE HELPERS =============

def hf_headers(accept_image: bool = False):
    headers = {}
    if HF_API_KEY:
        headers["Authorization"] = f"Bearer {HF_API_KEY}"
    if accept_image:
        headers["Accept"] = "image/png"
    return headers


def hf_text_completion(prompt: str) -> str:
    """
    Call a simple free text-generation model on HuggingFace (gpt2).
    This is not instruction-tuned, but with a good prompt it works ok.
    """
    url = f"https://api-inference.huggingface.co/models/{HF_TEXT_MODEL}"
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 200}}

    try:
        resp = requests.post(url, headers=hf_headers(), json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # HF returns list[{"generated_text": "..."}]
        if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
            return data[0]["generated_text"]
        return str(data)
    except Exception as e:
        return f"❌ Error from HuggingFace text model: {e}"


def hf_image_from_prompt(prompt: str) -> Image.Image | None:
    """
    Generate an image using Stable Diffusion via HF.
    This may require you to go to the HF model page and click 'Agree' once.
    """
    url = f"https://api-inference.huggingface.co/models/{HF_IMAGE_MODEL}"
    payload = {"inputs": prompt}

    try:
        resp = requests.post(url, headers=hf_headers(accept_image=True),
                             json=payload, timeout=120)
        resp.raise_for_status()
        img_bytes = resp.content
        img = Image.open(io.BytesIO(img_bytes))
        return img
    except Exception as e:
        st.warning(f"Could not generate concept image: {e}")
        return None


# ============= PRODUCT LOGIC =============

def find_relevant_products(user_query: str, max_results: int = 8) -> pd.DataFrame:
    """Basic keyword search over Product name + Color."""
    if not user_query:
        return df.head(max_results)

    q = user_query.lower()

    mask = (
        df["Product name"].fillna("").str.lower().str.contains(q)
        | df["Color"].fillna("").str.lower().str.contains(q)
    )

    results = df[mask].copy()
    if results.empty:
        results = df.sample(min(max_results, len(df)), random_state=0)

    return results.head(max_results)


def format_products_for_prompt(products: pd.DataFrame) -> str:
    """Compact text listing products for the model."""
    lines = []
    for _, row in products.iterrows():
        desc = row.get(DESC_COL, "") if DESC_COL else ""
        line = (
            f"- Name: {row.get('Product name', '')}\n"
            f"  Color: {row.get('Color', '')}\n"
            f"  Price: {row.get('Price', '')}\n"
            f"  Description: {desc}\n"
            f"  Amazon URL: {row.get('raw_amazon', '')}\n"
            f"  Image URL: {row.get('Image URL:', '')}"
        )
        lines.append(line)
    return "\n\n".join(lines)


def build_stylist_prompt(user_message: str, room_description: str,
                         products: pd.DataFrame) -> str:
    """
    Build one big prompt for gpt2.

    Because gpt2 isn't chat/instruction tuned, we include very explicit instructions
    and a desired output format.
    """
    product_block = format_products_for_prompt(products)
    room_part = room_description or "The user did not describe the room."

    prompt = f"""
You are an interior stylist for the home textile brand "Market & Place".

You must ONLY recommend products from the list below.

For each suggestion, output in THIS EXACT FORMAT:

1. Product name: ...
   Color: ...
   Price: ...
   Description: ...
   Amazon URL: ...
   Why it works: ...

2. Product name: ...
   ...

Use friendly, short sentences. Always copy the Amazon URL exactly as given.

User request:
{user_message}

Room description:
{room_part}

Available products:
{product_block}

Now give 3–5 numbered suggestions in the format above.
Answer:
"""
    return prompt.strip()


def concept_prompt(room_description: str, products: pd.DataFrame) -> str:
    """Prompt for Stable Diffusion concept image."""
    top_names = ", ".join(products["Product name"].head(3).tolist())
    return (
        "Photorealistic interior design concept, high resolution. "
        f"Room details: {room_description or 'no description given'}. "
        f"Style the room using textile products similar to: {top_names} from Market & Place. "
        "Soft natural light, cozy but modern, realistic colours."
    )


# ============= STREAMLIT STATE =============

if "last_products" not in st.session_state:
    st.session_state.last_products = df.head(4)

if "stylist_response" not in st.session_state:
    st.session_state.stylist_response = ""


# ============= LAYOUT =============

left, right = st.columns([2, 1])

# ----- LEFT side: AI stylist -----
with left:
    st.subheader("💬 Chat with the AI stylist")

    user_query = st.text_input(
        "What are you looking for?",
        placeholder="e.g. Neutral queen bedding under $80 for a bright small bedroom",
    )

    if st.button("Ask stylist"):
        products = find_relevant_products(user_query, max_results=8)
        st.session_state.last_products = products.copy()

        room_desc = st.session_state.get("room_description", "")
        prompt = build_stylist_prompt(user_query, room_desc, products)
        reply = hf_text_completion(prompt)
        st.session_state.stylist_response = reply

    if st.session_state.stylist_response:
        st.markdown("### Stylist suggestions")
        st.markdown(st.session_state.stylist_response)

# ----- RIGHT side: room + filters -----
with right:
    st.subheader("🖼️ Your room")

    uploaded_image = st.file_uploader(
        "Upload a photo of your room (optional)",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded room (reference)", use_column_width=True)

    room_description = st.text_area(
        "Describe your room & what you want:",
        placeholder="e.g. Small bedroom, white walls, light wood floors, want cozy neutral bedding...",
    )
    st.session_state["room_description"] = room_description

    st.markdown("---")
    st.subheader("🔎 Quick catalog peek")

    quick_query = st.text_input("Filter products by keyword")
    if quick_query:
        preview = find_relevant_products(quick_query, max_results=10)
    else:
        preview = df.head(10)

    st.dataframe(
        preview[["Product name", "Color", "Price", "raw_amazon"]],
        use_container_width=True,
        height=250,
    )

st.markdown("---")

# ----- AI-Recommended products section -----
st.subheader("⭐ AI-selected products (from your catalog)")

products = st.session_state.last_products

if products is not None and not products.empty:
    for _, row in products.iterrows():
        with st.container():
            cols = st.columns([1, 3])

            # image
            with cols[0]:
                img_url = row.get("Image URL:", "")
                if isinstance(img_url, str) and img_url.strip():
                    try:
                        st.image(img_url, use_column_width=True)
                    except Exception:
                        st.empty()
                else:
                    st.empty()

            # info
            with cols[1]:
                st.markdown(f"**{row.get('Product name', '')}**")
                st.markdown(f"- Color: **{row.get('Color', '')}**")
                st.markdown(f"- Price: **{row.get('Price', '')}**")

                if DESC_COL:
                    desc_val = row.get(DESC_COL, "")
                    if isinstance(desc_val, str) and desc_val.strip():
                        st.markdown(f"- Description: {desc_val}")

                url = row.get("raw_amazon", "")
                if isinstance(url, str) and url.strip():
                    # IMPORTANT: show Amazon URL exactly as stored
                    st.markdown(f"[View on Amazon]({url})")

            st.markdown("---")
else:
    st.write("Ask the stylist to see selected products here.")

# ----- Concept visualizer -----
st.subheader("🎨 AI concept visualizer (FREE)")

st.write(
    "Generate a **concept image** of a room styled with Market & Place products, "
    "based on your room description and currently selected products. "
    "Uses Stable Diffusion 2.1 via HuggingFace."
)

if st.button("Generate concept image"):
    prompt = concept_prompt(st.session_state.get("room_description", ""), products)
    with st.spinner("Generating concept image..."):
        img = hf_image_from_prompt(prompt)
        if img is not None:
            st.image(img, caption="AI-generated style concept", use_column_width=False)
        else:
            st.warning("Could not generate an image. If this persists, you may need to "
                       "go to the HuggingFace page for the model and click 'Agree' on the license.")

