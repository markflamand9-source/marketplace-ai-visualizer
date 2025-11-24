import os
import io
import base64
import requests

import streamlit as st
import pandas as pd
from PIL import Image

# ==================== CONFIG ====================

st.set_page_config(
    page_title="Market & Place AI Stylist (Free)",
    layout="wide",
)

st.title("🧠🛋️ Market & Place AI Stylist (Free)")
st.write(
    "Powered by free HuggingFace models. Chat with an AI stylist, search the "
    "Market & Place catalog, and generate concept visualizations."
)

HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")  # set this in Streamlit secrets

# Text model (chat/styling) – free tier
HF_TEXT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
# Image model (visualizer) – free tier
HF_IMAGE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


# ==================== DATA LOADING ====================

@st.cache_data
def load_data():
    return pd.read_excel("market_and_place_products.xlsx")

try:
    df = load_data()
except Exception as e:
    st.error(
        "❌ Could not load `market_and_place_products.xlsx`.\n\n"
        "Make sure the file is in the repo root and has columns at least:\n"
        "`Product name`, `Color`, `Price`, `raw_amazon`, and optionally "
        "`Image URL:` and `Description`."
    )
    st.stop()

# Normalise description column
if "Description" in df.columns:
    DESC_COL = "Description"
elif "Product description" in df.columns:
    DESC_COL = "Product description"
else:
    DESC_COL = None


# ==================== HUGGINGFACE HELPERS ====================

def hf_text_completion(prompt: str) -> str:
    """Call HuggingFace text model (Llama-3) to get a response."""
    if not HF_API_KEY:
        return (
            "⚠️ HUGGINGFACE_API_KEY is not set. "
            "Add it in Streamlit → Settings → Secrets."
        )

    url = f"https://api-inference.huggingface.co/models/{HF_TEXT_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 400,
            "temperature": 0.7,
            "return_full_text": False,
        },
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # HF sometimes returns list[ { "generated_text": ... } ]
        if isinstance(data, list) and len(data) > 0:
            item = data[0]
            if isinstance(item, dict) and "generated_text" in item:
                return item["generated_text"]

        # Fallback: just convert to string
        return str(data)

    except Exception as e:
        return f"❌ Error from HuggingFace text model: {e}"


def hf_image_from_prompt(prompt: str) -> Image.Image | None:
    """Generate an image using Stable Diffusion XL via HuggingFace."""
    if not HF_API_KEY:
        st.warning(
            "HUGGINGFACE_API_KEY is not set. Cannot generate images."
        )
        return None

    url = f"https://api-inference.huggingface.co/models/{HF_IMAGE_MODEL}"
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Accept": "image/png",
    }
    payload = {"inputs": prompt}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        img_bytes = resp.content
        img = Image.open(io.BytesIO(img_bytes))
        return img
    except Exception as e:
        st.warning(f"Could not generate concept image: {e}")
        return None


# ==================== PRODUCT LOGIC ====================

def find_relevant_products(user_query: str, max_results: int = 8) -> pd.DataFrame:
    """Simple keyword search over Product name and Color."""
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
    """Compact text representation of products for the model."""
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


def ai_stylist_reply(user_message: str, room_context: str, products: pd.DataFrame) -> str:
    """Use HF text model as stylist – ONLY recommends from provided products."""
    product_text = format_products_for_prompt(products)

    system_instructions = """
You are an interior stylist for Market & Place.

RULES:
- You ONLY recommend products from the list provided.
- For each suggestion, clearly show:
  • Product name
  • Color
  • Price
  • Short description (based on the data; invent tasteful text if missing)
  • Amazon URL (copy EXACTLY as given, do not modify)
- Group suggestions into a short, numbered list (3–5 items).
- Use friendly, concise language as if chatting with a customer.
"""

    room_part = room_context or "The user did not describe the room."

    prompt = (
        system_instructions
        + "\n\n"
        + "User request:\n"
        + user_message
        + "\n\nRoom description:\n"
        + room_part
        + "\n\nYou may ONLY choose from these products:\n\n"
        + product_text
        + "\n\nNow respond with your suggestions."
    )

    return hf_text_completion(prompt)


def concept_prompt(room_description: str, products: pd.DataFrame) -> str:
    """Build a prompt for Stable Diffusion XL."""
    top_names = ", ".join(products["Product name"].head(3).tolist())
    return (
        "Photorealistic interior design concept, high resolution. "
        f"Room details: {room_description or 'no description given'}. "
        f"Style the room using textile products similar to: {top_names} from Market & Place. "
        "Soft natural light, cozy but modern, realistic colours."
    )


# ==================== STREAMLIT STATE ====================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_products" not in st.session_state:
    st.session_state.last_products = df.head(4)

if "room_description" not in st.session_state:
    st.session_state.room_description = ""


# ==================== LAYOUT ====================

col_chat, col_side = st.columns([2, 1])

# ----- LEFT: CHAT -----
with col_chat:
    st.subheader("💬 Chat with the AI stylist")

    # show history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input(
        "Ask for ideas (e.g. 'neutral queen bedding under $80 for a small bright room')"
    )

    if user_input:
        # store user msg
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        # choose products
        candidate_products = find_relevant_products(user_input, max_results=8)
        st.session_state.last_products = candidate_products.copy()

        # room context
        room_desc = st.session_state.room_description

        # AI reply
        reply = ai_stylist_reply(user_input, room_desc, candidate_products)

        st.session_state.messages.append(
            {"role": "assistant", "content": reply}
        )

        st.rerun()

# ----- RIGHT: ROOM + QUICK FILTER -----
with col_side:
    st.subheader("🖼️ Your room")

    uploaded_image = st.file_uploader(
        "Upload a photo of your room (optional)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded room (reference)", use_column_width=True)

    room_description = st.text_area(
        "Describe your room & what you want:",
        value=st.session_state.room_description,
        placeholder="e.g. Small bedroom, white walls, light wood floors, want cozy neutral bedding...",
    )
    st.session_state.room_description = room_description

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

# ----- AI-RECOMMENDED PRODUCTS -----
st.subheader("⭐ AI-recommended products (from your catalog)")

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

            # text info
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
                    # IMPORTANT: show Amazon URL EXACTLY as stored
                    st.markdown(f"[View on Amazon]({url})")

            st.markdown("---")
else:
    st.write("Ask the stylist something to see recommendations here.")

# ----- CONCEPT VISUALIZER -----
st.subheader("🎨 AI concept visualizer (FREE)")

st.write(
    "Generates a **concept image** of a room styled with Market & Place products "
    "based on your room description and current recommended products. "
    "This uses free Stable Diffusion XL via HuggingFace."
)

if st.button("Generate concept image"):
    prompt = concept_prompt(st.session_state.room_description, products)
    with st.spinner("Generating concept image with Stable Diffusion XL..."):
        img = hf_image_from_prompt(prompt)
        if img is not None:
            st.image(img, caption="AI-generated style concept", use_column_width=False)
        else:
            st.warning("Could not generate an image. Check your HuggingFace token.")

