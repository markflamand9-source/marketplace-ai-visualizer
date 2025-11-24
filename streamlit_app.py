import os
import base64
import io

import streamlit as st
import pandas as pd
from PIL import Image
from openai import OpenAI


# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Market & Place AI Stylist",
    layout="wide",
)

st.title("🧠🛋️ Market & Place AI Stylist")
st.write(
    "Chat with an AI stylist, search the Market & Place catalog, and get suggestions "
    "for your room using your own product file."
)

# ---------- DATA LOADING ----------
@st.cache_data
def load_data():
    return pd.read_excel("market_and_place_products.xlsx")

try:
    df = load_data()
except Exception as e:
    st.error(
        "❌ Could not load `market_and_place_products.xlsx`.\n\n"
        "Make sure the file is in the repo root and has columns at least:\n"
        "`Product name`, `Color`, `Price`, `raw_amazon`, and optionally `Image URL:` and `Description`."
    )
    st.stop()

# Normalize possible description column names
if "Description" in df.columns:
    desc_col = "Description"
elif "Product description" in df.columns:
    desc_col = "Product description"
else:
    desc_col = None  # no description column


# ---------- OPENAI CLIENT ----------
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None


# ---------- PRODUCT SEARCH ----------
def find_relevant_products(user_query: str, max_results: int = 8) -> pd.DataFrame:
    """Very simple keyword search over product name and color."""
    if not user_query:
        return df.head(max_results)

    q = user_query.lower()

    mask = (
        df["Product name"].fillna("").str.lower().str.contains(q)
        | df["Color"].fillna("").str.lower().str.contains(q)
    )

    results = df[mask].copy()

    if results.empty:
        # Fallback so AI still has something to work with
        results = df.sample(min(max_results, len(df)), random_state=0)

    return results.head(max_results)


def format_products_for_prompt(products: pd.DataFrame) -> str:
    """Compact text version of product list for the AI."""
    lines = []
    for _, row in products.iterrows():
        desc = row.get(desc_col, "") if desc_col else ""
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


def call_ai_stylist(user_message: str, room_context: str, products: pd.DataFrame) -> str:
    """Chat completion that recommends ONLY from given products."""
    if client is None:
        return (
            "⚠️ AI is not configured.\n\n"
            "Set your OpenAI API key as `OPENAI_API_KEY` in the app secrets."
        )

    product_text = format_products_for_prompt(products)

    system_prompt = """
You are an interior stylist working for Market & Place.

RULES:
- You ONLY recommend products from the list provided to you.
- For each suggestion, clearly list:
  • Product name
  • Color
  • Price
  • Short description (from the data, or make a tasteful one if missing)
  • Amazon URL (copy EXACTLY from the data, do not change it at all)
- Group suggestions into a small, clear list (e.g. 3–5 items).
- Use friendly, concise language like you're chatting with a customer.
"""

    room_part = room_context or "The user did not describe the room."

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"User request:\n{user_message}\n\n"
                f"Room description:\n{room_part}\n\n"
                f"Here is the list of products you may choose from:\n\n{product_text}"
            ),
        },
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error calling AI: {e}"


def generate_concept_image(room_description: str, products: pd.DataFrame) -> Image.Image | None:
    """
    Generate a concept visualization (not an exact edit of the uploaded photo,
    but a photoreal render that matches the room + products).
    """
    if client is None:
        return None

    top_names = ", ".join(products["Product name"].head(3).tolist())
    prompt = (
        "Photoreal interior design concept. "
        f"Room details: {room_description or 'no description given'}. "
        f"Style it using textile products similar to: {top_names} from Market & Place. "
        "Soft natural lighting, high resolution."
    )

    try:
        img_resp = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024",
        )
        b64_data = img_resp.data[0].b64_json
        image_bytes = base64.b64decode(b64_data)
        img = Image.open(io.BytesIO(image_bytes))
        return img
    except Exception as e:
        st.warning(f"Could not generate concept image: {e}")
        return None


# ---------- STREAMLIT STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []  # chat history

if "last_products" not in st.session_state:
    st.session_state.last_products = df.head(4)  # default recommendations

if "room_description" not in st.session_state:
    st.session_state.room_description = ""


# ---------- LAYOUT ----------
col_chat, col_side = st.columns([2, 1])

# ====== LEFT: CHAT WITH AI STYLIST ======
with col_chat:
    st.subheader("💬 Chat with the AI stylist")

    # Show history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # New user message
    user_input = st.chat_input(
        "Ask for ideas (e.g. 'neutral queen bedding under $80 for a small bright room')"
    )

    if user_input:
        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Choose candidate products based on query
        candidate_products = find_relevant_products(user_input, max_results=8)
        st.session_state.last_products = candidate_products.copy()

        # Use stored room description from the right column
        room_desc = st.session_state.room_description

        # Call AI
        ai_reply = call_ai_stylist(user_input, room_desc, candidate_products)

        st.session_state.messages.append({"role": "assistant", "content": ai_reply})

        st.rerun()

# ====== RIGHT: ROOM + QUICK TOOLS ======
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
    st.subheader("🔎 Quick catalog peek (optional)")

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

    st.caption(
        "The AI stylist uses this same catalog behind the scenes when you chat."
    )

st.markdown("---")

# ====== RECOMMENDED PRODUCTS SECTION ======
st.subheader("⭐ AI-recommended products (from your catalog)")

products = st.session_state.last_products

if products is not None and not products.empty:
    for _, row in products.iterrows():
        with st.container():
            cols = st.columns([1, 3])
            # Image on left, info on right
            with cols[0]:
                img_url = row.get("Image URL:", "")
                if isinstance(img_url, str) and img_url.strip():
                    try:
                        st.image(img_url, use_column_width=True)
                    except Exception:
                        st.empty()
                else:
                    st.empty()

            with cols[1]:
                st.markdown(f"**{row.get('Product name', '')}**")
                st.markdown(f"- Color: **{row.get('Color', '')}**")
                st.markdown(f"- Price: **{row.get('Price', '')}**")

                if desc_col:
                    desc_val = row.get(desc_col, "")
                    if isinstance(desc_val, str) and desc_val.strip():
                        st.markdown(f"- Description: {desc_val}")

                url = row.get("raw_amazon", "")
                if isinstance(url, str) and url.strip():
                    # IMPORTANT: show Amazon URL exactly as-is
                    st.markdown(f"[View on Amazon]({url})")

            st.markdown("---")
else:
    st.write("No products selected yet. Ask the stylist a question to see suggestions here.")

# ====== CONCEPT VISUALIZER ======
st.subheader("🎨 AI concept visualizer")

st.write(
    "This generates a **concept image** of a room styled with Market & Place products "
    "based on your room description and the current recommended products."
)

if st.button("Generate concept image"):
    with st.spinner("Asking AI to create a styled-room concept..."):
        concept_img = generate_concept_image(st.session_state.room_description, products)
        if concept_img is not None:
            st.image(concept_img, caption="AI-generated style concept", use_column_width=False)
        else:
            st.warning("Could not generate an image. Check your API key and try again.")

