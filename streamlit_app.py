import os
import io
import base64

import streamlit as st
import pandas as pd
from PIL import Image
from openai import OpenAI


# ================== PAGE CONFIG ==================

st.set_page_config(
    page_title="Market & Place AI Stylist",
    layout="wide",
)

st.title("🧠🛋️ Market & Place AI Stylist")
st.write(
    "Chat with an AI stylist, search the Market & Place catalog, and generate "
    "concept visualizations using your own product file."
)


# ================== OPENAI CLIENT ==================

def get_openai_client() -> OpenAI:
    """Get OpenAI client using OPENAI_API_KEY from env or Streamlit secrets."""
    api_key = os.environ.get("OPENAI_API_KEY")

    # try Streamlit secrets if env var not set
    try:
        if not api_key and "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

    if not api_key:
        st.error(
            "❌ OPENAI_API_KEY not found.\n\n"
            "Go to Streamlit → **⋯ → Settings → Secrets** and add:\n\n"
            '`OPENAI_API_KEY = "sk-...."`'
        )
        st.stop()

    return OpenAI(api_key=api_key)


client = get_openai_client()


# ================== DATA LOADING ==================

@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_excel("market_and_place_products.xlsx")


try:
    df = load_data()
except Exception as e:
    st.error(
        "❌ Could not load `market_and_place_products.xlsx`.\n\n"
        "Make sure the file is in the repo root and has columns at least:\n"
        "`Product name`, `Color`, `Price`, `raw_amazon`, and optionally "
        "`Image URL:` and `Description` or `Product description`.\n\n"
        f"Technical details: {e}"
    )
    st.stop()

# normalise description column
if "Description" in df.columns:
    DESC_COL = "Description"
elif "Product description" in df.columns:
    DESC_COL = "Product description"
else:
    DESC_COL = None


# ================== PRODUCT HELPERS ==================

def find_relevant_products(query: str, max_results: int = 8) -> pd.DataFrame:
    """Simple keyword search over product name and color."""
    if not query:
        return df.head(max_results)

    q = query.lower()

    mask = (
        df["Product name"].fillna("").str.lower().str.contains(q)
        | df["Color"].fillna("").str.lower().str.contains(q)
    )

    results = df[mask].copy()

    # fallback so AI always has something
    if results.empty:
        results = df.sample(min(max_results, len(df)), random_state=0)

    return results.head(max_results)


def format_products_for_prompt(products: pd.DataFrame) -> str:
    """Compact text block describing products for the model."""
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


# ================== OPENAI CALLS ==================

def call_stylist_model(user_message: str, room_context: str,
                       products: pd.DataFrame) -> str:
    """Ask GPT-4.1-mini for styling suggestions based on the catalog."""
    product_block = format_products_for_prompt(products)
    room_part = room_context or "The user did not describe the room."

    system_prompt = """
You are an interior stylist working for the home-textile brand **Market & Place**.

You must follow these rules carefully:

- You ONLY recommend products from the product list provided.
- For each suggested product, you MUST clearly output:
  • Product name  
  • Color  
  • Price  
  • Short description  
  • Amazon URL (copy EXACTLY from the data, do not change, trim, or add any parameters)  
- Group recommendations into a short numbered list (3–5 items).
- Explain briefly *why* each product fits the user's room.
- Do NOT invent products or URLs that are not in the list.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"User request:\n{user_message}\n\n"
                f"Room description:\n{room_part}\n\n"
                "Here is the ONLY catalog you may use:\n\n"
                f"{product_block}"
            ),
        },
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=600,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error calling OpenAI chat model: `{e}`"


def generate_concept_image(room_description: str,
                           products: pd.DataFrame) -> Image.Image | None:
    """
    Generate a concept visualization of a room styled with Market & Place
    products. This is NOT a perfect edit of the uploaded photo; it’s a
    photoreal concept based on the text description + product vibe.
    """
    top_names = ", ".join(products["Product name"].head(3).tolist())
    prompt = (
        "Photorealistic interior design concept. "
        f"Room details: {room_description or 'no description given'}. "
        f"Style the space using textile products similar to: {top_names} "
        "from Market & Place. Soft natural lighting, cozy modern atmosphere."
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


# ================== STREAMLIT STATE ==================

if "messages" not in st.session_state:
    st.session_state.messages = []   # chat history

if "last_products" not in st.session_state:
    st.session_state.last_products = df.head(4)

if "room_description" not in st.session_state:
    st.session_state.room_description = ""


# ================== LAYOUT ==================

col_chat, col_side = st.columns([2, 1])

# ----- LEFT: CHAT -----
with col_chat:
    st.subheader("💬 Chat with the AI stylist")

    # display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input(
        "Ask for ideas (e.g. 'neutral queen bedding under $80 for a small bright room')"
    )

    if user_input:
        # show user message immediately
        st.session_state.messages.append({"role": "user", "content": user_input})

        # choose candidate products based on query
        candidate_products = find_relevant_products(user_input, max_results=8)
        st.session_state.last_products = candidate_products.copy()

        # use stored room description from the right column
        room_desc = st.session_state.room_description

        # call OpenAI
        reply = call_stylist_model(user_input, room_desc, candidate_products)

        st.session_state.messages.append({"role": "assistant", "content": reply})

        st.rerun()

# ----- RIGHT: ROOM + QUICK TOOLS -----
with col_side:
    st.subheader("🖼️ Your room")

    uploaded_image = st.file_uploader(
        "Upload a photo of your room (optional, for reference only)",
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

# ----- PRODUCT CARDS -----
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
                    # IMPORTANT: show Amazon URL EXACTLY as stored
                    st.markdown(f"[View on Amazon]({url})")

            st.markdown("---")
else:
    st.write("Ask the stylist a question to see recommendations here.")

# ----- CONCEPT VISUALIZER -----
st.subheader("🎨 AI concept visualizer")

st.write(
    "Generate a **concept image** of a room styled with Market & Place "
    "products based on your room description and the current recommended products."
)

if st.button("Generate concept image"):
    with st.spinner("Asking OpenAI to create a styled-room concept..."):
        concept_img = generate_concept_image(
            st.session_state.room_description,
            products,
        )
        if concept_img is not None:
            st.image(concept_img, caption="AI-generated style concept", use_column_width=False)
        else:
            st.warning("Could not generate an image. Check your API key and billing.")


