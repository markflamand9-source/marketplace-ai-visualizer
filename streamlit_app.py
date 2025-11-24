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

# ---------- Centered logo styling ----------
st.markdown(
    """
    <style>
        .center-logo-wrapper {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: -30px;
            margin-bottom: 10px;
        }
        .center-logo-wrapper img {
            width: 55%;
            max-width: 650px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Logo ----------
st.markdown(
    """
    <div class="center-logo-wrapper">
        <img src="logo.png" alt="Market & Place logo">
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Title & subtitle ----------
st.title("Market & Place AI Stylist")
st.write(
    "Chat with an AI stylist, search the Market & Place catalog, and generate concept "
    "visualizations using your own product file."
)

# ---------- Return to main website ----------
st.markdown(
    "[← Return to Market & Place website](https://marketandplace.co/)"
)


# ================== OPENAI CLIENT ==================

def get_openai_client() -> OpenAI:
    """Get OpenAI client using OPENAI_API_KEY from env or Streamlit secrets."""
    api_key = os.environ.get("OPENAI_API_KEY")

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
    """Load Market & Place catalog from Excel in the repo."""
    return pd.read_excel("market_and_place_products.xlsx")


try:
    df = load_data()
except Exception as e:
    st.error(
        "❌ Could not load `market_and_place_products.xlsx`.\n\n"
        "Make sure the file is in the repo root and has these columns:\n"
        "`Product name`, `Color`, `Price`, `raw_amazon`, `Image URL:`.\n\n"
        f"Technical details: {e}"
    )
    st.stop()

DESC_COL = None  # no dedicated description column in the file


# ================== PRODUCT HELPERS ==================

def find_relevant_products(query: str, max_results: int = 6) -> pd.DataFrame:
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
        line = (
            f"- Name: {row.get('Product name', '')}\n"
            f"  Color: {row.get('Color', '')}\n"
            f"  Price: {row.get('Price', '')}\n"
            f"  Amazon URL: {row.get('raw_amazon', '')}"
        )
        lines.append(line)
    return "\n\n".join(lines)


# ================== OPENAI CALLS (TEXT) ==================

def call_stylist_model(user_message: str, room_context: str,
                       products: pd.DataFrame) -> str:
    """
    Ask GPT-4.1-mini for styling suggestions based on the catalog.

    STRICT / NUCLEAR RULES:
    - It may ONLY recommend products from the provided product list.
    - It MUST NOT invent new products, colors, prices, or URLs.
    - If it cannot find something suitable, it must say so.
    """

    product_block = format_products_for_prompt(products)
    room_part = room_context or "The user did not describe the room."

    system_prompt = """
You are an interior stylist working exclusively for the home-textile brand **Market & Place**.

NUCLEAR RULES — you MUST follow ALL of these:

1. You may ONLY recommend products that appear in the catalog block I give you.
2. You MUST NOT invent, imagine, or guess:
   - new product names
   - new colors
   - new prices
   - new Amazon URLs
   - new product lines or brands
3. Every recommended item MUST include:
   - Product name (exact from catalog)
   - Color (exact from catalog)
   - Price (exact from catalog)
   - 1–2 sentence reason why it fits
   - Amazon URL (copied EXACTLY from catalog, no edits)
4. Group recommendations as a short numbered list (3–5 items).
5. If the catalog block does not contain anything suitable, say:
   - "I’m not allowed to recommend products outside the Market & Place catalog for this query."
   and STOP. Do NOT make anything up.
6. Do NOT talk about other brands, stores, or products.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"User request:\n{user_message}\n\n"
                f"Room description:\n{room_part}\n\n"
                "HERE IS THE ONLY CATALOG YOU ARE ALLOWED TO USE:\n\n"
                f"{product_block}"
            ),
        },
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.4,
            max_tokens=800,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error calling OpenAI chat model: `{e}`"


# ================== OPENAI CALLS (IMAGES) ==================

def generate_concept_image(
    mode: str,
    room_description: str,
    products: pd.DataFrame,
    room_image_bytes: bytes | None = None,
) -> Image.Image | None:
    """
    Generate a concept visualization.

    mode == "room":   style the user's room (bathroom stays bathroom, etc.)
    mode == "shelf":  show products as if displayed on retail store shelves.

    HARD / NUCLEAR CONSTRAINTS (best-effort — enforced via prompt):

    - NEVER invent non–Market & Place products.
    - Use ONLY the provided product list as inspiration for colors/patterns.
    - For ROOM mode:
        * Keep the same room TYPE (bathroom stays bathroom, bedroom stays bedroom).
        * Keep architecture and hard surfaces: walls, floor, windows, doors,
          mirrors, vanity, toilet, shower, tub, cabinets, shelves, lights, etc.
        * ONLY change TEXTILES:
          towels, bath mats, rugs, curtains, shower curtains, bedding, throws, cushions.
        * Do NOT move furniture, change layout, or add new furniture pieces.
        * If unsure, keep the original surfaces and only overlay new textiles.
    - If these constraints cannot be respected, the model should return a
      very minimal change instead of redesigning the whole space.
    """

    # Prepare a compact product summary for the image model
    top_products = products.head(6).copy()
    product_summaries = []
    for _, row in top_products.iterrows():
        product_summaries.append(
            f"{row.get('Product name','')} (Color: {row.get('Color','')}, "
            f"Price: {row.get('Price','')})"
        )
    product_text = "; ".join(product_summaries) or "Market & Place textiles."

    if mode == "shelf":
        base_instruction = (
            "Create a photorealistic image of a retail store shelf or aisle, "
            "showing Market & Place products neatly displayed. "
            "Use ONLY these Market & Place textiles as inspiration for color and pattern: "
            f"{product_text}. "
            "Do not show any other brands or packaging. "
            "The overall feeling should be clean, modern, and inviting, as if this were "
            "a professional product photo for a catalog."
        )
    else:
        # ROOM styling mode
        base_instruction = (
            "You are restyling an EXISTING ROOM image using Market & Place textiles only.\n"
            "NUCLEAR ROOM RULES (DO NOT BREAK THESE):\n"
            "1. Keep the SAME room type as the uploaded photo (bathroom stays bathroom, "
            "bedroom stays bedroom, etc.).\n"
            "2. KEEP all hard surfaces and layout exactly the same: walls, floors, windows, "
            "doors, ceiling, vanities, shower glass, tubs, mirrors, toilets, cabinets, "
            "shelves, lighting, furniture frames.\n"
            "3. ONLY modify textiles: towels, bath mats, rugs, shower curtains, curtains, "
            "bedding, throws, blankets, and cushions.\n"
            "4. DO NOT move or replace the vanity, shower, tub, toilet, mirror, shelves, "
            "tables, bed frame, nightstands, or any large furniture.\n"
            "5. DO NOT change the architecture, wall color, or flooring material.\n"
            "6. Use ONLY Market & Place textiles as inspiration. No other brands, "
            "no invented products.\n"
            "7. If you are not sure how to change something without breaking these rules, "
            "leave it as-is.\n\n"
            "STYLE BRIEF:\n"
            f"{room_description or 'Use a tasteful, modern, cozy style.'}\n\n"
            "TEXTILE PALETTE (Market & Place catalog):\n"
            f"{product_text}\n"
        )

    # We cannot truly 'edit' the user image with this endpoint, but we can provide
    # a strong textual instruction so the generated image stays as faithful as possible.
    if room_image_bytes is not None and mode == "room":
        base_instruction += (
            "\nAssume the user has uploaded a reference photo. "
            "Your job is to imagine that SAME room with only the textiles swapped "
            "to the Market & Place products. Keep camera angle, walls, layout, and "
            "furniture essentially unchanged."
        )

    try:
        img_resp = client.images.generate(
            model="gpt-image-1",
            prompt=base_instruction,
            size="1024x1024",
        )
        b64_data = img_resp.data[0].b64_json
        image_bytes = base64.b64decode(b64_data)
        img = Image.open(io.BytesIO(image_bytes))
        return img

    except Exception as e:
        st.warning(
            "Could not generate concept image. This can be due to org verification, "
            "billing, or API limits.\n\n"
            f"Details: {e}"
        )
        return None


# ================== STREAMLIT STATE ==================

if "messages" not in st.session_state:
    st.session_state.messages = []   # chat history

if "last_products" not in st.session_state:
    st.session_state.last_products = df.head(4)

if "room_description" not in st.session_state:
    st.session_state.room_description = ""

if "uploaded_room_image_bytes" not in st.session_state:
    st.session_state.uploaded_room_image_bytes = None

if "concept_image_bytes" not in st.session_state:
    st.session_state.concept_image_bytes = None

if "visual_mode" not in st.session_state:
    st.session_state.visual_mode = "Your room"


# ================== LAYOUT ==================

col_left, col_right = st.columns([2.1, 1.4])


# ================== LEFT COLUMN: CHAT + CATALOG ==================
with col_left:
    st.subheader("Ask the AI stylist")

    user_input = st.text_input(
        "Describe what you're looking for:",
        placeholder="e.g. neutral queen bedding under $80 for a bright room",
        key="stylist_query",
    )

    send_clicked = st.button("Send", key="send_button")

    # When user sends a query
    if send_clicked and user_input.strip():
        # Store user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Pick candidate products
        candidate_products = find_relevant_products(user_input, max_results=6)
        st.session_state.last_products = candidate_products.copy()

        room_desc = st.session_state.room_description

        # AI reply
        reply = call_stylist_model(user_input, room_desc, candidate_products)
        st.session_state.messages.append({"role": "assistant", "content": reply})

    # Chat history
    st.subheader("Chat with the AI stylist")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Recommended products (cards) for THIS conversation
    products = st.session_state.last_products
    if products is not None and not products.empty:
        st.markdown("#### Recommended products for this conversation")

        for _, row in products.iterrows():
            with st.container():
                c_img, c_info = st.columns([1, 3])

                # image (from Image URL:)
                with c_img:
                    img_url = row.get("Image URL:", "")
                    if isinstance(img_url, str) and img_url.strip():
                        try:
                            st.image(img_url, use_column_width=True)
                        except Exception:
                            st.empty()
                    else:
                        st.empty()

                # info
                with c_info:
                    st.markdown(f"**{row.get('Product name', '')}**")
                    st.markdown(f"- Color: **{row.get('Color', '')}**")
                    st.markdown(f"- Price: **{row.get('Price', '')}**")

                    if DESC_COL:
                        desc_val = row.get(DESC_COL, "")
                        if isinstance(desc_val, str) and desc_val.strip():
                            st.markdown(f"- Description: {desc_val}")

                    url = row.get("raw_amazon", "")
                    if isinstance(url, str) and url.strip():
                        st.markdown(f"[View on Amazon]({url})")

            st.markdown("---")

    # Quick catalog peek (now under chat, scrolls with it)
    st.subheader("🔎 Quick catalog peek")

    quick_query = st.text_input(
        "Filter products by keyword",
        key="quick_query",
        placeholder="e.g. towel, bath, queen, flannel...",
    )
    if quick_query:
        preview = find_relevant_products(quick_query, max_results=12)
    else:
        preview = df.head(12)

    st.dataframe(
        preview[["Product name", "Color", "Price", "raw_amazon"]],
        use_container_width=True,
        height=260,
    )


# ================== RIGHT COLUMN: IMAGE TOOLS ==================
with col_right:
    st.subheader("🖼️ Your image")

    uploaded_image = st.file_uploader(
        "Upload a photo of your room (bathroom, bedroom, etc.) "
        "or leave empty if you just want a store shelf visualization.",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded_image is not None:
        img_bytes = uploaded_image.getvalue()
        st.session_state.uploaded_room_image_bytes = img_bytes
        st.image(uploaded_image, caption="Uploaded room", use_column_width=True)

    room_description = st.text_area(
        "Describe your room & what you want:",
        value=st.session_state.room_description,
        placeholder="e.g. Small bathroom, white tiles, want colorful towels and bath mat...",
    )
    st.session_state.room_description = room_description

    st.markdown("---")
    st.subheader("🎨 AI concept visualizer")

    st.write(
        "Generates a **styled concept** using Market & Place products. "
        "For rooms, the AI is strictly instructed to keep the architecture "
        "and layout the same and ONLY change textiles. "
        "You can also preview what the products might look like on **retail shelves**."
    )

    visual_mode = st.radio(
        "What would you like to visualize?",
        ["Your room", "Retail store shelves"],
        key="visual_mode_radio",
    )
    st.session_state.visual_mode = visual_mode

    if st.button("Generate concept image"):
        mode_key = "room" if visual_mode == "Your room" else "shelf"

        with st.spinner("Asking OpenAI to generate a strict, catalog-only concept..."):
            products_for_image = st.session_state.last_products
            img = generate_concept_image(
                mode=mode_key,
                room_description=st.session_state.room_description,
                products=products_for_image,
                room_image_bytes=st.session_state.uploaded_room_image_bytes,
            )
            if img is not None:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                st.session_state.concept_image_bytes = buf.getvalue()

    if st.session_state.concept_image_bytes is not None:
        st.image(
            st.session_state.concept_image_bytes,
            caption=(
                "AI-generated style concept "
                f"({'room' if st.session_state.visual_mode == 'Your room' else 'retail shelves'})"
            ),
            use_column_width=True,
        )

        # Show which products were used as inspiration
        products_for_image = st.session_state.last_products
        if products_for_image is not None and not products_for_image.empty:
            st.markdown("##### Products used as inspiration for this concept")
            for _, row in products_for_image.iterrows():
                name = row.get("Product name", "")
                color = row.get("Color", "")
                url = row.get("raw_amazon", "")
                line = f"- **{name}** — *{color}*"
                if isinstance(url, str) and url.strip():
                    line += f" — [View on Amazon]({url})"
                st.markdown(line)












