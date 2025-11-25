import io
import base64
from typing import List, Dict

import pandas as pd
from PIL import Image
import streamlit as st
from openai import OpenAI

# ---------- CONFIG ----------

st.set_page_config(
    page_title="Market & Place AI Stylist",
    layout="wide",
)

client = OpenAI()

PRODUCT_FILE = "market_and_place_products.xlsx"
LOGO_FILE = "logo.png"
MARKET_AND_PLACE_URL = "https://marketandplace.co/"


# ---------- DATA LOADING ----------

@st.cache_data(show_spinner=False)
def load_products() -> pd.DataFrame:
    df = pd.read_excel(PRODUCT_FILE)

    # Normalise column names just in case
    df.columns = [c.strip() for c in df.columns]

    # Make sure required columns exist
    required = ["Product name", "Color", "Price", "raw_amazon", "Image URL:"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns in product file: {missing}")
    return df


products_df = load_products()


# ---------- PRODUCT FILTERING LOGIC ----------

def filter_products_for_query(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Very opinionated routing of queries to products:

    - If query mentions "luxury / hotel / spa" + towels -> prefer non-beach luxury towels.
    - If query mentions "beach / cabana" -> beach towels only.
    - If query mentions "towel" generically -> non-beach towels by default.
    - Otherwise, fall back to case-insensitive substring match on Product name with all query words.
    """
    q = query.lower()

    def name_contains(pattern: str) -> pd.Series:
        return df["Product name"].str.contains(pattern, case=False, na=False)

    # --- TOWELS LOGIC ---
    if "towel" in q:
        is_towel = name_contains("towel")
        is_beach = name_contains("beach")
        is_luxury_word = name_contains("luxury") | name_contains("hotel") | name_contains("spa")

        # 1) Luxury / hotel / spa towels (non-beach) when explicitly requested
        if any(w in q for w in ["luxury", "hotel", "spa"]):
            luxury_mask = is_towel & ~is_beach & is_luxury_word
            luxury_df = df[luxury_mask]
            if not luxury_df.empty:
                return luxury_df

        # 2) Explicit beach / cabana towels
        if any(w in q for w in ["beach", "cabana"]):
            beach_df = df[is_towel & is_beach]
            if not beach_df.empty:
                return beach_df

        # 3) Generic towels → default to non-beach bath / hand towels
        non_beach_df = df[is_towel & ~is_beach]
        if not non_beach_df.empty:
            return non_beach_df

        # Fallback: all towels
        towel_df = df[is_towel]
        if not towel_df.empty:
            return towel_df

    # --- OTHER SIMPLE CATEGORIES (sheets, quilts, etc.) ---
    if any(w in q for w in ["sheet", "sheets", "bedding", "duvet", "quilt", "comforter"]):
        is_sheet_like = (
            name_contains("sheet")
            | name_contains("bedding")
            | name_contains("duvet")
            | name_contains("quilt")
            | name_contains("comforter")
        )
        sheet_df = df[is_sheet_like]
        if not sheet_df.empty:
            return sheet_df

    # --- GENERIC KEYWORD MATCHING FALLBACK ---
    words = [w for w in q.replace(",", " ").split() if w]
    if words:
        mask = True
        for w in words:
            mask = mask & df["Product name"].str.contains(w, case=False, na=False)
        generic_df = df[mask]
        if not generic_df.empty:
            return generic_df

    # Last resort: return everything so we at least have something
    return df


def format_product_row(row: pd.Series) -> str:
    """Format a single product for chat text output."""
    price_str = f"${row['Price']}" if pd.notna(row["Price"]) else "Price: N/A"
    lines = [
        f"**{row['Product name']}**",
        f"- Color: {row['Color']}",
        f"- Price: {price_str}",
        f"- [View on Amazon]({row['raw_amazon']})",
    ]
    return "\n".join(lines)


# ---------- OPENAI HELPERS ----------

def call_stylist_model(user_query: str, products: List[Dict]) -> str:
    """
    Ask GPT to write a friendly explanation that ONLY references the products we pass in.
    No inventing extra products.
    """
    product_descriptions = []
    for p in products:
        product_descriptions.append(
            f"- {p['Product name']} (Color: {p['Color']}, Price: {p['Price']}, Amazon: {p['raw_amazon']})"
        )
    product_block = "\n".join(product_descriptions) if product_descriptions else "No products."

    system_msg = (
        "You are the Market & Place AI stylist. "
        "You MUST obey these rules:\n"
        "1. You are only allowed to recommend products from the list I give you.\n"
        "2. You are not allowed to invent or hallucinate new products, colors, or prices.\n"
        "3. If the products aren't a perfect match, explain that gracefully but still only talk about the products provided.\n"
        "4. Be concise, friendly, and practical."
    )

    user_msg = (
        f"Customer question: {user_query}\n\n"
        f"Here is the ONLY product list you are allowed to use:\n"
        f"{product_block}\n\n"
        "Write a short recommendation that picks a few of these that make the most sense "
        "for the question. Include name, color, price, and Amazon link for each."
    )

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


def generate_concept_image(
    mode: str,
    room_type: str,
    user_query: str,
    selected_products: List[Dict],
) -> str:
    """
    Generate a concept image and return the base64 PNG string.

    mode: "room" or "shelves"
    room_type: "bathroom", "bedroom", "store / showroom", ...
    selected_products: list of dicts from DataFrame rows
    """
    # Build a short description of the products so the model gets the vibe right.
    if selected_products:
        product_bits = []
        for p in selected_products[:5]:
            product_bits.append(
                f"{p['Product name']} in color {p['Color']}"
            )
        products_text = "; ".join(product_bits)
    else:
        products_text = "striped and solid cotton Market & Place textiles"

    # Extreme safety / anti-hallucination rules embedded in the prompt
    core_rules = """
ABSOLUTE, NON-NEGOTIABLE RULES (DO NOT BREAK):

1. You may ONLY depict textiles that could reasonably be Market & Place products:
   towels, bath rugs, shower curtains, bed sheets, pillowcases, and quilts
   with simple stripes or solids. Do NOT invent wild patterns, logos, or branding.
2. You are NOT allowed to show any additional decorative objects, furniture,
   plants, people, windows, art, mirrors, or props beyond what is required
   by the mode described below.
3. You MUST NOT show any packaging or brand names.
"""

    if mode == "shelves":
        # Store / showroom shelves view
        mode_rules = """
MODE: STORE SHELF / SHOWROOM

4. Show ONLY simple retail shelves and neatly folded stacks of towels
   (and optionally a matching folded rug or a small hanging towel).
5. DO NOT show bathtubs, showers, sinks, toilets, faucets, or vanities.
6. The focus is the shelves and the folded textiles. The background should be
   very plain and unobtrusive (like a neutral store wall or ceiling).
"""
        scene_desc = (
            f"A clean, modern store shelf / showroom view of folded towels that look like: {products_text}. "
            "The scene is bright, well lit, and photorealistic."
        )
    else:
        # Simple generic room concept (not editing a specific photo – API limitation)
        mode_rules = f"""
MODE: SIMPLE {room_type.upper()} CONCEPT

4. Show a very simple {room_type} with minimal fixtures.
5. Keep fixtures generic and unobtrusive so the focus is on towels and textiles.
6. Do NOT add any extra decorative objects beyond what is necessary.
"""
        scene_desc = (
            f"A photorealistic {room_type} concept that showcases textiles resembling: {products_text}. "
            "Keep layout simple and realistic, with the textiles clearly visible."
        )

    full_prompt = core_rules + mode_rules + "\nSCENE TO CREATE:\n" + scene_desc

    img_response = client.images.generate(
        model="dall-e-3",
        prompt=full_prompt,
        size="1024x1024",
        n=1,
    )

    return img_response.data[0].b64_json


# ---------- STREAMLIT STATE SETUP ----------

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "last_products_for_image" not in st.session_state:
    st.session_state["last_products_for_image"] = []


# ---------- LAYOUT: HEADER & LOGO ----------

try:
    logo_img = Image.open(LOGO_FILE)
    header_cols = st.columns([1, 3, 1])
    with header_cols[1]:
        st.image(logo_img, use_column_width=True)
except Exception:
    # Fallback: just show text if the logo can't be loaded for some reason
    st.markdown("## Market & Place")

st.markdown(
    "<h1 style='text-align: center; margin-top: 0;'>Market & Place AI Stylist</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='text-align: center;'>Chat with an AI stylist, search the Market & Place "
    "catalog, and generate concept visualizations using your own product file.</p>",
    unsafe_allow_html=True,
)

st.markdown(
    f"[← Return to Market & Place website]({MARKET_AND_PLACE_URL})",
)


# ---------- MAIN LAYOUT (TWO COLUMNS) ----------

left_col, right_col = st.columns([1.1, 1])

with left_col:
    st.subheader("Ask the AI stylist")

    user_query = st.text_input(
        "Describe what you're looking for:",
        key="mp_user_query",
        placeholder="e.g. luxury bath towels for a modern bathroom",
    )
    send_clicked = st.button("Send", key="mp_send_button")

    if send_clicked and user_query.strip():
        # Filter products by query
        matched_df = filter_products_for_query(products_df, user_query).copy()

        # Save some products for the image generator
        st.session_state["last_products_for_image"] = matched_df.head(10).to_dict(orient="records")

        # For the text reply we'll pass a smaller subset (to keep prompt size sane)
        subset_for_chat = matched_df.head(8).to_dict(orient="records")

        # If nothing matched at all, explain that
        if matched_df.empty:
            reply_text = (
                "I couldn't find anything in the current Market & Place product file that matches "
                "that request. You might want to try a different description or category."
            )
        else:
            reply_text = call_stylist_model(user_query, subset_for_chat)

        # Store in chat history
        st.session_state.chat_history.insert(
            0,
            {
                "query": user_query,
                "reply": reply_text,
            },
        )

    # Display chat history (newest first, under the input)
    for msg in st.session_state.chat_history:
        st.markdown(f"### 🧵 {msg['query']}")
        st.markdown(msg["reply"])
        st.markdown("---")

    # Quick catalog peek under the chat
    st.subheader("Quick catalog peek")
    peek_query = st.text_input(
        "Filter products by keyword:",
        key="mp_peek_query",
        placeholder="e.g. cabana stripe, queen sheet, luxury towel",
    )

    if peek_query.strip():
        peek_df = filter_products_for_query(products_df, peek_query).head(20)
    else:
        peek_df = products_df.head(20)

    for _, row in peek_df.iterrows():
        with st.container():
            cols = st.columns([1, 3])
            # Product thumbnail, if available
            img_url = row["Image URL:"]
            with cols[0]:
                if isinstance(img_url, str) and img_url.strip():
                    st.image(img_url, use_column_width=True)
            with cols[1]:
                st.markdown(f"**{row['Product name']}**")
                st.markdown(f"- Color: {row['Color']}")
                if pd.notna(row["Price"]):
                    st.markdown(f"- Price: ${row['Price']}")
                st.markdown(f"- [View on Amazon]({row['raw_amazon']})")
        st.markdown("---")


with right_col:
    st.subheader("🖼️ Your image")

    st.write(
        "Upload a photo of your room (bathroom, bedroom, etc.) or, if you're a distributor, "
        "leave it empty and generate a simple store shelf / showroom view."
    )

    room_type = st.selectbox(
        "Room type:",
        ["bathroom", "bedroom", "store / showroom", "living room"],
        index=0,
        key="mp_room_type",
    )

    view_mode = st.radio(
        "What do you want to visualize?",
        options=["In a simple room concept", "On store shelves / showroom"],
        index=0,
        key="mp_view_mode",
    )

    uploaded_image = st.file_uploader(
        "Optional: upload a reference photo of your room (JPEG/PNG). "
        "Current OpenAI image APIs can't directly edit your exact photo yet, "
        "but the reference can help you think about the space.",
        type=["jpg", "jpeg", "png"],
        key="mp_room_upload",
    )

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded room (reference)", use_column_width=True)

    if st.button("Generate concept image", key="mp_generate_image"):
        if not st.session_state["last_products_for_image"]:
            st.warning(
                "Ask the AI stylist a question first so I know which Market & Place products "
                "to base the image on."
            )
        else:
            with st.spinner("Generating image..."):
                mode = "room"
                if "shelf" in view_mode.lower() or "showroom" in view_mode.lower():
                    mode = "shelves"

                b64_png = generate_concept_image(
                    mode=mode,
                    room_type=room_type,
                    user_query=user_query,
                    selected_products=st.session_state["last_products_for_image"],
                )

                img_bytes = base64.b64decode(b64_png)
                img = Image.open(io.BytesIO(img_bytes))
                st.image(img, caption="AI-generated style concept", use_column_width=True)

                # Optional: allow download
                st.download_button(
                    "Download image",
                    data=img_bytes,
                    file_name="market_and_place_concept.png",
                    mime="image/png",
                )












