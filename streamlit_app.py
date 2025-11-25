import base64
import io
import os
from typing import List, Tuple

import pandas as pd
import streamlit as st
from openai import OpenAI

# ---------------- CONFIG ---------------- #

st.set_page_config(
    page_title="Market & Place AI Stylist",
    page_icon="🧵",
    layout="wide",
)

OPENAI_MODEL_VISION = "gpt-4.1-mini"
OPENAI_MODEL_IMAGE = "gpt-image-1"

client = OpenAI()

DATA_PATH = "market_and_place_products.xlsx"
LOGO_PATH = "logo.png"
SHELF_PHOTO_PATH = "store shelf.jpg"  # real Market & Place shelf photo


# ---------------- DATA LOADING ---------------- #

@st.cache_data(show_spinner=False)
def load_catalog(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    df = df.rename(
        columns={
            "Product name": "name",
            "Color": "color",
            "Price": "price",
            "raw_amazon": "amazon",
            "Image URL:": "image_url",
            "Category": "category",  # optional – only if you added it
        }
    )

    for col in ["name", "color", "amazon", "image_url", "category"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    df["name_lower"] = df["name"].str.lower()
    df["color_lower"] = df["color"].str.lower()
    df["category_lower"] = df.get("category", "").astype(str).str.lower()

    return df


catalog_df = load_catalog(DATA_PATH)


# ---------------- HELPER CONSTANTS / FUNCTIONS ---------------- #

CATEGORY_KEYWORDS = {
    "towel": ["towel", "bath towel", "hand towel"],
    "beach_towel": ["beach towel", "cabana"],
    "sheet": ["sheet set", "sheet"],
    "quilt": ["quilt", "coverlet"],
    "comforter": ["comforter", "duvet"],
    "bedding": ["quilt", "coverlet", "sheet", "comforter", "bedding"],
}

COLOR_WORDS = [
    "white",
    "ivory",
    "cream",
    "grey",
    "gray",
    "black",
    "navy",
    "blue",
    "aqua",
    "teal",
    "green",
    "yellow",
    "gold",
    "mustard",
    "red",
    "burgundy",
    "pink",
    "blush",
    "orange",
    "coral",
    "brown",
    "taupe",
    "beige",
]


def detect_category_terms(text: str) -> Tuple[List[str], List[str]]:
    """Simple keyword extractor for product type and color."""
    t = text.lower()
    found_cats: List[str] = []
    found_colors: List[str] = []

    if "beach towel" in t or ("beach" in t and "towel" in t):
        found_cats.append("beach_towel")
    elif "towel" in t:
        found_cats.append("towel")

    if "sheet" in t:
        found_cats.append("sheet")
    if "quilt" in t or "coverlet" in t:
        found_cats.append("quilt")
    if "comforter" in t or "duvet" in t:
        found_cats.append("comforter")
    if "bedding" in t or "bed set" in t:
        found_cats.append("bedding")

    for c in COLOR_WORDS:
        if c in t:
            found_colors.append(c)

    found_cats = list(dict.fromkeys(found_cats))
    found_colors = list(dict.fromkeys(found_colors))
    return found_cats, found_colors


def filter_catalog_by_query(df: pd.DataFrame, query: str, max_results: int = 8) -> pd.DataFrame:
    """Looser matching: categories + colors + fallback to substring search."""
    if not query:
        return df.head(max_results)

    q = query.lower()
    cat_terms, color_terms = detect_category_terms(q)

    mask = pd.Series(True, index=df.index)

    if cat_terms:
        cat_mask = pd.Series(False, index=df.index)
        for cat in cat_terms:
            keywords = CATEGORY_KEYWORDS.get(cat, [])
            for kw in keywords:
                cat_mask |= df["name_lower"].str.contains(kw, na=False)
        mask &= cat_mask

    if color_terms:
        color_mask = pd.Series(False, index=df.index)
        for c in color_terms:
            color_mask |= df["color_lower"].str.contains(c, na=False) | df[
                "name_lower"
            ].str.contains(c, na=False)

        narrowed = df[mask & color_mask]
        if not narrowed.empty:
            mask &= color_mask

    results = df[mask]

    if results.empty:
        simple_mask = pd.Series(False, index=df.index)
        for word in q.split():
            simple_mask |= df["name_lower"].str.contains(word, na=False)
        results = df[simple_mask]

    return results.head(max_results)


def render_product_card(row: pd.Series) -> None:
    cols = st.columns([1, 2.5])
    with cols[0]:
        if row.get("image_url") and str(row["image_url"]).startswith("http"):
            st.image(row["image_url"], use_column_width=True)
    with cols[1]:
        st.markdown(f"**{row['name']}**")
        st.write(f"• Color: {row['color']}")
        st.write(f"• Price: {row['price']}")
        if row.get("amazon"):
            st.markdown(f"[View on Amazon]({row['amazon']})")


def render_product_list(df: pd.DataFrame) -> None:
    if df.empty:
        st.info(
            "We couldn’t find matching products in the catalog for that request. "
            "Try adding a bit of detail, like *'navy striped bath towels'* or "
            "*'white quilt for queen bed'*."
        )
        return

    for _, row in df.iterrows():
        render_product_card(row)
        st.markdown("---")


def image_bytes_to_data_url(data: bytes, mime: str = "image/jpeg") -> str:
    """Convert raw image bytes to a data-URL string for the API."""
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


# ---------------- VISION + IMAGE GENERATION ---------------- #

def describe_photo_with_vision(image_bytes: bytes, user_request: str) -> str:
    """
    Use the vision model to get a short description of the uploaded room/shelf photo.
    The image must be passed as a data URL.
    """
    data_url = image_bytes_to_data_url(image_bytes)

    prompt = (
        "You are an interior stylist. Briefly describe the layout, style and key surfaces "
        "in this photo. Focus on elements that matter for placing new Market & Place textiles "
        "(towels, sheets, quilts, etc.). "
        "User request (what they want to change/add): "
        f"'{user_request}'. "
        "Keep it under 80 words."
    )

    resp = client.responses.create(
        model=OPENAI_MODEL_VISION,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": {"url": data_url}},
                ],
            }
        ],
    )

    try:
        return resp.output[0].content[0].text
    except Exception:
        return "A room or shelf scene suitable for styling with Market & Place textiles."


def generate_room_concept_image(room_bytes: bytes, textiles_request: str) -> bytes:
    """
    Edit the uploaded room photo, adding Market & Place products that match the request.
    """
    room_desc = describe_photo_with_vision(room_bytes, textiles_request)

    matching_products = filter_catalog_by_query(
        catalog_df, textiles_request or "", max_results=6
    )

    bullet_lines = []
    for _, r in matching_products.iterrows():
        bullet_lines.append(f"- {r['name']} (color: {r['color']})")
    product_snippet = "\n".join(bullet_lines) or "coordinated Market & Place textiles"

    data_url = image_bytes_to_data_url(room_bytes)

    prompt = f"""
You are an AI interior stylist for Market & Place.

Base photo description:
{room_desc}

Products to use as inspiration (Market & Place catalog):
{product_snippet}

User request for this room:
\"\"\"{textiles_request}\"\"\".

Instructions:
- Keep the existing room layout, furniture and architecture.
- Only add or change **textiles** (towels, bath mats, shower curtain, sheets, quilts, etc.).
- Use the colours and patterns from the products above, do NOT invent random designs.
- Make it look realistic and photographic.
"""

    img_resp = client.images.generate(
        model=OPENAI_MODEL_IMAGE,
        prompt=prompt,
        size="1024x1024",
        image=data_url,  # base image for editing
        n=1,
    )

    b64 = img_resp.data[0].b64_json
    return base64.b64decode(b64)


def generate_store_shelf_concept(textiles_request: str) -> bytes:
    """
    Edit the real store shelf photo, placing Market & Place products on it.
    """
    # Load the fixed shelf photo from the repo
    with open(SHELF_PHOTO_PATH, "rb") as f:
        shelf_bytes = f.read()

    shelf_desc = describe_photo_with_vision(shelf_bytes, textiles_request)

    matching_products = filter_catalog_by_query(
        catalog_df, textiles_request or "", max_results=6
    )

    bullet_lines = []
    for _, r in matching_products.iterrows():
        bullet_lines.append(
            f"- {r['name']} (color: {r['color']}, category: {r.get('category', '')})"
        )
    product_snippet = "\n".join(bullet_lines) or "Market & Place towels and bedding"

    data_url = image_bytes_to_data_url(shelf_bytes)

    prompt = f"""
You are generating a **retail store shelf** concept for Market & Place.

Base photo description:
{shelf_desc}

Products from the Market & Place catalog to place on this shelf:
{product_snippet}

Instructions:
- Keep the existing store interior and shelving structure exactly the same.
- Populate the shelves with neatly folded stacks and some hanging pieces,
  using only the colours, patterns and product types listed above.
- No people, no extra fixtures, no brand logos.
- The image must still clearly look like the same store shelf photo, just stocked.
"""

    img_resp = client.images.generate(
        model=OPENAI_MODEL_IMAGE,
        prompt=prompt,
        size="1024x1024",
        image=data_url,  # edit real shelf photo
        n=1,
    )

    b64 = img_resp.data[0].b64_json
    return base64.b64decode(b64)


# ---------------- HEADER ---------------- #

header_cols = st.columns([1, 2, 1])
with header_cols[1]:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=260)  # smaller logo, centred column
    st.markdown(
        "<h2 style='text-align:center; margin-bottom:0.25rem;'>Market & Place AI Stylist</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; font-size:0.95rem; margin-bottom:0.25rem;'>"
        "Chat with an AI stylist, search the Market & Place catalog, and generate "
        "concept visualizations using your own product file."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; font-size:0.9rem; margin-top:0;'>"
        "<a href='https://marketandplace.co/' style='text-decoration:none;'>"
        "← Return to Market & Place website</a></p>",
        unsafe_allow_html=True,
    )

st.markdown("---")


# ---------------- LAYOUT: LEFT / RIGHT ---------------- #

left_col, right_col = st.columns([1, 1], gap="large")

# ----- LEFT: ASK THE AI STYLIST + QUICK CATALOG PEEK ----- #

with left_col:
    st.subheader("Ask the AI stylist")

    with st.form("stylist_form", clear_on_submit=False):
        user_query = st.text_input(
            "Describe what you're looking for:",
            placeholder="e.g. luxury bath towels in grey, striped beach towels, queen quilt ideas",
        )
        submitted = st.form_submit_button("Send")

    if submitted and user_query.strip():
        st.markdown(f"### 🧵 {user_query.strip()}")
        matches = filter_catalog_by_query(catalog_df, user_query, max_results=6)
        render_product_list(matches)

    st.markdown("---")
    st.subheader("Quick catalog peek")

    peek_query = st.text_input(
        "Filter products by keyword:",
        placeholder="e.g. cabana stripe, flannel sheet, navy",
        key="peek_query",
    )

    if peek_query.strip():
        peek_matches = filter_catalog_by_query(catalog_df, peek_query, max_results=12)
        render_product_list(peek_matches)
    else:
        st.write("Type a keyword above to quickly peek at the catalog.")


# ----- RIGHT: AI CONCEPT VISUALIZER (ROOM + STORE SHELF) ----- #

with right_col:
    st.subheader("AI concept visualizer")

    mode = st.radio(
        "What do you want to visualize?",
        ["Room concept image", "Store shelf concept image"],
        horizontal=True,
    )

    # --- ROOM CONCEPT IMAGE --- #
    if mode == "Room concept image":
        st.markdown("### Room concept image")

        uploaded_room = st.file_uploader(
            "Upload a photo of your room (bathroom, bedroom, etc.):",
            type=["jpg", "jpeg", "png"],
        )

        textiles_request = st.text_input(
            "What textiles would you like to add or change?",
            placeholder="e.g. navy striped bath towels, white quilt, cabana stripe shower curtain",
            key="room_textiles_request",
        )

        if st.button("Generate room concept image"):
            if uploaded_room is None:
                st.error("Please upload a room photo first.")
            else:
                with st.spinner("Generating room concept…"):
                    room_bytes = uploaded_room.getvalue()
                    try:
                        img_bytes = generate_room_concept_image(
                            room_bytes, textiles_request or ""
                        )
                        st.image(img_bytes, use_column_width=True)
                        st.caption(
                            "Concept image generated by editing your uploaded room "
                            "photo using Market & Place catalog products."
                        )
                    except Exception as e:
                        st.error(f"Image generation failed: {e}")

    # --- STORE SHELF CONCEPT IMAGE --- #
    else:
        st.markdown("### Store shelf concept image")

        st.caption(
            "This uses your real Market & Place store shelf photo from the app repo, "
            "and virtually stocks it with Market & Place products."
        )

        textiles_request = st.text_input(
            "What do you want on the shelf?",
            placeholder="e.g. cabana stripe beach towels in aqua and navy, luxury bath towels, sheet sets",
            key="shelf_textiles_request",
        )

        # Show base shelf photo so users know what is being edited
        if os.path.exists(SHELF_PHOTO_PATH):
            st.markdown("**Base shelf photo (before styling):**")
            st.image(SHELF_PHOTO_PATH, use_column_width=True)

        if st.button("Generate store shelf concept image"):
            with st.spinner("Generating store shelf concept…"):
                try:
                    img_bytes = generate_store_shelf_concept(textiles_request or "")
                    st.markdown("**AI-generated shelf concept (after styling):**")
                    st.image(img_bytes, use_column_width=True)
                    st.caption(
                        "Concept image generated by editing the real Market & Place "
                        "store shelf photo using catalog products."
                    )
                except Exception as e:
                    st.error(f"Image generation failed: {e}")









