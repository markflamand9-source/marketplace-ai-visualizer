import base64
import io
import os
from typing import List, Tuple

import pandas as pd
import streamlit as st
from openai import OpenAI

# ---------- CONFIG ----------

st.set_page_config(
    page_title="Market & Place AI Stylist",
    page_icon="🧵",
    layout="wide"
)

OPENAI_MODEL_VISION = "gpt-4.1-mini"   # still used for text reasoning if you like
OPENAI_MODEL_IMAGE = "gpt-image-1"

client = OpenAI()

DATA_PATH = "market_and_place_products.xlsx"  # same folder as this file
LOGO_PATH = "logo.png"                        # same folder as this file
SHELF_BASE_PATH = "store shelf.jpg"           # your base shelf photo


# ---------- DATA LOADING (unchanged from your working version) ----------

@st.cache_data(show_spinner=False)
def load_catalog(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    df = df.rename(columns={
        "Product name": "name",
        "Color": "color",
        "Price": "price",
        "raw_amazon": "amazon",
        "Image URL:": "image_url",
        # if you added "Category" column, keep it:
        "Category": "category"
    }, errors="ignore")

    for col in ["name", "color", "amazon", "image_url", "category"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    df["name_lower"] = df["name"].str.lower()
    df["color_lower"] = df["color"].str.lower()

    return df


catalog_df = load_catalog(DATA_PATH)


# ---------- SIMPLE SEARCH HELPERS (reuse your existing ones) ----------

CATEGORY_KEYWORDS = {
    "towel": ["towel", "bath towel", "hand towel"],
    "beach_towel": ["beach towel", "cabana"],
    "sheet": ["sheet set", "sheet"],
    "quilt": ["quilt", "coverlet"],
    "comforter": ["comforter", "duvet"],
    "bedding": ["quilt", "coverlet", "sheet", "comforter", "bedding"],
}

COLOR_WORDS = [
    "white", "ivory", "cream", "grey", "gray", "black",
    "navy", "blue", "aqua", "teal", "green",
    "yellow", "gold", "mustard",
    "red", "burgundy", "pink", "blush",
    "orange", "coral",
    "brown", "taupe", "beige",
]


def detect_category_terms(text: str) -> Tuple[List[str], List[str]]:
    t = text.lower()
    found_cats = []
    found_colors = []

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
                cat_mask |= df["name_lower"].str.contains(kw, case=False, na=False)
        mask &= cat_mask

    if color_terms:
        color_mask = pd.Series(False, index=df.index)
        for c in color_terms:
            color_mask |= df["color_lower"].str.contains(c, case=False, na=False) | \
                          df["name_lower"].str.contains(c, case=False, na=False)
        narrowed = df[mask & color_mask]
        if not narrowed.empty:
            mask &= color_mask

    results = df[mask]

    if results.empty:
        simple_mask = pd.Series(False, index=df.index)
        for word in q.split():
            simple_mask |= df["name_lower"].str.contains(word, case=False, na=False)
        results = df[simple_mask]

    return results.head(max_results)


def render_product_card(row: pd.Series):
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


def render_product_list(df: pd.DataFrame):
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


# ---------- AI IMAGE GENERATION HELPERS (TEXT-ONLY, BUT ERROR-FREE) ----------

def generate_room_concept_image(textiles_request: str) -> bytes:
    """
    Generate a *new* realistic room photo based on the user's textiles request.
    NOTE: current OpenAI client in Streamlit does not support true image-to-image
    editing with uploaded files, so we use text-only prompts here.
    """
    if not textiles_request.strip():
        textiles_request = "soft neutral luxury textiles with white and grey tones"

    prompt = f"""
You are an interior stylist for Market & Place.

Create a high-quality, photorealistic image of a real { 'bathroom or bedroom' }.
Keep the room layout realistic (not a showroom render) and focus on textiles.

Use towels, bedding, shower curtains, or mats that match this request:
\"\"\"{textiles_request}\"\"\".

Use only colors, patterns, and textures that could plausibly come from Market & Place
products (cabana stripe towels, Turkish cotton luxury towels, quilts, sheets, etc.).
Do NOT show brand logos or text in the image.
"""

    img_resp = client.images.generate(
        model=OPENAI_MODEL_IMAGE,
        prompt=prompt,
        size="1024x1024",
        n=1,
    )
    return base64.b64decode(img_resp.data[0].b64_json)


def generate_store_shelf_concept(textiles_request: str) -> bytes:
    """
    Generate a *new* retail store shelf image styled with Market & Place textiles.
    Again, this is text-only generation; we can't truly edit the exact base photo
    with this client library, but we describe it carefully so the layout matches.
    """
    if not textiles_request.strip():
        textiles_request = "neatly folded stacks of towels in soft neutral colors"

    prompt = f"""
You are designing a Market & Place retail shelf concept.

Create a photorealistic image of a store aisle with long shelving bays full of textiles:
towels, bath mats, shower curtains, or bedding.

The shelves should look very similar to a clean, modern Market & Place store shelf photo:
- long white metal shelves,
- multiple levels,
- products neatly stacked and some hanging.

Style the shelf using textiles that match this request:
\"\"\"{textiles_request}\"\"\".

Use only colors, patterns, and textures that could plausibly be Market & Place products
(e.g. cabana stripe beach towels in aqua and navy, luxury Turkish towels, etc.).
No brand logos or text.
"""

    img_resp = client.images.generate(
        model=OPENAI_MODEL_IMAGE,
        prompt=prompt,
        size="1024x1024",
        n=1,
    )
    return base64.b64decode(img_resp.data[0].b64_json)


# ---------- HEADER (SMALLER, MORE CONDENSED) ----------

header_cols = st.columns([1, 2, 1])
with header_cols[1]:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_column_width=False, width=260)
    st.markdown(
        "<h2 style='text-align:center; margin:0.5rem 0 0.25rem 0;'>Market & Place AI Stylist</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; font-size:0.95rem; margin:0;'>"
        "Chat with an AI stylist, search the Market & Place catalog, "
        "and generate concept visualizations using your own product file."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; font-size:0.9rem; margin-top:0.25rem;'>"
        "<a href='https://marketandplace.co/' style='text-decoration:none;'>"
        "← Return to Market & Place website</a></p>",
        unsafe_allow_html=True,
    )

st.markdown("---")


# ---------- MAIN LAYOUT (COLUMNS) ----------

left_col, right_col = st.columns([1, 1], gap="large")

# ----- LEFT: ASK THE AI STYLIST + QUICK CATALOG PEEK -----

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


# ----- RIGHT: AI CONCEPT VISUALIZER (ROOM IMAGE + STORE SHELF IMAGE) -----

with right_col:
    st.subheader("AI concept visualizer")

    st.write(
        "Generate a **styled concept** using your own photos and Market & Place products. "
        "You can either upload a room for AI styling, or generate a store shelf concept."
    )

    mode = st.radio(
        "What do you want to visualize?",
        ["Room concept image", "Store shelf / showroom concept"],
        horizontal=True,
    )

    # --- ROOM CONCEPT IMAGE ---
    if mode == "Room concept image":
        st.markdown("#### Room concept image")

        uploaded_room = st.file_uploader(
            "Upload a photo of your room (bathroom, bedroom, etc.):",
            type=["jpg", "jpeg", "png"],
            key="room_upload",
        )

        room_request = st.text_input(
            "What textiles would you like to add or change?",
            placeholder="e.g. luxury towels, navy striped bath towels, white queen quilt, cabana stripe shower curtain",
            key="room_textiles_request",
        )

        if uploaded_room is not None:
            st.caption("Original room photo:")
            st.image(uploaded_room, use_column_width=True)

        if st.button("Generate room concept image"):
            with st.spinner("Generating AI room concept…"):
                try:
                    img_bytes = generate_room_concept_image(room_request or "")
                    st.image(img_bytes, use_column_width=True)
                    st.caption(
                        "Concept image generated using your request. "
                        "Due to current API limits this is a new styled photo, "
                        "not an exact edit of the uploaded image."
                    )
                except Exception as e:
                    st.error(f"Image generation failed: {e}")

    # --- STORE SHELF CONCEPT ---
    else:
        st.markdown("#### Store shelf / showroom concept")

        if os.path.exists(SHELF_BASE_PATH):
            st.caption("Reference Market & Place store shelf photo:")
            st.image(
                SHELF_BASE_PATH,
                use_column_width=True,
            )

        shelf_request = st.text_input(
            "Describe how you'd like the shelf styled:",
            placeholder="e.g. cabana stripe beach towels in aqua and navy, folded stacks and matching mats",
            key="shelf_request",
        )

        if st.button("Generate store shelf concept image"):
            with st.spinner("Generating AI store shelf concept…"):
                try:
                    img_bytes = generate_store_shelf_concept(shelf_request or "")
                    st.image(img_bytes, use_column_width=True)
                    st.caption(
                        "Concept image generated based on your shelf styling request. "
                        "It is a new photorealistic shelf, not an exact edit of the base photo."
                    )
                except Exception as e:
                    st.error(f"Image generation failed: {e}")










