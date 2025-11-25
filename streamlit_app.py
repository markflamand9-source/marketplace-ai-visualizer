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

OPENAI_MODEL_VISION = "gpt-4.1-mini"
OPENAI_MODEL_IMAGE = "gpt-image-1"

client = OpenAI()

DATA_PATH = "market_and_place_products.xlsx"  # same folder as this file
LOGO_PATH = "logo.png"                        # same folder as this file


# ---------- DATA LOADING ----------

@st.cache_data(show_spinner=False)
def load_catalog(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    # Standardise column names just once
    df = df.rename(columns={
        "Product name": "name",
        "Color": "color",
        "Price": "price",
        "raw_amazon": "amazon",
        "Image URL:": "image_url"
    })

    # Safety: ensure strings
    for col in ["name", "color", "amazon", "image_url"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Pre-computed helper columns for simple keyword search
    df["name_lower"] = df["name"].str.lower()
    df["color_lower"] = df["color"].str.lower()

    return df


catalog_df = load_catalog(DATA_PATH)


# ---------- HELPER FUNCTIONS ----------

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
    "brown", "taupe", "beige"
]


def detect_category_terms(text: str) -> Tuple[List[str], List[str]]:
    """
    Very simple keyword extractor.
    Returns (category_terms, color_terms).
    """
    t = text.lower()
    found_cats = []
    found_colors = []

    # category by explicit word like "towels", "sheets", "quilt"
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

    # color words
    for c in COLOR_WORDS:
        if c in t:
            found_colors.append(c)

    # de-duplicate while preserving order
    found_cats = list(dict.fromkeys(found_cats))
    found_colors = list(dict.fromkeys(found_colors))

    return found_cats, found_colors


def filter_catalog_by_query(df: pd.DataFrame, query: str, max_results: int = 8) -> pd.DataFrame:
    """
    Looser matching:
    - First, filter by detected category (towels vs sheets vs quilts, etc.).
    - Then optionally narrow by detected color words.
    - If still nothing, fall back to substring match on product name.
    """
    if not query:
        return df.head(max_results)

    q = query.lower()
    cat_terms, color_terms = detect_category_terms(q)

    mask = pd.Series(True, index=df.index)

    # category constraints
    if cat_terms:
        cat_mask = pd.Series(False, index=df.index)
        for cat in cat_terms:
            keywords = CATEGORY_KEYWORDS.get(cat, [])
            for kw in keywords:
                cat_mask |= df["name_lower"].str.contains(kw, case=False, na=False)
        mask &= cat_mask

    # color constraints (soft — if they remove everything, skip them)
    if color_terms:
        color_mask = pd.Series(False, index=df.index)
        for c in color_terms:
            color_mask |= df["color_lower"].str.contains(c, case=False, na=False) | \
                          df["name_lower"].str.contains(c, case=False, na=False)
        narrowed = df[mask & color_mask]
        if not narrowed.empty:
            mask &= color_mask

    results = df[mask]

    # If still empty, fall back to general substring search over name
    if results.empty:
        simple_mask = pd.Series(False, index=df.index)
        for word in q.split():
            simple_mask |= df["name_lower"].str.contains(word, case=False, na=False)
        results = df[simple_mask]

    return results.head(max_results)


def render_product_card(row):
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


def generate_store_shelf_image(products: pd.DataFrame, description: str) -> bytes:
    """
    Use only Market & Place products as inspiration to generate a *retail store shelf* scene.
    Returns raw image bytes (PNG).
    """
    # Build a compact product summary for the prompt
    bullet_lines = []
    for _, r in products.iterrows():
        bullet_lines.append(
            f"- {r['name']} (color: {r['color']}, price: {r['price']})"
        )

    product_snippet = "\n".join(bullet_lines) if bullet_lines else "Market & Place towels and textiles."

    prompt = f"""
You are generating a **retail store shelf or aisle** concept image for Market & Place.

Scene requirements (very important):
- It must clearly be a store shelf / showroom, not a home bathroom or bedroom.
- Show long shelving bays with neatly folded stacks of textiles and some hanging pieces.
- Do NOT show bathtubs, sinks, toilets, beds, or home furniture.
- No people, no brand logos.

Use ONLY the following Market & Place products as inspiration for colors, patterns, and textures
(do not invent totally different products):

{product_snippet}

Customer request / styling direction:
\"\"\"{description}\"\"\".

Render a clean, well-lit store interior with rows of shelves and an end-cap, styled using those products.
"""

    img_resp = client.images.generate(
        model=OPENAI_MODEL_IMAGE,
        prompt=prompt,
        size="1024x1024",
        n=1,
    )

    b64 = img_resp.data[0].b64_json
    return base64.b64decode(b64)


# ---------- LAYOUT: HEADER ----------

header_cols = st.columns([1, 2, 1])
with header_cols[1]:
    # logo centered
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_column_width=False, width=400)
    st.markdown(
        "<h1 style='text-align:center; margin-bottom:0'>Market & Place AI Stylist</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;'>Chat with an AI stylist, search the Market & Place "
        "catalog, and generate concept visualizations using your own product file.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;'><a href='https://marketandplace.co/' "
        "style='text-decoration:none;'>← Return to Market & Place website</a></p>",
        unsafe_allow_html=True,
    )

st.markdown("---")


# ---------- MAIN LAYOUT (2 COLUMNS) ----------

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


# ----- RIGHT: AI CONCEPT VISUALIZER (ROOM SUGGESTIONS + STORE SHELF) -----

with right_col:
    st.subheader("AI concept visualizer")

    st.write(
        "Generate a **styled concept** using Market & Place products. "
        "You can either get product suggestions for a room, or generate a store shelf image."
    )

    mode = st.radio(
        "What do you want to visualize?",
        ["Room product suggestions", "Store shelf / showroom concept"],
        horizontal=True,
    )

    # --- ROOM PRODUCT SUGGESTIONS (no image generation, catalog only) ---
    if mode == "Room product suggestions":
        st.markdown("#### Room product suggestions")

        uploaded_room = st.file_uploader(
            "Optional: upload a photo of your room (bathroom, bedroom, etc.)",
            type=["jpg", "jpeg", "png"],
            help="The image is just for your reference – suggestions are catalog-only.",
        )

        textiles_request = st.text_input(
            "What textiles would you like to add or change?",
            placeholder="e.g. navy striped bath towels, white quilt, cabana stripe shower curtain",
            key="room_textiles_request",
        )

        if st.button("Get product suggestions"):
            st.markdown("### Suggested Market & Place products for your room")

            if uploaded_room is not None:
                st.caption("Reference photo uploaded – suggestions are still catalog-only, not image edits.")
                st.image(uploaded_room, use_column_width=True)

            suggestions = filter_catalog_by_query(
                catalog_df,
                textiles_request or "",
                max_results=8,
            )
            render_product_list(suggestions)

    # --- STORE SHELF / SHOWROOM IMAGE GENERATOR ---
    else:
        st.markdown("#### Store shelf / showroom concept")

        shelf_request = st.text_input(
            "Describe the shelf you want to visualize:",
            placeholder="e.g. cabana stripe beach towels in aqua and navy, folded stacks and matching mats",
            key="shelf_request",
        )

        if st.button("Generate store shelf concept image"):
            with st.spinner("Generating store shelf concept…"):
                # Choose products purely from the catalog, using the same looser matching
                shelf_products = filter_catalog_by_query(catalog_df, shelf_request or "", max_results=6)

                # Always generate an image, but make sure prompt uses only Market & Place products
                img_bytes = generate_store_shelf_image(shelf_products, shelf_request or "")

                st.markdown("### AI-generated store shelf concept")
                st.image(img_bytes, use_column_width=True)

                st.caption(
                    "Concept image generated from the Market & Place catalog. "
                    "Scene is intended as a store shelf / showroom, not a home bathroom."
                )

            # We no longer show the explicit 'products used as inspiration' list here,
            # per your request to keep the focus on the visual concept only.










