import base64
import os
from typing import List, Tuple, Dict

import pandas as pd
import streamlit as st
from openai import OpenAI

# ---------- CONFIG ----------

st.set_page_config(
    page_title="Market & Place AI Stylist",
    page_icon="🧵",
    layout="wide",
)

OPENAI_MODEL_VISION = "gpt-4.1-mini"   # kept for possible future use
OPENAI_MODEL_IMAGE = "gpt-image-1"

client = OpenAI()

DATA_PATH = "market_and_place_products.xlsx"   # same folder as this file
LOGO_PATH = "logo.png"                         # same folder as this file
SHELF_BASE_PATH = "store shelf.jpg"            # your blank shelf photo


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
        "Image URL:": "image_url",
        "Category": "category",        # if present
    })

    # Safety: ensure strings
    for col in ["name", "color", "amazon", "image_url", "category"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    df["name_lower"] = df["name"].str.lower()
    df["color_lower"] = df["color"].str.lower()

    # Build a human-friendly label for dropdowns
    def make_label(row):
        pieces = [row["name"]]
        if row.get("color") and row["color"].lower() != "nan":
            pieces.append(f"Color: {row['color']}")
        if row.get("category") and row["category"].lower() != "nan":
            pieces.append(f"Category: {row['category']}")
        if row.get("price") and str(row["price"]).lower() != "nan":
            pieces.append(f"${row['price']}")
        return " | ".join(pieces)

    df["option_label"] = df.apply(make_label, axis=1)

    return df


catalog_df = load_catalog(DATA_PATH)

# Map from dropdown label -> dataframe index
LABEL_TO_INDEX: Dict[str, int] = {
    row.option_label: idx for idx, row in catalog_df.iterrows()
}
PRODUCT_OPTIONS: List[str] = list(LABEL_TO_INDEX.keys())


# ---------- HELPER FUNCTIONS (SEARCH) ----------

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


# ---------- RENDER HELPERS ----------

def render_product_card(row: pd.Series):
    cols = st.columns([1, 2.5])
    with cols[0]:
        # Prefer local image file if it exists, otherwise image_url
        local_name = f"{row['name']}.jpg"
        if os.path.exists(local_name):
            st.image(local_name, use_column_width=True)
        elif row.get("image_url") and str(row["image_url"]).startswith("http"):
            st.image(row["image_url"], use_column_width=True)
    with cols[1]:
        st.markdown(f"**{row['name']}**")
        st.write(f"• Color: {row['color']}")
        if str(row.get("category", "")).lower() not in ("", "nan"):
            st.write(f"• Category: {row['category']}")
        st.write(f"• Price: {row['price']}")
        if row.get("amazon") and str(row["amazon"]).startswith("http"):
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


# ---------- IMAGE GENERATION HELPERS (TEXT-TO-IMAGE) ----------

def guess_room_type(notes: str) -> str:
    t = (notes or "").lower()
    if any(w in t for w in ["bedroom", "bed", "quilt", "sheet", "duvet", "comforter"]):
        return "bedroom"
    if "beach" in t:
        return "beach bathroom"
    if "living" in t or "sofa" in t:
        return "living room"
    return "bathroom"


def generate_room_concept_image(
    product_row: pd.Series,
    user_notes: str,
) -> bytes:
    """
    Uses gpt-image-1 in a robust text-to-image mode.
    We *describe* the desired result instead of trying to send the raw room photo
    to avoid the 'image_url' / 'input[0].content' errors.
    """
    room_type = guess_room_type(user_notes)

    prompt = f"""
Photo-realistic {room_type} styled with Market & Place textiles.

Use the following Market & Place product as the hero item for all towels / bedding / textiles:

- Name: {product_row['name']}
- Color: {product_row['color']}
- Category: {product_row.get('category', '')}

Design goals:
- Keep a layout and architecture similar to a real customer photo.
- Make it clear that the visible towels / bedding / shower curtain are this product.
- Natural lighting, clean and modern, suitable for an e-commerce lifestyle image.

Customer styling notes (optional):
\"\"\"{user_notes}\"\"\".
"""

    img_resp = client.images.generate(
        model=OPENAI_MODEL_IMAGE,
        prompt=prompt,
        size="1024x1024",
        n=1,
    )
    b64 = img_resp.data[0].b64_json
    return base64.b64decode(b64)


def generate_store_shelf_concept(
    product_row: pd.Series,
    user_notes: str,
) -> bytes:
    """
    Same idea: pure text prompt describing the shelf fully stocked
    with the selected Market & Place product.
    """
    prompt = f"""
Retail store shelf / showroom concept for Market & Place.

Scene requirements:
- A long aisle of real retail shelving, similar to standard store gondola shelves.
- Shelves fully stocked and neatly folded / hung.
- No bathtubs, beds, or home furniture. This is a store, not a home.
- No logos or text in the image.

Use ONLY this Market & Place product as inspiration for all visible textiles:

- Name: {product_row['name']}
- Color: {product_row['color']}
- Category: {product_row.get('category', '')}

Customer styling notes (optional):
\"\"\"{user_notes}\"\"\".
"""

    img_resp = client.images.generate(
        model=OPENAI_MODEL_IMAGE,
        prompt=prompt,
        size="1024x1024",
        n=1,
    )
    b64 = img_resp.data[0].b64_json
    return base64.b64decode(b64)


# ---------- LAYOUT: HEADER (CONDENSED) ----------

header_cols = st.columns([1, 2, 1])
with header_cols[1]:
    if os.path.exists(LOGO_PATH):
        # Smaller logo, centered
        st.image(LOGO_PATH, use_column_width=False, width=260)
    st.markdown(
        "<h1 style='text-align:center; margin-top:0.1rem; margin-bottom:0.4rem;'>"
        "Market & Place AI Stylist</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; margin-bottom:0.2rem;'>"
        "Chat with an AI stylist, search the Market & Place catalog, "
        "and generate concept visualizations using your own product file."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; font-size:0.9rem;'>"
        "<a href='https://marketandplace.co/' style='text-decoration:none;'>"
        "← Return to Market & Place website</a></p>",
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

    # No search bar here anymore — just random products from the catalog
    if len(catalog_df) > 0:
        preview_df = catalog_df.sample(
            n=min(5, len(catalog_df)),
            replace=False,
        )
        render_product_list(preview_df)
    else:
        st.write("Catalog is empty.")


# ----- RIGHT: AI CONCEPT VISUALIZER (ROOM IMAGE + STORE SHELF IMAGE) -----

with right_col:
    st.subheader("AI concept visualizer")

    st.write(
        "Generate a **styled concept** using Market & Place products. "
        "Upload a room for AI styling, or generate a store shelf concept."
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

        # Dropdown of real catalog products
        st.markdown("**Choose a Market & Place product to feature in this room:**")
        selected_label_room = st.selectbox(
            "Product from catalog:",
            options=PRODUCT_OPTIONS,
            key="room_product_select",
        )
        selected_row_room = catalog_df.loc[LABEL_TO_INDEX[selected_label_room]]

        room_notes = st.text_input(
            "Optional: any styling notes?",
            placeholder="e.g. modern spa look, neutrals, add striped accents",
            key="room_notes",
        )

        if uploaded_room is not None:
            st.caption("Reference room photo (used as style inspiration):")
            st.image(uploaded_room, use_column_width=True)

        if st.button("Generate room concept image"):
            with st.spinner("Generating AI room concept…"):
                try:
                    img_bytes = generate_room_concept_image(
                        selected_row_room,
                        room_notes or "",
                    )
                    st.image(img_bytes, use_column_width=True)
                    st.caption(
                        "Concept image generated from your notes and the selected "
                        "Market & Place product. Layout is inspired by a typical room "
                        "similar to your photo."
                    )
                except Exception as e:
                    st.error(f"Image generation failed: {e}")

    # --- STORE SHELF CONCEPT ---
    else:
        st.markdown("#### Store shelf / showroom concept")

        if os.path.exists(SHELF_BASE_PATH):
            st.image(
                SHELF_BASE_PATH,
                caption="Market & Place store shelf photo (visual reference).",
                use_column_width=True,
            )

        st.markdown("**Choose a Market & Place product to feature on the shelf:**")
        selected_label_shelf = st.selectbox(
            "Product from catalog:",
            options=PRODUCT_OPTIONS,
            key="shelf_product_select",
        )
        selected_row_shelf = catalog_df.loc[LABEL_TO_INDEX[selected_label_shelf]]

        shelf_notes = st.text_input(
            "Optional: any styling notes for the shelf?",
            placeholder="e.g. alternating aqua and navy stacks, matching bath mats on bottom shelf",
            key="shelf_notes",
        )

        if st.button("Generate store shelf concept image"):
            with st.spinner("Generating AI store shelf concept…"):
                try:
                    img_bytes = generate_store_shelf_concept(
                        selected_row_shelf,
                        shelf_notes or "",
                    )
                    st.image(img_bytes, use_column_width=True)
                    st.caption(
                        "Concept image generated from the selected Market & Place product "
                        "and your styling notes. Scene is a retail shelf / showroom."
                    )
                except Exception as e:
                    st.error(f"Image generation failed: {e}")











