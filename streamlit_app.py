import base64
import os
from typing import List, Tuple

import pandas as pd
import streamlit as st
from openai import OpenAI

# ---------- CONFIG ----------

LOGO_PATH = "logo.png"                  # same folder as this file
DATA_PATH = "market_and_place_products.xlsx"
SHELF_BASE_PATH = "store shelf.jpg"     # base shelf photo in repo

OPENAI_MODEL_IMAGE = "gpt-image-1"

client = OpenAI()

st.set_page_config(
    page_title="Market & Place AI Stylist",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🧵",
    layout="wide",
)

# ---------- DATA LOADING ----------

@st.cache_data(show_spinner=False)
def load_catalog(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    df = df.rename(columns={
        "Product name": "name",
        "Color": "color",
        "Price": "price",
        "raw_amazon": "amazon",
        "Image URL:": "image_url",
        "Category": "category",  # if present
    })

    for col in ["name", "color", "amazon", "image_url", "category"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    df["name_lower"] = df["name"].str.lower()
    df["color_lower"] = df["color"].str.lower()

    return df


catalog_df = load_catalog(DATA_PATH)


# ---------- SIMPLE SEARCH HELPERS (for left column tools) ----------

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


# ---------- RENDERING HELPERS ----------

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


# ---------- IMAGE GENERATION HELPERS ----------

def decode_image_response(img_resp) -> bytes:
    b64 = img_resp.data[0].b64_json
    return base64.b64decode(b64)


def generate_room_concept_image(room_bytes: bytes, product_row: pd.Series, styling_notes: str) -> bytes:
    """
    Edit the uploaded room photo so it keeps the same layout & camera angle,
    but swaps textiles to match the selected Market & Place product.
    """
    temp_room_path = "tmp_room_base.png"
    with open(temp_room_path, "wb") as f:
        f.write(room_bytes)

    prompt = f"""
You are an interior stylist for Market & Place.

Use the uploaded room photo as the base image. KEEP:
- the same room type (bedroom stays bedroom, bathroom stays bathroom),
- the same layout, walls, windows, doors, furniture, camera angle and lighting.

ONLY update textiles and related soft goods so they showcase this Market & Place product:

Name: {product_row['name']}
Color: {product_row['color']}

Where appropriate in the room, swap bedding, towels, shower curtains, rugs or pillows so they visually match this product.
Do NOT remove architectural features, change the room type, or add people.

Extra styling notes from the customer: {styling_notes or 'no extra notes'}.
"""

    img_resp = client.images.edit(
        model=OPENAI_MODEL_IMAGE,
        image=[open(temp_room_path, "rb")],
        prompt=prompt,
        size="1024x1024",
        n=1,
    )
    return decode_image_response(img_resp)


def generate_store_shelf_concept(product_row: pd.Series, styling_notes: str) -> bytes:
    """
    Edit the fixed Market & Place shelf photo so the shelves are filled with the
    selected Market & Place product.
    """
    if not os.path.exists(SHELF_BASE_PATH):
        raise FileNotFoundError(f"Base shelf photo not found: {SHELF_BASE_PATH}")

    prompt = f"""
Use the uploaded store shelf photo as the base image.

KEEP the same store interior, shelving layout, perspective and lighting.
Fill the shelves with neatly folded and/or hanging stacks of this Market & Place product:

Name: {product_row['name']}
Color: {product_row['color']}

Style the shelf like a real retail display, with consistent facings and tidy rows.
Do not change the store architecture, flooring or ceiling.

Extra styling notes from the customer: {styling_notes or 'no extra notes'}.
"""

    img_resp = client.images.edit(
        model=OPENAI_MODEL_IMAGE,
        image=[open(SHELF_BASE_PATH, "rb")],
        prompt=prompt,
        size="1024x1024",
        n=1,
    )
    return decode_image_response(img_resp)


# ------------ HEADER ------------

import base64

def load_base64_image(path):
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return encoded

# Load logo safely
logo_b64 = load_base64_image(LOGO_PATH)

st.markdown(
    f"""
    <style>
        .header-wrapper {{
            text-align: center;
            margin-top: -40px;
            margin-bottom: -10px;
        }}
        .header-logo {{
            width: 260px;
            margin-left: auto;
            margin-right: auto;
            display: block;
        }}
        .header-title {{
            font-size: 1.7rem !important;
            font-weight: 600;
            margin-top: 0px;
        }}
        .header-sub {{
            font-size: 1rem !important;
            margin-top: -5px;
        }}
    </style>

    <div class="header-wrapper">
        <img class="header-logo" src="data:image/png;base64,{logo_b64}">
        <div class="header-title">Market & Place AI Stylist</div>
        <div class="header-sub">
            Chat with an AI stylist, search the Market & Place catalog, and generate concept visualizations using your own product file.
        </div>
        <p><a href='https://marketandplace.co/'>← Return to Market & Place website</a></p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")


# ---------- MAIN LAYOUT (2 COLUMNS) ----------

left_col, right_col = st.columns([1, 1], gap="large")

# ----- LEFT: ASK THE AI STYLIST + RANDOM CATALOG PEEK -----

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

    st.caption("Here are a few Market & Place products pulled at random from the catalog:")
    sample_count = min(5, len(catalog_df))
    sample_df = catalog_df.sample(sample_count) if sample_count > 0 else catalog_df.head(0)
    render_product_list(sample_df)


# ----- RIGHT: AI CONCEPT VISUALIZER (ROOM IMAGE + STORE SHELF IMAGE) -----

with right_col:
    st.subheader("AI concept visualizer")

    st.write(
        "Generate a **styled concept** using your own photos and Market & Place products. "
        "You can either upload a room for AI styling, or generate a store shelf concept "
        "using the Market & Place shelf photo."
    )

    mode = st.radio(
        "What do you want to visualize?",
        ["Room concept image", "Store shelf / showroom concept"],
        horizontal=True,
    )

    product_names = sorted(catalog_df["name"].unique().tolist())

    # --- ROOM CONCEPT IMAGE (image edit) ---
    if mode == "Room concept image":
        st.markdown("#### Room concept image")

        if not product_names:
            st.warning("No products found in the catalog file.")
        else:
            room_product_name = st.selectbox(
                "Choose a Market & Place product to feature in this room:",
                product_names,
                key="room_product_select",
            )
            room_product_row = catalog_df[catalog_df["name"] == room_product_name].iloc[0]
            render_product_card(room_product_row)

            styling_notes = st.text_input(
                "Optional: any styling notes?",
                placeholder="e.g. modern spa look, neutrals, add striped accents",
                key="room_styling_notes",
            )

            uploaded_room = st.file_uploader(
                "Upload a photo of your room (bathroom, bedroom, etc.):",
                type=["jpg", "jpeg", "png"],
                key="room_upload",
            )

            if uploaded_room is not None:
                st.image(uploaded_room, caption="Uploaded room photo", use_column_width=True)

            if st.button("Generate room concept image"):
                if uploaded_room is None:
                    st.error("Please upload a room photo first.")
                else:
                    with st.spinner("Generating AI room concept…"):
                        try:
                            room_bytes = uploaded_room.getvalue()
                            img_bytes = generate_room_concept_image(
                                room_bytes, room_product_row, styling_notes
                            )
                            st.image(img_bytes, use_column_width=True)
                            st.caption(
                                "The AI kept your room layout and generated updated textiles "
                                "based on your selected Market & Place product."
                            )
                        except Exception as e:
                            st.error(f"Image generation failed: {e}")

    # --- STORE SHELF CONCEPT (image edit from fixed shelf photo) ---
    else:
        st.markdown("#### Store shelf / showroom concept")

        if not os.path.exists(SHELF_BASE_PATH):
            st.error(f"Base store shelf photo not found at `{SHELF_BASE_PATH}` in the repo.")
        elif not product_names:
            st.warning("No products found in the catalog file.")
        else:
            st.image(
                SHELF_BASE_PATH,
                caption="Market & Place store shelf photo (base image used for AI concept).",
                use_column_width=True,
            )

            shelf_product_name = st.selectbox(
                "Choose a Market & Place product to feature on this shelf:",
                product_names,
                key="shelf_product_select",
            )
            shelf_product_row = catalog_df[catalog_df["name"] == shelf_product_name].iloc[0]
            render_product_card(shelf_product_row)

            shelf_notes = st.text_input(
                "Optional: any styling notes for the shelf?",
                placeholder="e.g. color-block the stacks, mix folded and hanging towels",
                key="shelf_styling_notes",
            )

            if st.button("Generate store shelf concept image"):
                with st.spinner("Generating AI store shelf concept…"):
                    try:
                        img_bytes = generate_store_shelf_concept(
                            shelf_product_row, shelf_notes
                        )
                        st.image(img_bytes, use_column_width=True)
                        st.caption(
                            "The AI kept the same store shelf layout and filled it with "
                            "Market & Place products based on your selection."
                        )
                    except Exception as e:
                        st.error(f"Image generation failed: {e}")










