import base64
import os
from typing import List, Tuple

import pandas as pd
import streamlit as st
from openai import OpenAI

# ---------- CONFIG ----------

st.set_page_config(
    page_title="Market & Place AI Stylist",
    page_icon="🧵",
    layout="wide",
)

OPENAI_MODEL_IMAGE = "gpt-image-1"

client = OpenAI()

DATA_PATH = "market_and_place_products.xlsx"  # catalog
LOGO_PATH = "logo.png"                        # top logo
STORE_SHELF_PATH = "store shelf.jpg"         # empty shelf photo


# ---------- DATA LOADING ----------

@st.cache_data(show_spinner=False)
def load_catalog(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    # Standardise column names
    df = df.rename(
        columns={
            "Product name": "name",
            "Color": "color",
            "Price": "price",
            "raw_amazon": "amazon",
            "Image URL:": "image_url",
            "Category": "category",  # if you added this
        }
    )

    # Safety: ensure strings
    for col in ["name", "color", "amazon", "image_url", "category"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    df["name_lower"] = df["name"].str.lower()
    df["color_lower"] = df["color"].str.lower()
    df["category_lower"] = df.get("category", "").astype(str).str.lower()

    return df


@st.cache_data(show_spinner=False)
def load_logo_b64() -> str:
    if not os.path.exists(LOGO_PATH):
        return ""
    with open(LOGO_PATH, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


@st.cache_data(show_spinner=False)
def load_store_shelf_bytes() -> bytes:
    if not os.path.exists(STORE_SHELF_PATH):
        return b""
    with open(STORE_SHELF_PATH, "rb") as f:
        return f.read()


catalog_df = load_catalog(DATA_PATH)
logo_b64 = load_logo_b64()
store_shelf_bytes = load_store_shelf_bytes()


# ---------- SIMPLE QUERY → CATALOG FILTERING ----------

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
    """
    Very simple keyword extractor.
    Returns (category_terms, color_terms).
    """
    t = text.lower()
    found_cats: List[str] = []
    found_colors: List[str] = []

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

    found_cats = list(dict.fromkeys(found_cats))
    found_colors = list(dict.fromkeys(found_colors))
    return found_cats, found_colors


def filter_catalog_by_query(df: pd.DataFrame, query: str, max_results: int = 8) -> pd.DataFrame:
    """
    Looser matching:
    - First, filter by detected category (towels vs sheets vs quilts, etc.).
    - Then optionally narrow by colors.
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
                cat_mask |= df["name_lower"].str.contains(kw, na=False)
        mask &= cat_mask

    # color constraints (soft)
    if color_terms:
        color_mask = pd.Series(False, index=df.index)
        for c in color_terms:
            color_mask |= (
                df["color_lower"].str.contains(c, na=False)
                | df["name_lower"].str.contains(c, na=False)
            )

        narrowed = df[mask & color_mask]
        if not narrowed.empty:
            mask &= color_mask

    results = df[mask]

    # fallback: simple substring match on name
    if results.empty:
        simple_mask = pd.Series(False, index=df.index)
        for word in q.split():
            simple_mask |= df["name_lower"].str.contains(word, na=False)
        results = df[simple_mask]

    return results.head(max_results)


# ---------- RENDER HELPERS ----------

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


def build_catalog_snippet(df: pd.DataFrame) -> str:
    """
    Build a short bullet list of catalog items for the image prompt.
    """
    lines = []
    for _, r in df.iterrows():
        lines.append(f"- {r['name']} (color: {r['color']}, price: {r['price']})")
    if not lines:
        return "- Market & Place towels and textiles in neutral and blue shades."
    return "\n".join(lines)


# ---------- IMAGE GENERATION HELPERS (NO VISION API) ----------

def generate_room_concept_image(room_bytes: bytes, textiles_request: str) -> bytes:
    """
    Edit the user's room photo:
    - Keep layout + fixtures.
    - Only change/add textiles based on Market & Place catalog.
    """
    matched = filter_catalog_by_query(catalog_df, textiles_request or "", max_results=6)
    product_snippet = build_catalog_snippet(matched)

    prompt = f"""
You are editing the user's existing room photo for Market & Place.

STRICT RULES:
- Keep the walls, tiles, fixtures, windows, furniture, and layout IDENTICAL to the original photo.
- Only change or add soft textiles: towels, bath mats, shower curtains, quilts, sheets, pillow shams, etc.
- Do NOT move or remove any architectural elements or furniture.
- No people, no clutter, no logos.

Use ONLY these Market & Place products as inspiration for colors, patterns, and textures
(do not invent completely different products):

{product_snippet}

User styling request:
\"\"\"{textiles_request}\"\"\".

Output a realistic edited photograph of the SAME room with the updated textiles.
"""

    img_resp = client.images.generate(
        model=OPENAI_MODEL_IMAGE,
        prompt=prompt,
        size="1024x1024",
        n=1,
        image=room_bytes,  # image editing
    )
    b64 = img_resp.data[0].b64_json
    return base64.b64decode(b64)


def generate_store_shelf_concept_image(shelf_request: str) -> bytes:
    """
    Edit the base store shelf photo:
    - Keep the same shelving + environment.
    - Fill shelves with Market & Place textiles that match the request.
    """
    matched = filter_catalog_by_query(catalog_df, shelf_request or "", max_results=8)
    product_snippet = build_catalog_snippet(matched)

    prompt = f"""
You are editing the attached Market & Place STORE SHELF photograph.

STRICT RULES:
- Do NOT change the store architecture, camera angle, ceiling, floor, or shelving units.
- Only add or change folded stacks and hanging textiles on the shelves (towels, sheets, quilts, mats).
- Do NOT add any packaging, boxes, bottles, or non-textile products.
- No people, signage, or extra props.

Use ONLY these Market & Place products as inspiration for colors, stripes, and textures:

{product_snippet}

User styling request:
\"\"\"{shelf_request}\"\"\".

Output a realistic edited photograph of the SAME store shelf, now fully merchandised with Market & Place products.
"""

    img_resp = client.images.generate(
        model=OPENAI_MODEL_IMAGE,
        prompt=prompt,
        size="1024x1024",
        n=1,
        image=store_shelf_bytes,  # edit your base shelf photo
    )
    b64 = img_resp.data[0].b64_json
    return base64.b64decode(b64)


# ---------- HEADER (SMALLER & CONDENSED) ----------

header_html = """
<div style="text-align:center; margin-top:0.5rem; margin-bottom:0.25rem;">
  {logo}
</div>
<div style="text-align:center; margin-bottom:0.5rem;">
  <h3 style="margin:0.1rem 0 0.2rem 0;">Market &amp; Place AI Stylist</h3>
  <p style="margin:0.1rem 0; font-size:0.95rem;">
    Chat with an AI stylist, search the Market &amp; Place catalog, and generate concept visualizations using your own product file.
  </p>
  <p style="margin:0.1rem 0; font-size:0.9rem;">
    <a href="https://marketandplace.co/" style="text-decoration:none;">
      &larr; Return to Market &amp; Place website
    </a>
  </p>
</div>
"""

if logo_b64:
    logo_tag = f'<img src="data:image/png;base64,{logo_b64}" style="width:220px;" />'
else:
    logo_tag = ""

st.markdown(
    header_html.format(logo=logo_tag),
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
        sent = st.form_submit_button("Send")

    if sent and user_query.strip():
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
        "Generate **visual concepts** using Market & Place products. "
        "You can either edit a *room photo* you upload or edit the *store shelf* photo in the app."
    )

    mode = st.radio(
        "What do you want to visualize?",
        ["Room concept image", "Store shelf / showroom concept image"],
        horizontal=True,
    )

    # --- ROOM CONCEPT IMAGE (image editing) ---
    if mode == "Room concept image":
        st.markdown("#### Room concept image")

        uploaded_room = st.file_uploader(
            "Upload a photo of your room (bathroom, bedroom, etc.)",
            type=["jpg", "jpeg", "png"],
            help="The AI will keep the same room and only change textiles using Market & Place products.",
        )

        textiles_request = st.text_input(
            "What textiles would you like to add or change?",
            placeholder="e.g. navy striped bath towels, white quilt, cabana stripe shower curtain",
            key="room_textiles_request",
        )

        if st.button("Generate room concept image"):
            if uploaded_room is None:
                st.warning("Please upload a room photo first.")
            else:
                with st.spinner("Generating room concept…"):
                    room_bytes = uploaded_room.getvalue()
                    img_bytes = generate_room_concept_image(
                        room_bytes=room_bytes,
                        textiles_request=textiles_request or "",
                    )
                st.markdown("### AI-generated room concept")
                st.image(img_bytes, use_column_width=True)
                st.caption(
                    "The room structure is preserved while textiles are styled "
                    "using Market & Place products from the catalog."
                )

    # --- STORE SHELF CONCEPT IMAGE (image editing of your base shelf) ---
    else:
        st.markdown("#### Store shelf / showroom concept image")

        if not store_shelf_bytes:
            st.error("Store shelf base image file not found in the repo (`store shelf.jpg`).")
        else:
            st.caption("Base Market & Place store shelf photo used for all shelf concepts:")
            st.image(store_shelf_bytes, use_column_width=True)

        shelf_request = st.text_input(
            "Describe how you want the shelf styled:",
            placeholder="e.g. cabana stripe beach towels in aqua and navy, folded stacks and matching mats",
            key="shelf_request",
        )

        if st.button("Generate store shelf concept image"):
            if not store_shelf_bytes:
                st.error("Store shelf base image is missing, so the concept image can't be generated.")
            else:
                with st.spinner("Generating store shelf concept…"):
                    img_bytes = generate_store_shelf_concept_image(
                        shelf_request=shelf_request or "",
                    )
                st.markdown("### AI-generated store shelf concept")
                st.image(img_bytes, use_column_width=True)
                st.caption(
                    "Shelf layout is kept the same; textiles are styled using Market & Place products "
                    "based on your request."
                )








