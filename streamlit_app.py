import base64
import os
from typing import List, Tuple, Optional

import pandas as pd
import streamlit as st
from openai import OpenAI

# ---------- CONFIG ----------

st.set_page_config(
    page_title="Market & Place AI Stylist",
    page_icon="🧵",
    layout="wide",
)

OPENAI_MODEL_VISION = "gpt-4.1-mini"
OPENAI_MODEL_IMAGE = "gpt-image-1"

client = OpenAI()

DATA_PATH = "market_and_place_products.xlsx"  # Excel in repo
LOGO_PATH = "logo.png"                        # Brand logo in repo
STORE_SHELF_REFERENCE = "store shelf.jpg"     # Reference shelf photo in repo


# ---------- DATA LOADING ----------

@st.cache_data(show_spinner=False)
def load_catalog(path: str) -> pd.DataFrame:
    """Load and standardise the Market & Place catalog."""
    df = pd.read_excel(path)

    df = df.rename(
        columns={
            "Product name": "name",
            "Color": "color",
            "Price": "price",
            "raw_amazon": "amazon",
            "Image URL:": "image_url",
            "Category": "category",
        }
    )

    # Safety: ensure strings
    for col in ["name", "color", "amazon", "image_url", "category"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Helper columns for simple search
    df["name_lower"] = df["name"].str.lower()
    df["color_lower"] = df["color"].str.lower()
    df["category_lower"] = df["category"].str.lower()

    return df


catalog_df = load_catalog(DATA_PATH)

# Unique categories from the sheet (e.g. beach, bathroom, bedroom)
CATEGORIES = sorted(
    {c.capitalize() for c in catalog_df["category"].unique() if isinstance(c, str)}
)


# ---------- SIMPLE NLP HELPERS ----------

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
    """
    Very simple keyword extractor.
    Returns (category_terms, color_terms).
    """
    t = text.lower()
    found_cats: List[str] = []
    found_colors: List[str] = []

    # Category-type words (towels vs sheets vs quilts, etc.)
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

    # Colour words
    for c in COLOR_WORDS:
        if c in t:
            found_colors.append(c)

    # De-duplicate while preserving order
    found_cats = list(dict.fromkeys(found_cats))
    found_colors = list(dict.fromkeys(found_colors))

    return found_cats, found_colors


def detect_room_category(text: str) -> Optional[str]:
    """
    Guess room category from free text: Bathroom / Bedroom / Beach.
    Returns the capitalised category from CATEGORIES if likely, else None.
    """
    t = text.lower()

    # Beach / outdoor
    if "beach" in t or "pool" in t or "cabana" in t:
        for c in CATEGORIES:
            if c.lower().startswith("beach"):
                return c

    # Bathroom
    if any(word in t for word in ["bathroom", "shower", "vanity", "bath towel", "bath mat"]):
        for c in CATEGORIES:
            if c.lower().startswith("bath"):
                return c

    # Bedroom / bed
    if any(word in t for word in ["bedroom", "bed", "duvet", "comforter", "sheet", "pillow"]):
        for c in CATEGORIES:
            if c.lower().startswith("bed"):
                return c

    return None


# ---------- PRODUCT IMAGE HELPER ----------

def guess_local_image_path(row: pd.Series) -> Optional[str]:
    """
    Best-effort guess for a local image filename based on product name & colour.
    If it can't find a match, returns None and we fall back to the URL.
    """

    def slugify(text: str) -> str:
        text = text.replace("%", "").replace("/", " ").replace("|", " ")
        text = "".join(ch if ch.isalnum() or ch in (" ", "-", "_") else "" for ch in text)
        return "_".join(text.split())

    # Try "Name Color.ext"
    candidates = []

    if "name" in row and "color" in row:
        base = f"{row['name']} {row['color']}"
        candidates.append(slugify(base))

    if "name" in row:
        candidates.append(slugify(row["name"]))

    for slug in candidates:
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            path = f"{slug}{ext}"
            if os.path.exists(path):
                return path

    return None


# ---------- CATALOG SEARCH ----------

def filter_catalog_by_query(
    df: pd.DataFrame,
    query: str,
    max_results: int = 8,
    room_category: Optional[str] = None,
) -> pd.DataFrame:
    """
    Looser matching:
    - Optionally filter by detected ROOM CATEGORY (bathroom / bedroom / beach).
    - Filter by detected product category words (towels vs sheets vs quilts).
    - Optionally narrow by colour words.
    - If still empty, fall back to substring match on product name.
    """
    if not query:
        if room_category:
            cat_mask = df["category_lower"].str.contains(room_category.lower(), na=False)
            return df[cat_mask].head(max_results)
        return df.head(max_results)

    q = query.lower()
    cat_terms, color_terms = detect_category_terms(q)

    mask = pd.Series(True, index=df.index)

    # 1. Room category constraint (from excel Category column)
    if room_category:
        mask &= df["category_lower"].str.contains(room_category.lower(), na=False)

    # 2. Product category words (towel, sheet, quilt, etc.)
    if cat_terms:
        cat_mask = pd.Series(False, index=df.index)
        for cat in cat_terms:
            keywords = CATEGORY_KEYWORDS.get(cat, [])
            for kw in keywords:
                cat_mask |= df["name_lower"].str.contains(kw, na=False)
        mask &= cat_mask

    # 3. Colour constraints (soft — if they remove everything, skip them)
    if color_terms:
        color_mask = pd.Series(False, index=df.index)
        for c in color_terms:
            color_mask |= df["color_lower"].str.contains(c, na=False) | df["name_lower"].str.contains(
                c, na=False
            )
        narrowed = df[mask & color_mask]
        if not narrowed.empty:
            mask &= color_mask

    results = df[mask]

    # 4. Fallback: substring search on product name
    if results.empty:
        simple_mask = pd.Series(False, index=df.index)
        for word in q.split():
            simple_mask |= df["name_lower"].str.contains(word, na=False)
        if room_category:
            simple_mask &= df["category_lower"].str.contains(room_category.lower(), na=False)
        results = df[simple_mask]

    return results.head(max_results)


# ---------- RENDERING HELPERS ----------

def render_product_card(row: pd.Series) -> None:
    cols = st.columns([1, 2.5])
    with cols[0]:
        local_path = guess_local_image_path(row)
        if local_path:
            st.image(local_path, use_column_width=True)
        elif row.get("image_url") and str(row["image_url"]).startswith("http"):
            st.image(row["image_url"], use_column_width=True)

    with cols[1]:
        st.markdown(f"**{row['name']}**")
        st.write(f"• Color: {row['color']}")
        st.write(f"• Price: {row['price']}")
        if row.get("category"):
            st.write(f"• Category: {row['category']}")
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


# ---------- STORE SHELF IMAGE GENERATION ----------

def generate_store_shelf_image(products: pd.DataFrame, description: str) -> bytes:
    """
    Use only Market & Place products as inspiration to generate a *retail store shelf* scene.
    Returns raw image bytes (PNG).
    """
    bullet_lines = []
    for _, r in products.iterrows():
        bullet_lines.append(
            f"- {r['name']} (category: {r['category']}, color: {r['color']}, price: {r['price']})"
        )

    product_snippet = (
        "\n".join(bullet_lines) if bullet_lines else "Market & Place towels and bedding in neutral colours."
    )

    prompt = f"""
You are generating a **retail store shelf or aisle** concept image for Market & Place.

STRICT SCENE REQUIREMENTS:
- It must clearly be a store shelf / showroom, not a home bathroom or bedroom.
- Show long shelving bays with neatly folded stacks of textiles and some hanging pieces.
- Do NOT show bathtubs, sinks, toilets, beds, sofas, or residential furniture.
- No people and no visible brand logos.

Use ONLY the following Market & Place products as inspiration for colours, patterns, and textures.
Do not invent totally unrelated products:

{product_snippet}

Customer request / styling direction:
\"\"\"{description}\"\"\".

Render a clean, well-lit store interior with multiple rows of shelves and an end-cap display,
styled using those products. Prioritise matching product CATEGORY and COLOUR.
"""

    img_resp = client.images.generate(
        model=OPENAI_MODEL_IMAGE,
        prompt=prompt,
        size="1024x1024",
        n=1,
    )

    b64 = img_resp.data[0].b64_json
    return base64.b64decode(b64)


# ---------- LAYOUT: COMPACT HEADER ----------

header_cols = st.columns([1, 3, 1])
with header_cols[1]:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_column_width=False, width=420)

    st.markdown(
        "<h1 style='text-align:center; margin-top:0.5rem; margin-bottom:0.25rem;'>"
        "Market & Place AI Stylist</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; margin:0; font-size:0.95rem;'>"
        "Chat with an AI stylist, search the Market & Place catalog, and generate concept "
        "visualizations using your own product file."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; margin-top:0.35rem;'>"
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
        detected_room_cat = detect_room_category(user_query)
        cat_note = (
            f" _(detected room category: **{detected_room_cat}**)_"
            if detected_room_cat
            else ""
        )
        st.markdown(f"### 🧵 {user_query.strip()}{cat_note}")

        matches = filter_catalog_by_query(
            catalog_df,
            user_query,
            max_results=6,
            room_category=detected_room_cat,
        )
        render_product_list(matches)

    st.markdown("---")
    st.subheader("Quick catalog peek")

    peek_query = st.text_input(
        "Filter products by keyword:",
        placeholder="e.g. cabana stripe, flannel sheet, navy",
        key="peek_query",
    )

    if peek_query.strip():
        # No room context here – just show relevant products
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

    # --- ROOM PRODUCT SUGGESTIONS (catalog-only, no edits) ---
    if mode == "Room product suggestions":
        st.markdown("#### Room product suggestions")

        uploaded_room = st.file_uploader(
            "Optional: upload a photo of your room (bathroom, bedroom, etc.)",
            type=["jpg", "jpeg", "png"],
            help="The image is just for your reference – suggestions are catalog-only right now.",
        )

        textiles_request = st.text_input(
            "What textiles would you like to add or change?",
            placeholder="e.g. navy striped bath towels, white quilt, cabana stripe shower curtain",
            key="room_textiles_request",
        )

        if st.button("Get product suggestions"):
            st.markdown("### Suggested Market & Place products for your room")

            if uploaded_room is not None:
                st.caption(
                    "Reference photo uploaded – suggestions below are from the catalog only "
                    "(no image edits yet)."
                )
                st.image(uploaded_room, use_column_width=True)

            detected_room_cat = detect_room_category(textiles_request)
            suggestions = filter_catalog_by_query(
                catalog_df,
                textiles_request or "",
                max_results=8,
                room_category=detected_room_cat,
            )
            if detected_room_cat:
                st.caption(f"Filtering by room category: **{detected_room_cat}**")
            render_product_list(suggestions)

    # --- STORE SHELF / SHOWROOM IMAGE GENERATOR ---
    else:
        st.markdown("#### Store shelf / showroom concept")

        cols_top = st.columns([2, 1])
        with cols_top[0]:
            shelf_request = st.text_input(
                "Describe the shelf you want to visualize:",
                placeholder="e.g. cabana stripe beach towels in aqua and navy, folded stacks and matching mats",
                key="shelf_request",
            )

        with cols_top[1]:
            shelf_category = st.selectbox(
                "Shelf category (from catalog):",
                options=CATEGORIES,
                index=0 if CATEGORIES else 0,
                help="This ensures the AI only uses products from the chosen section.",
            )

        # Optional reference shelf image
        if os.path.exists(STORE_SHELF_REFERENCE):
            st.caption("Reference Market & Place shelf photo (for style only):")
            st.image(STORE_SHELF_REFERENCE, use_column_width=True)

        if st.button("Generate store shelf concept image"):
            with st.spinner("Generating store shelf concept…"):
                shelf_products = filter_catalog_by_query(
                    catalog_df,
                    shelf_request or "",
                    max_results=6,
                    room_category=shelf_category,
                )

                img_bytes = generate_store_shelf_image(shelf_products, shelf_request or "")

                st.markdown("### AI-generated store shelf concept")
                st.image(img_bytes, use_column_width=True)

                st.caption(
                    "Concept image generated from the Market & Place catalog. "
                    "The scene is intended as a retail shelf / showroom, not a home bathroom or bedroom."
                )










