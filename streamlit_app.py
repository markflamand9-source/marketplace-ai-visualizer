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

OPENAI_MODEL_VISION = "gpt-4.1-mini"
OPENAI_MODEL_IMAGE = "gpt-image-1"

client = OpenAI()

DATA_PATH = "market_and_place_products.xlsx"  # same folder as this file
LOGO_PATH = "logo.png"                        # same folder as this file
SHELF_PHOTO_PATH = "store shelf.jpg"         # your empty store shelf photo


# ---------- DATA LOADING ----------

@st.cache_data(show_spinner=False)
def load_catalog(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    # Standardise column names just once
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

    # Pre-computed helper columns for simple keyword search
    df["name_lower"] = df["name"].str.lower()
    df["color_lower"] = df["color"].str.lower()
    df["category_lower"] = df.get("category", "").str.lower()

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
    found_cats = []
    found_colors = []

    # category by explicit words
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


def filter_catalog_by_query(
    df: pd.DataFrame, query: str, max_results: int = 8
) -> pd.DataFrame:
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
            color_mask |= df["color_lower"].str.contains(
                c, case=False, na=False
            ) | df["name_lower"].str.contains(c, case=False, na=False)
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


def generate_store_shelf_image(
    products: pd.DataFrame, description: str, shelf_category: str
) -> bytes:
    """
    Generate a **store shelf** concept image using only Market & Place products
    as inspiration. We can't literally paint onto the uploaded JPEG with this
    API, but we can strongly nudge it to copy that style/layout.
    """
    bullet_lines = []
    for _, r in products.iterrows():
        bullet_lines.append(
            f"- {r['name']} (color: {r['color']}, price: {r['price']})"
        )

    product_snippet = (
        "\n".join(bullet_lines)
        if bullet_lines
        else "Market & Place towels, quilts, sheets and other textiles."
    )

    category_hint = shelf_category.lower()
    if "bath" in category_hint:
        cat_desc = (
            "Focus on bath towels, hand towels, bath mats and maybe a few shower curtains."
        )
    elif "beach" in category_hint:
        cat_desc = (
            "Focus on colourful beach towels and cabana stripes, maybe a few rolled towels in bins."
        )
    elif "bed" in category_hint:
        cat_desc = (
            "Focus on folded quilts, comforters, sheet sets and maybe some stacked pillows."
        )
    else:
        cat_desc = (
            "Use a mix of towels, bedding and other textiles that make sense together."
        )

    prompt = f"""
You are generating a **retail store shelf or aisle** concept image for Market & Place.

Scene requirements (very important):
- It must clearly be a store shelf / showroom, not a home bathroom or bedroom.
- Use long white gondola shelving in rows, similar to a typical department store.
- Camera angle similar to the reference photo: looking down an aisle of empty white shelves.
- Do NOT show bathtubs, sinks, toilets, beds, or home furniture.
- No people, and no visible brand logos.

Shelf focus:
- Shelf category: {shelf_category}
- {cat_desc}

Use ONLY the following Market & Place products as inspiration for colours, stripes, patterns and textures.
Don't invent totally unrelated products; stay close to these:

{product_snippet}

Customer request / styling direction:
\"\"\"{description}\"\"\".

Render a clean, well-lit store interior with those shelves fully merchandised using Market & Place textiles.
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
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=260, use_column_width=False)
    st.markdown(
        "<h2 style='text-align:center; margin-bottom:0.3rem;'>Market & Place AI Stylist</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; font-size:0.9rem; margin-bottom:0.2rem;'>"
        "Chat with an AI stylist, search the Market & Place catalog, and generate concept visualizations using your own product file."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; font-size:0.85rem;'>"
        "<a href='https://marketandplace.co/' style='text-decoration:none;'>"
        "← Return to Market & Place website</a></p>",
        unsafe_allow_html=True,
    )

st.markdown("---")


# ---------- MAIN LAYOUT (2 COLUMNS) ----------

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
        peek_matches = filter_catalog_by_query(
            catalog_df, peek_query, max_results=12
        )
        render_product_list(peek_matches)
    else:
        st.write("Type a keyword above to quickly peek at the catalog.")


# ----- RIGHT: AI CONCEPT VISUALIZER (ROOM SUGGESTIONS + STORE SHELF) ----- #

with right_col:
    st.subheader("AI concept visualizer")

    st.write(
        "Generate a **styled concept** using Market & Place products. "
        "You can either get product suggestions for a room, or generate a store shelf concept image."
    )

    mode = st.radio(
        "What do you want to visualize?",
        ["Room product suggestions", "Store shelf / showroom concept"],
        horizontal=True,
    )

    # --- ROOM PRODUCT SUGGESTIONS (catalog only, no image edits) --- #
    if mode == "Room product suggestions":
        # NOTE: this is the behaviour you said is “good enough now” — unchanged.
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
                st.caption(
                    "Reference photo uploaded – suggestions are still catalog-only, not image edits."
                )
                st.image(uploaded_room, use_column_width=True)

            suggestions = filter_catalog_by_query(
                catalog_df, textiles_request or "", max_results=8
            )
            render_product_list(suggestions)

    # --- STORE SHELF / SHOWROOM IMAGE GENERATOR --- #
    else:
        st.markdown("#### Store shelf / showroom concept")

        # show your real empty shelf photo as reference
        if os.path.exists(SHELF_PHOTO_PATH):
            st.image(
                SHELF_PHOTO_PATH,
                caption=(
                    "This is your real Market & Place store shelf photo. "
                    "The AI concept below is generated to match this kind of layout."
                ),
                use_column_width=True,
            )

        shelf_query = st.text_input(
            "Describe the shelf you want to visualize:",
            placeholder="e.g. cabana stripe beach towels in aqua and navy, folded stacks and matching mats",
            key="shelf_request",
        )

        # if you have a Category column, use its unique values
        if "category" in catalog_df.columns:
            unique_cats = (
                catalog_df["category"]
                .dropna()
                .replace("nan", "")
                .unique()
                .tolist()
            )
            # Fallback if messy data
            if not unique_cats:
                unique_cats = ["Bathroom", "Beach", "Bedroom"]
        else:
            unique_cats = ["Bathroom", "Beach", "Bedroom"]

        shelf_category = st.selectbox(
            "Shelf category (from catalog):",
            unique_cats,
            help="This nudges the AI toward bathroom, beach, or bedding products.",
        )

        if st.button("Generate store shelf concept image"):
            with st.spinner("Generating store shelf concept…"):
                # Limit catalog to that category if possible
                subset = catalog_df
                if "category_lower" in subset.columns and shelf_category:
                    cat_lower = shelf_category.lower()
                    subset = subset[
                        subset["category_lower"].str.contains(
                            cat_lower, case=False, na=False
                        )
                    ]
                    if subset.empty:
                        subset = catalog_df

                shelf_products = filter_catalog_by_query(
                    subset, shelf_query or "", max_results=10
                )

                img_bytes = generate_store_shelf_image(
                    shelf_products, shelf_query or "", shelf_category
                )

                st.markdown("### AI-generated store shelf concept")
                st.image(img_bytes, use_column_width=True)

                st.caption(
                    "Concept image generated from the Market & Place catalog. "
                    "Because of current API limits it can’t paint directly onto the exact JPG above, "
                    "but it is designed to mimic that shelf layout and use only Market & Place products "
                    "for colours and patterns."
                )









