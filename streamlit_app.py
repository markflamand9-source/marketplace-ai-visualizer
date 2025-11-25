import base64
import io
import os
from typing import List, Tuple, Optional

import pandas as pd
import streamlit as st
from openai import OpenAI

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

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
STORE_SHELF_PHOTO = "store shelf.jpg"  # your reference shelf photo in repo root

# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_catalog(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    # Normalise / rename common columns
    df = df.rename(
        columns={
            "Product name": "name",
            "Color": "color",
            "Price": "price",
            "raw_amazon": "amazon",
            "Image URL:": "image_url",
            "Category": "category",
            "category": "category",
            "Image file": "image_file",
            "Image File": "image_file",
            "Local image file": "image_file",
        }
    )

    # Ensure expected columns exist
    for col in ["name", "color", "price", "amazon", "image_url", "category", "image_file"]:
        if col not in df.columns:
            df[col] = ""

    # Make sure strings where appropriate
    for col in ["name", "color", "amazon", "image_url", "category", "image_file"]:
        df[col] = df[col].astype(str).fillna("")

    # Helper lowercase columns for search
    df["name_lower"] = df["name"].str.lower()
    df["color_lower"] = df["color"].str.lower()
    df["category_lower"] = df["category"].str.lower()

    return df


catalog_df = load_catalog(DATA_PATH)

# ---------------------------------------------------------
# KEYWORD / CATEGORY HANDLING
# ---------------------------------------------------------

CATEGORY_KEYWORDS = {
    "towel": ["towel", "hand towel", "bath towel"],
    "beach_towel": ["beach towel", "cabana"],
    "sheet": ["sheet set", "sheet"],
    "quilt": ["quilt", "coverlet"],
    "comforter": ["comforter", "duvet"],
    "bedding": ["quilt", "coverlet", "sheet", "comforter", "bedding"],
}

# Map product categories in Excel to our high-level usage buckets
# (these should match values you put in the Category column, e.g. Bathroom / Bedroom / Beach)
CATEGORY_ROOM_MAP = {
    "towel": ["bathroom", "beach"],
    "beach_towel": ["beach"],
    "sheet": ["bedroom"],
    "quilt": ["bedroom"],
    "comforter": ["bedroom"],
    "bedding": ["bedroom"],
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
    """Return (category_terms, color_terms) from a free-text query."""
    t = text.lower()
    found_cats: List[str] = []
    found_colors: List[str] = []

    # category by words in query
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

    # colors
    for c in COLOR_WORDS:
        if c in t:
            found_colors.append(c)

    # de-dup while preserving order
    found_cats = list(dict.fromkeys(found_cats))
    found_colors = list(dict.fromkeys(found_colors))

    return found_cats, found_colors


def filter_catalog_by_query(
    df: pd.DataFrame, query: str, max_results: int = 8
) -> pd.DataFrame:
    """
    Looser matching using:
      - category words in the query
      - optional Category column
      - color words
      - fallback to substring match on name
    """
    if not query:
        return df.head(max_results)

    q = query.lower()
    cat_terms, color_terms = detect_category_terms(q)

    mask = pd.Series(True, index=df.index)

    # 1) Prefer filtering by Excel Category when possible
    if cat_terms and "category_lower" in df.columns:
        cat_mask = pd.Series(False, index=df.index)
        for cat in cat_terms:
            target_cats = CATEGORY_ROOM_MAP.get(cat, [])
            for room_cat in target_cats:
                cat_mask |= df["category_lower"].str.contains(room_cat, na=False)
        if cat_mask.any():
            mask &= cat_mask

    # 2) Fallback: filter by keywords in the product name
    if cat_terms:
        by_name_mask = pd.Series(False, index=df.index)
        for cat in cat_terms:
            kws = CATEGORY_KEYWORDS.get(cat, [])
            for kw in kws:
                by_name_mask |= df["name_lower"].str.contains(kw, na=False)
        if by_name_mask.any():
            mask &= by_name_mask

    # 3) Color constraints (soft: only use if they don't erase everything)
    if color_terms:
        color_mask = pd.Series(False, index=df.index)
        for c in color_terms:
            color_mask |= df["color_lower"].str.contains(c, na=False)
            color_mask |= df["name_lower"].str.contains(c, na=False)
        narrowed = df[mask & color_mask]
        if not narrowed.empty:
            mask &= color_mask

    results = df[mask]

    # 4) Final fallback: simple word match on product name
    if results.empty:
        simple_mask = pd.Series(False, index=df.index)
        for word in q.split():
            simple_mask |= df["name_lower"].str.contains(word, na=False)
        results = df[simple_mask]

    return results.head(max_results)


# ---------------------------------------------------------
# RENDERING UTILITIES
# ---------------------------------------------------------


def get_local_image_path(row: pd.Series) -> Optional[str]:
    """
    Prefer local image file from the repo if specified and exists.
    Otherwise return None (the caller can fall back to image_url).
    """
    fname = row.get("image_file", "")
    fname = str(fname).strip()
    if not fname or fname.lower() == "nan":
        return None

    # The file is assumed to live in the same directory as the app (repo root)
    path = os.path.join(".", fname)
    if os.path.exists(path):
        return path
    return None


def render_product_card(row: pd.Series) -> None:
    cols = st.columns([1, 2.5])
    with cols[0]:
        local_path = get_local_image_path(row)
        if local_path:
            st.image(local_path, use_column_width=True)
        elif row.get("image_url") and str(row["image_url"]).startswith("http"):
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
            "Try a bit more detail, like *'navy striped bath towels'* or "
            "*'white quilt for queen bed'*."
        )
        return

    for _, row in df.iterrows():
        render_product_card(row)
        st.markdown("---")


def describe_room_layout(image_bytes: bytes) -> str:
    """
    Use the vision model to describe the layout of an uploaded room photo.
    This description is passed into the image generator so it at least
    tries to keep a similar layout.
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL_VISION,
            messages=[
                {
                    "role": "system",
                    "content": "You are an interior designer. "
                    "Describe the layout and fixed elements of the room, "
                    "like where the bed/shower/vanity is, wall colors, and flooring.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this room's layout in 4–6 short sentences."},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            max_tokens=250,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        # Fallback if anything goes wrong
        return (
            "Use a clean, realistic layout typical for this room type. "
            "Include the usual fixed elements (vanity, shower, bed, etc.) "
            "but focus on styling the textiles."
        )


def generate_room_concept_image(
    room_type: str,
    textiles_request: str,
    uploaded_room_bytes: Optional[bytes],
    catalog: pd.DataFrame,
) -> Tuple[bytes, pd.DataFrame]:
    """
    Generate a concept image for a room using Market & Place products as inspiration.
    Returns (image_bytes, products_used).
    """
    layout_description = ""
    if uploaded_room_bytes:
        layout_description = describe_room_layout(uploaded_room_bytes)

    # Choose products from the catalog that best match the request
    products = filter_catalog_by_query(catalog, textiles_request or room_type, max_results=6)

    # Build product snippet used to constrain the model
    if products.empty:
        product_snippet = (
            "Market & Place textiles from the catalog – towels, sheets, quilts, and related items."
        )
    else:
        bullet_lines = []
        for _, r in products.iterrows():
            bullet_lines.append(
                f"- {r['name']} (color: {r['color']}, price: {r['price']})"
            )
        product_snippet = "\n".join(bullet_lines)

    prompt = f"""
You are designing a **Market & Place** styled interior.

Room type: {room_type}

If a reference room layout is described, keep that layout very similar:
{layout_description or "Use a simple, realistic layout typical for this room type."}

Customer request for textiles and style:
\"\"\"{textiles_request or "Create a tasteful, modern Market & Place look using coordinating textiles."}\"\"\".

VERY IMPORTANT RULES (NUCLEAR PRIORITY):
- Textiles (towels, sheets, quilts, duvet covers, curtains, throws) must be inspired ONLY by
  the following Market & Place products. Do **not** invent random new products or wild patterns.

Market & Place products (the only textile inspirations you may use):
{product_snippet}

Additional rules:
- Keep walls, floors, and fixed fixtures realistic and not overly busy.
- Focus visual interest on the textiles from the products above.
- Photorealistic rendering, soft natural lighting, no people, no brand logos.
"""

    img_resp = client.images.generate(
        model=OPENAI_MODEL_IMAGE,
        prompt=prompt,
        size="1024x1024",
        n=1,
    )
    b64 = img_resp.data[0].b64_json
    img_bytes = base64.b64decode(b64)

    return img_bytes, products


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

with st.container():
    h1, h2, h3 = st.columns([1, 2, 1])
    with h2:
        if os.path.exists(LOGO_PATH):
            st.image(
                LOGO_PATH,
                use_column_width=False,
                width=260,
            )
        st.markdown(
            "<h2 style='text-align:center; margin-bottom:0.3rem;'>Market & Place AI Stylist</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:center; margin-top:0;'>"
            "Chat with an AI stylist, search the Market & Place catalog, "
            "and generate concept visualizations using your own product file."
            "</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:center; margin-top:0.25rem;'>"
            "<a href='https://marketandplace.co/' style='text-decoration:none;'>"
            "← Return to Market & Place website</a></p>",
            unsafe_allow_html=True,
        )

st.markdown("---")

# ---------------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------------

left_col, right_col = st.columns([1, 1], gap="large")

# -------------------- LEFT: AI STYLIST & CATALOG --------------------

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


# -------------------- RIGHT: AI CONCEPT VISUALIZER --------------------

with right_col:
    st.subheader("AI concept visualizer")

    st.write(
        "Generate a **styled concept** using Market & Place products. "
        "You can either create a room concept image or preview a store shelf look."
    )

    mode = st.radio(
        "What do you want to visualize?",
        ["Room concept image", "Store shelf / showroom concept"],
        horizontal=True,
    )

    # -------- ROOM CONCEPT IMAGE (with optional room upload) --------

    if mode == "Room concept image":
        st.markdown("#### Room concept image")

        room_type = st.selectbox(
            "Room type:",
            ["Bathroom", "Bedroom", "Other"],
            index=0,
        )

        uploaded_room = st.file_uploader(
            "Optional: upload a photo of your room to guide the layout (JPG/PNG).",
            type=["jpg", "jpeg", "png"],
            key="room_photo",
        )

        textiles_request = st.text_input(
            "What textiles or style would you like to see?",
            placeholder="e.g. navy striped bath towels, neutral quilt with blue accents",
            key="room_textiles_request",
        )

        if st.button("Generate room concept image"):
            uploaded_bytes: Optional[bytes] = None
            if uploaded_room is not None:
                uploaded_bytes = uploaded_room.getvalue()

            with st.spinner("Generating room concept image…"):
                img_bytes, used_products = generate_room_concept_image(
                    room_type=room_type,
                    textiles_request=textiles_request or "",
                    uploaded_room_bytes=uploaded_bytes,
                    catalog=catalog_df,
                )

            st.markdown("### AI-generated room concept")
            st.image(img_bytes, use_column_width=True)
            st.caption(
                "Concept image generated from the Market & Place catalog. "
                "Layout is approximate; focus is on textile style inspired by your products."
            )

            if not used_products.empty:
                st.markdown("#### Products used as inspiration")
                render_product_list(used_products)

    # -------- STORE SHELF CONCEPT (using your real photo) --------

    else:
        st.markdown("#### Store shelf / showroom concept")

        st.write(
            "This uses your uploaded **store shelf** photo as the visual base and "
            "recommends Market & Place products for that shelf."
        )

        shelf_request = st.text_input(
            "Describe what you want to feature on the shelf:",
            placeholder="e.g. aqua and navy cabana stripe beach towels, matching bath mats",
            key="shelf_request",
        )

        if st.button("Show shelf and product ideas"):
            st.markdown("### Store shelf photo")

            if os.path.exists(STORE_SHELF_PHOTO):
                st.image(STORE_SHELF_PHOTO, use_column_width=True)
            else:
                st.warning(
                    f"Could not find `{STORE_SHELF_PHOTO}` in the repo. "
                    "Please make sure the file is present next to this app."
                )

            st.caption(
                "This is your real Market & Place store shelf photo. "
                "Below are catalog products that match your request."
            )

            suggestions = filter_catalog_by_query(
                catalog_df,
                shelf_request or "",
                max_results=12,
            )
            st.markdown("#### Suggested Market & Place products for this shelf")
            render_product_list(suggestions)










