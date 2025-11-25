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
    layout="wide",
)

OPENAI_MODEL_VISION = "gpt-4.1-mini"
OPENAI_MODEL_IMAGE = "gpt-image-1"

client = OpenAI()

DATA_PATH = "market_and_place_products.xlsx"  # Excel catalog
LOGO_PATH = "logo.png"                        # top logo
SHELF_PHOTO_PATH = "store shelf.jpg"         # base store shelf photo


# ---------- DATA LOADING ----------

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
            "Category": "category",
        }
    )

    # Safety: ensure strings
    for col in ["name", "color", "amazon", "image_url", "category"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    df["name_lower"] = df["name"].str.lower()
    df["color_lower"] = df["color"].str.lower()
    df["category_lower"] = df.get("category", "").str.lower()

    return df


catalog_df = load_catalog(DATA_PATH)


# ---------- TEXT SEARCH HELPERS ----------

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


def filter_catalog_by_query(
    df: pd.DataFrame,
    query: str,
    max_results: int = 8,
) -> pd.DataFrame:
    if not query:
        return df.head(max_results)

    q = query.lower()
    cat_terms, color_terms = detect_category_terms(q)

    mask = pd.Series(True, index=df.index)

    # Category constraints (product *type* like towel / sheet / quilt)
    if cat_terms:
        cat_mask = pd.Series(False, index=df.index)
        for cat in cat_terms:
            keywords = CATEGORY_KEYWORDS.get(cat, [])
            for kw in keywords:
                cat_mask |= df["name_lower"].str.contains(kw, na=False)
        mask &= cat_mask

    # Color constraints (soft)
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

    # Fallback: simple substring match on product name if empty
    if results.empty:
        simple_mask = pd.Series(False, index=df.index)
        for word in q.split():
            simple_mask |= df["name_lower"].str.contains(word, na=False)
        results = df[simple_mask]

    return results.head(max_results)


def filter_catalog_by_category(df: pd.DataFrame, shelf_cat: str) -> pd.DataFrame:
    if "category_lower" not in df.columns or not shelf_cat:
        return df
    shelf_cat = shelf_cat.lower()
    if shelf_cat == "any":
        return df
    return df[df["category_lower"].str.contains(shelf_cat, na=False)]


# ---------- PRODUCT RENDERING ----------

def render_product_card(row: pd.Series):
    cols = st.columns([1, 2.5])
    with cols[0]:
        img_shown = False
        if row.get("image_url") and str(row["image_url"]).startswith("http"):
            try:
                st.image(row["image_url"], use_column_width=True)
                img_shown = True
            except Exception:
                img_shown = False

        # Optional: try local image if URL fails / is missing
        if not img_shown:
            # attempt to use "<Product name>.jpg"
            base_name = f"{row['name']}.jpg"
            if os.path.exists(base_name):
                st.image(base_name, use_column_width=True)

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


# ---------- SHELF PHOTO DESCRIPTION (VISION) ----------

@st.cache_data(show_spinner=False)
def describe_shelf_photo() -> str:
    """
    Use the vision model to describe the real 'store shelf.jpg' layout
    so that the image model can mimic it.
    Cached so we only pay once.
    """
    if not os.path.exists(SHELF_PHOTO_PATH):
        return (
            "A large, neutral store aisle with long empty shelves on both sides. "
            "Modern lighting above, tiled floor below."
        )

    with open(SHELF_PHOTO_PATH, "rb") as f:
        img_bytes = f.read()

    b64 = base64.b64encode(img_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"

    resp = client.responses.create(
        model=OPENAI_MODEL_VISION,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "You see a store shelf photo. "
                            "Describe the layout, camera angle, perspective, "
                            "materials, lighting, and main geometry in ~250 words. "
                            "Be as precise as possible so another model can reconstruct "
                            "an almost identical empty shelf scene."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": {"url": data_url},
                    },
                ],
            }
        ],
    )

    try:
        description = resp.output[0].content[0].text
    except Exception:
        description = (
            "A neutral modern store aisle with central gondola shelving and "
            "sidewall shelves, bright even lighting, and tiled floor."
        )

    return description


# ---------- IMAGE GENERATION HELPERS ----------

def build_product_snippet(products: pd.DataFrame) -> str:
    if products.empty:
        return "Use Market & Place textiles from our catalog (towels, quilts, sheets)."

    lines = []
    for _, r in products.iterrows():
        lines.append(f"- {r['name']} (color: {r['color']}, price: {r['price']})")
    return "\n".join(lines)


def generate_room_concept(
    room_type: str,
    user_text: str,
    ref_image: bytes | None,
) -> bytes:
    """
    Generate a room concept image, trying to keep the layout similar if a reference
    image is provided, and using Market & Place products as textiles.
    """
    room_desc = ""
    if ref_image is not None:
        b64 = base64.b64encode(ref_image).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{b64}"

        vision_resp = client.responses.create(
            model=OPENAI_MODEL_VISION,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Describe this room in ~200 words focusing on layout, "
                                "viewpoint, and surfaces so an image model can recreate it. "
                                f"The room type is: {room_type}."
                            ),
                        },
                        {
                            "type": "input_image",
                            "image_url": {"url": data_url},
                        },
                    ],
                }
            ],
        )
        try:
            room_desc = vision_resp.output[0].content[0].text
        except Exception:
            room_desc = f"A {room_type} with neutral finishes."

    # Small product snippet to encourage correct textiles
    products_for_room = filter_catalog_by_query(
        catalog_df,
        user_text or room_type,
        max_results=6,
    )
    product_snippet = build_product_snippet(products_for_room)

    prompt = f"""
You are generating a styled {room_type} concept image for Market & Place.

If you received a room description, keep:
- The same layout, camera angle, windows, major fixtures and furniture.
- Only change the **textiles** (bed linens, towels, shower curtains, rugs, etc.).

Room description (if given):
{room_desc}

Use ONLY Market & Place products as inspiration for the textiles, matching:
{product_snippet}

User request / styling direction:
\"\"\"{user_text}\"\"\".

Do NOT show any brands or logos. The result should look photorealistic.
"""

    img_resp = client.images.generate(
        model=OPENAI_MODEL_IMAGE,
        prompt=prompt,
        size="1024x1024",
        n=1,
    )
    b64_out = img_resp.data[0].b64_json
    return base64.b64decode(b64_out)


def generate_store_shelf_concept(
    shelf_query: str,
    shelf_category: str,
) -> bytes:
    """
    Generate a store shelf image that closely mimics the real 'store shelf.jpg'
    layout, but populated with Market & Place products matching shelf_query
    and shelf_category.
    """
    base_desc = describe_shelf_photo()

    # Products from catalog, filtered by shelf category then text query
    base_df = filter_catalog_by_category(catalog_df, shelf_category)
    shelf_products = filter_catalog_by_query(base_df, shelf_query or "", max_results=8)
    product_snippet = build_product_snippet(shelf_products)

    prompt = f"""
You are generating a **retail store shelf / showroom** concept for Market & Place.

First, here is a detailed description of our real empty store shelf photo.
You must keep the same layout, camera angle, proportions, and overall environment:

{base_desc}

Now, populate those shelves ONLY with Market & Place products, following this list
for colors, patterns, and product types (do not invent totally different products):

{product_snippet}

User styling direction for how the shelf should look:
\"\"\"{shelf_query}\"\"\".

Critical rules:
- It MUST look like a store shelf / showroom, not a home bathroom or bedroom.
- Keep the shelves, floor, ceiling, and walls like the base scene.
- Arrange folded stacks, hanging textiles, and maybe small stacks of matching items.
- No bathtubs, sinks, toilets, beds, or home furniture.
- No people, no brand logos.
"""

    img_resp = client.images.generate(
        model=OPENAI_MODEL_IMAGE,
        prompt=prompt,
        size="1024x1024",
        n=1,
    )
    b64_out = img_resp.data[0].b64_json
    return base64.b64decode(b64_out)


# ---------- HEADER (COMPACT) ----------

header_cols = st.columns([1, 2.5, 1])
with header_cols[1]:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_column_width=False, width=320)
    st.markdown(
        "<h2 style='text-align:center; margin-top:0.4rem; margin-bottom:0.2rem;'>"
        "Market & Place AI Stylist</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; margin-bottom:0.4rem;'>"
        "Chat with an AI stylist, search the Market & Place catalog, "
        "and generate concept visualizations using your own product file."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; margin-bottom:0.2rem;'>"
        "<a href='https://marketandplace.co/' style='text-decoration:none;'>"
        "← Return to Market & Place website</a></p>",
        unsafe_allow_html=True,
    )

st.markdown("<hr style='margin-top:0.6rem; margin-bottom:0.8rem;'>", unsafe_allow_html=True)


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


# ----- RIGHT: AI CONCEPT VISUALIZER (ROOMS + STORE SHELF) -----

with right_col:
    st.subheader("AI concept visualizer")

    st.write(
        "Generate styled concepts using Market & Place products. "
        "You can either create a room concept or a store shelf concept."
    )

    mode = st.radio(
        "What do you want to visualize?",
        ["In my room photo", "Store shelf / showroom"],
        horizontal=True,
    )

    # --- ROOM CONCEPT MODE ---
    if mode == "In my room photo":
        st.markdown("#### Room concept (Market & Place textiles)")

        room_type = st.selectbox(
            "Room type:",
            ["bathroom", "bedroom", "living room"],
            index=0,
        )

        uploaded_room = st.file_uploader(
            "Upload a photo of your room (recommended):",
            type=["jpg", "jpeg", "png"],
            help="The AI will try to keep the same layout and only change textiles.",
        )

        what_change = st.text_input(
            "What would you like to visualize?",
            placeholder="e.g. navy striped towels and bath mat, neutral quilt with blue pillows",
            key="room_change_text",
        )

        if st.button("Generate room concept image"):
            if not what_change.strip():
                st.warning("Please describe what you want to change (towels, quilt, etc.) first.")
            else:
                ref_bytes = uploaded_room.read() if uploaded_room is not None else None
                with st.spinner("Generating room concept…"):
                    img_bytes = generate_room_concept(room_type, what_change, ref_bytes)

                st.markdown("### AI-generated room concept")
                st.image(img_bytes, use_column_width=True)
                st.caption(
                    "Concept image generated from your room description using Market & Place "
                    "products as textile inspiration."
                )

    # --- STORE SHELF MODE ---
    else:
        st.markdown("#### Store shelf / showroom concept")

        if os.path.exists(SHELF_PHOTO_PATH):
            st.image(
                SHELF_PHOTO_PATH,
                caption="Real Market & Place store shelf photo (layout reference).",
                use_column_width=True,
            )

        shelf_query = st.text_input(
            "What textiles do you want to display on this shelf?",
            placeholder="e.g. cabana stripe beach towels in aqua and navy, folded stacks and matching mats",
            key="shelf_query_text",
        )

        # Shelf category from catalog (bathroom / bedroom / beach, etc.)
        cat_options = ["Any"]
        if "category" in catalog_df.columns:
            cat_options.extend(sorted(set(catalog_df["category"])))
        cat_options = list(dict.fromkeys(cat_options))  # unique, preserve order

        shelf_category = st.selectbox(
            "Shelf category (from catalog):",
            options=cat_options,
            index=0,
            help="Filters which products in the catalog are allowed on this shelf.",
        )

        if st.button("Generate store shelf concept image"):
            with st.spinner("Generating store shelf concept…"):
                img_bytes = generate_store_shelf_concept(shelf_query, shelf_category)

            st.markdown("### AI-generated store shelf concept")
            st.image(img_bytes, use_column_width=True)
            st.caption(
                "Concept image generated to closely match your real store shelf photo, "
                "using only Market & Place catalog products as inspiration."
            )










