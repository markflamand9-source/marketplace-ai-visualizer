import base64
import io
import os
from typing import List, Tuple

import pandas as pd
import streamlit as st
from openai import OpenAI

# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="Market & Place AI Stylist",
    page_icon="🧵",
    layout="wide",
)

# Small CSS tweak to tighten up the header / padding
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .mp-header-title {
        text-align: center;
        margin-top: 0.4rem;
        margin-bottom: 0.1rem;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .mp-header-subtitle {
        text-align: center;
        margin-top: 0.1rem;
        margin-bottom: 0.1rem;
        font-size: 0.95rem;
        color: #555555;
    }
    .mp-header-link {
        text-align: center;
        margin-top: 0.2rem;
        margin-bottom: 0.4rem;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

OPENAI_MODEL_VISION = "gpt-4.1-mini"
OPENAI_MODEL_IMAGE = "gpt-image-1"

client = OpenAI()

DATA_PATH = "market_and_place_products.xlsx"   # catalog
LOGO_PATH = "logo.png"                         # header logo
SHELF_PHOTO_PATH = "store shelf.jpg"           # base shelf reference photo


# =========================
# DATA LOADING
# =========================

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
            "Category": "category",  # if you added a Category column
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


# =========================
# SIMPLE TEXT / COLOR MATCHING
# =========================

CATEGORY_KEYWORDS = {
    "towel": ["towel", "bath towel", "hand towel", "bath sheet"],
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
            kws = CATEGORY_KEYWORDS.get(cat, [])
            for kw in kws:
                cat_mask |= df["name_lower"].str.contains(kw, na=False)
        mask &= cat_mask

    if color_terms:
        color_mask = pd.Series(False, index=df.index)
        for c in color_terms:
            color_mask |= df["color_lower"].str.contains(c, na=False)
            color_mask |= df["name_lower"].str.contains(c, na=False)
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


# =========================
# RENDERING HELPERS
# =========================

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

    for _, r in df.iterrows():
        render_product_card(r)
        st.markdown("---")


def catalog_snippet(df: pd.DataFrame, max_items: int = 6) -> str:
    lines: List[str] = []
    for _, r in df.head(max_items).iterrows():
        lines.append(f"- {r['name']} (color: {r['color']}, price: {r['price']})")
    if not lines:
        return "Market & Place textiles (towels, sheets, quilts and bedding)."
    return "\n".join(lines)


# =========================
# VISION: DESCRIBE A PHOTO
# =========================

def _image_bytes_to_data_url(image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def describe_photo_with_vision(image_bytes: bytes, task: str) -> str:
    """
    Use gpt-4.1-mini vision to describe a photo.
    task: short instruction, e.g. "Describe this room layout and style."
    """
    data_url = _image_bytes_to_data_url(image_bytes)

    resp = client.responses.create(
        model=OPENAI_MODEL_VISION,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": task + " Keep it concise (2–3 sentences).",
                    },
                    {
                        "type": "input_image",
                        "image_url": {"url": data_url},
                    },
                ],
            }
        ],
    )

    # Parse out the text from the response
    try:
        return resp.output[0].content[0].text.strip()
    except Exception:
        # Fallback in case the structure changes
        if hasattr(resp, "output_text"):
            return resp.output_text.strip()
        return ""


@st.cache_data(show_spinner=False)
def get_base_shelf_description() -> str:
    with open(SHELF_PHOTO_PATH, "rb") as f:
        b = f.read()
    return describe_photo_with_vision(b, "Describe this empty retail store shelf / aisle layout.")


# =========================
# IMAGE GENERATION HELPERS
# =========================

def generate_room_concept_image(
    room_image_bytes: bytes,
    textiles_request: str,
    catalog: pd.DataFrame,
) -> bytes:
    """
    1) Describe the uploaded room with vision.
    2) Use catalog + user request to generate a new concept image of the same room layout
       but styled with Market & Place products.
    """
    room_desc = describe_photo_with_vision(
        room_image_bytes,
        "Describe this room's layout, camera angle, major surfaces, colors and existing textiles.",
    )

    products = filter_catalog_by_query(catalog, textiles_request or "", max_results=8)
    snippet = catalog_snippet(products)

    prompt = f"""
You are an interior stylist for Market & Place.

First, reconstruct the same room as described below, keeping the **same camera angle, layout and architecture**:

Room description:
\"\"\"{room_desc}\"\"\"

Then, style this room using **only Market & Place products** summarised here:

{snippet}

User request for textiles:
\"\"\"{textiles_request}\"\"\"

Respect the type of products (towels vs sheets vs quilts, etc.), and place them logically
in the scene (on towel bars, shelves, bed, etc.). Do not invent completely different products.
Generate a single photorealistic image.
    """.strip()

    img_resp = client.images.generate(
        model=OPENAI_MODEL_IMAGE,
        prompt=prompt,
        size="1024x1024",
        n=1,
    )

    b64 = img_resp.data[0].b64_json
    return base64.b64decode(b64)


def generate_store_shelf_concept(
    shelf_text_request: str,
    catalog: pd.DataFrame,
) -> bytes:
    """
    1) Describe the fixed 'store shelf.jpg' image with vision (cached).
    2) Generate a concept image of that shelf filled with Market & Place products matching the request.
    """
    with open(SHELF_PHOTO_PATH, "rb") as f:
        shelf_bytes = f.read()

    base_desc = get_base_shelf_description()
    products = filter_catalog_by_query(catalog, shelf_text_request or "", max_results=8)
    snippet = catalog_snippet(products)

    prompt = f"""
You are generating a **retail store shelf / aisle** concept image for Market & Place.

Recreate the same store shelf layout described below, keeping the same camera angle, feeling of depth,
shelf structure, floor and lighting:

Shelf layout description:
\"\"\"{base_desc}\"\"\"

Fill these shelves **only with Market & Place products** summarised here:

{snippet}

User request for what should be on the shelf:
\"\"\"{shelf_text_request}\"\"\"

Use stacks of folded textiles and some neatly hung pieces. Do not add bathtubs, sinks, beds, people or logos.
Generate a single, photorealistic store interior image.
    """.strip()

    img_resp = client.images.generate(
        model=OPENAI_MODEL_IMAGE,
        prompt=prompt,
        size="1024x1024",
        n=1,
    )

    b64 = img_resp.data[0].b64_json
    return base64.b64decode(b64)


# =========================
# HEADER
# =========================

header_cols = st.columns([1, 2, 1])
with header_cols[1]:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_column_width=False, width=260)
    st.markdown(
        "<div class='mp-header-title'>Market & Place AI Stylist</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='mp-header-subtitle'>Chat with an AI stylist, search the Market & Place catalog, "
        "and generate concept visualizations using your own product file.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='mp-header-link'><a href='https://marketandplace.co/' "
        "style='text-decoration:none;'>← Return to Market & Place website</a></div>",
        unsafe_allow_html=True,
    )

st.markdown("---")


# =========================
# MAIN LAYOUT
# =========================

left_col, right_col = st.columns([1, 1], gap="large")

# ----- LEFT: Ask the AI stylist + Quick catalog peek -----

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


# ----- RIGHT: AI concept visualizer (rooms + store shelf) -----

with right_col:
    st.subheader("AI concept visualizer")

    st.write(
        "Generate a **styled concept** using Market & Place products. "
        "Upload a room photo to see it restyled, or create a store shelf concept."
    )

    mode = st.radio(
        "What do you want to visualize?",
        ["Room concept image", "Store shelf / showroom concept"],
        horizontal=True,
    )

    # --- ROOM MODE ---
    if mode == "Room concept image":
        st.markdown("#### Room concept image")

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
                room_bytes = uploaded_room.read()
                with st.spinner("Generating room concept…"):
                    try:
                        img_bytes = generate_room_concept_image(
                            room_bytes,
                            textiles_request or "",
                            catalog_df,
                        )
                        st.markdown("### AI-generated room concept")
                        st.image(img_bytes, use_column_width=True)
                        st.caption(
                            "Concept generated from your room photo and the Market & Place catalog. "
                            "The layout and angle are based on your image."
                        )
                    except Exception as e:
                        st.error(f"Image generation failed: {e}")

    # --- STORE SHELF MODE ---
    else:
        st.markdown("#### Store shelf / showroom concept")

        st.caption(
            "This uses your real Market & Place shelf photo as the layout reference, "
            "then fills it with products from the catalog."
        )
        if os.path.exists(SHELF_PHOTO_PATH):
            st.image(SHELF_PHOTO_PATH, caption="Store shelf photo (layout reference)", use_column_width=True)

        shelf_request = st.text_input(
            "Describe what you want on the shelf:",
            placeholder="e.g. cabana stripe beach towels in aqua and navy, folded stacks and matching mats",
            key="shelf_request",
        )

        if st.button("Generate store shelf concept image"):
            with st.spinner("Generating store shelf concept…"):
                try:
                    img_bytes = generate_store_shelf_concept(shelf_request or "", catalog_df)
                    st.markdown("### AI-generated store shelf concept")
                    st.image(img_bytes, use_column_width=True)
                    st.caption(
                        "Concept generated using the base shelf layout and Market & Place products "
                        "that match your description."
                    )
                except Exception as e:
                    st.error(f"Image generation failed: {e}")








