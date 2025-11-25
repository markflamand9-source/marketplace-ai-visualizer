import base64
import io
import os
from typing import List, Tuple

import pandas as pd
import streamlit as st
from openai import OpenAI

# =========================
#  CONFIG
# =========================

st.set_page_config(
    page_title="Market & Place AI Stylist",
    page_icon="🧵",
    layout="wide",
)

# IMPORTANT: use a vision-capable model here
OPENAI_MODEL_VISION = "gpt-4.1"      # was "gpt-4.1-mini" (no image support)
OPENAI_MODEL_IMAGE = "gpt-image-1"

client = OpenAI()

DATA_PATH = "market_and_place_products.xlsx"  # same folder as this file
LOGO_PATH = "logo.png"
STORE_SHELF_PHOTO_PATH = "store shelf.jpg"    # base store shelf layout photo


# =========================
#  DATA LOADING
# =========================

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
            "Category": "category",          # optional
            "Image file": "image_file",      # optional local image filename
        }
    )

    # Ensure string columns
    for col in ["name", "color", "amazon", "image_url", "category", "image_file"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Helper lowercase columns for matching
    df["name_lower"] = df["name"].str.lower()
    df["color_lower"] = df["color"].str.lower()
    if "category" in df.columns:
        df["category_lower"] = df["category"].str.lower()
    else:
        df["category_lower"] = ""

    return df


catalog_df = load_catalog(DATA_PATH)


# =========================
#  HELPER FUNCTIONS
# =========================

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
    Tiny keyword extractor.
    Returns (category_terms, color_terms).
    """
    t = text.lower()
    found_cats: List[str] = []
    found_colors: List[str] = []

    # categories
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

    # de-dupe & preserve order
    found_cats = list(dict.fromkeys(found_cats))
    found_colors = list(dict.fromkeys(found_colors))
    return found_cats, found_colors


def filter_catalog_by_query(df: pd.DataFrame, query: str, max_results: int = 8) -> pd.DataFrame:
    """
    Looser matching:
    - Use category / color hints if present.
    - Fall back to keyword match on product name.
    """
    if not query:
        return df.head(max_results)

    q = query.lower()
    cat_terms, color_terms = detect_category_terms(q)

    mask = pd.Series(True, index=df.index)

    # Category-based name matching
    if cat_terms:
        cat_mask = pd.Series(False, index=df.index)
        for cat in cat_terms:
            keywords = CATEGORY_KEYWORDS.get(cat, [])
            for kw in keywords:
                cat_mask |= df["name_lower"].str.contains(kw, case=False, na=False)
        mask &= cat_mask

    # Try to use 'category' column as a hint, if it exists
    if "category_lower" in df.columns and any(term in q for term in ["bath", "bed", "beach"]):
        cat_hint_mask = pd.Series(False, index=df.index)
        for term in ["bath", "bed", "beach"]:
            if term in q:
                cat_hint_mask |= df["category_lower"].str.contains(term, case=False, na=False)
        # only apply if it doesn't kill everything
        narrowed = df[mask & cat_hint_mask]
        if not narrowed.empty:
            mask &= cat_hint_mask

    # Colors (soft filter)
    if color_terms:
        color_mask = pd.Series(False, index=df.index)
        for c in color_terms:
            color_mask |= (
                df["color_lower"].str.contains(c, case=False, na=False)
                | df["name_lower"].str.contains(c, case=False, na=False)
            )
        narrowed = df[mask & color_mask]
        if not narrowed.empty:
            mask &= color_mask

    results = df[mask]

    # Final fallback – keyword search on name
    if results.empty:
        simple_mask = pd.Series(False, index=df.index)
        for word in q.split():
            simple_mask |= df["name_lower"].str.contains(word, case=False, na=False)
        results = df[simple_mask]

    return results.head(max_results)


def get_local_or_remote_image(row) -> str:
    """
    Prefer a local product image if provided; fall back to image_url.
    Returns a path/URL suitable for st.image.
    """
    # Local image from repo
    if "image_file" in row and isinstance(row["image_file"], str) and row["image_file"].strip():
        local_path = row["image_file"].strip()
        if os.path.exists(local_path):
            return local_path

    # Remote URL
    if "image_url" in row and str(row["image_url"]).startswith("http"):
        return row["image_url"]

    return ""


def render_product_card(row):
    cols = st.columns([1, 2.5])
    with cols[0]:
        img_src = get_local_or_remote_image(row)
        if img_src:
            st.image(img_src, use_column_width=True)
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


def encode_file_to_data_url(binary: bytes, mime_type: str = "image/jpeg") -> str:
    b64 = base64.b64encode(binary).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def describe_photo_with_vision(image_bytes: bytes, mime_type: str, guidance: str) -> str:
    """
    Use GPT-4.1 with vision to describe a photo (layout, surfaces, etc.).
    """
    data_url = encode_file_to_data_url(image_bytes, mime_type=mime_type)

    resp = client.responses.create(
        model=OPENAI_MODEL_VISION,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": guidance,
                    },
                    {
                        "type": "input_image",
                        "image_url": {"url": data_url},
                    },
                ],
            }
        ],
    )

    # Text output lives in resp.output[0].content[0].text
    return resp.output[0].content[0].text


def generate_concept_image(prompt: str) -> bytes:
    """
    Text-to-image using gpt-image-1.
    """
    img_resp = client.images.generate(
        model=OPENAI_MODEL_IMAGE,
        prompt=prompt,
        size="1024x1024",
        n=1,
    )
    b64 = img_resp.data[0].b64_json
    return base64.b64decode(b64)


@st.cache_data(show_spinner=False)
def describe_base_shelf_photo() -> str:
    """
    Shelf layout description (cached – same photo every time).
    """
    if not os.path.exists(STORE_SHELF_PHOTO_PATH):
        return "a neutral store interior with empty white shelving bays"

    with open(STORE_SHELF_PHOTO_PATH, "rb") as f:
        bytes_data = f.read()

    return describe_photo_with_vision(
        bytes_data,
        mime_type="image/jpeg",
        guidance=(
            "Describe the layout and structure of this store shelf photo in a few sentences. "
            "Focus on how many bays, the positions of shelves, perspective and general feel. "
            "Do NOT mention any brand names."
        ),
    )


def build_product_snippet(products: pd.DataFrame) -> str:
    lines = []
    for _, r in products.iterrows():
        cat = r.get("category", "")
        if cat and cat.lower() != "nan":
            cat_part = f" [{cat}]"
        else:
            cat_part = ""
        lines.append(f"- {r['name']}{cat_part} (color: {r['color']}, price: {r['price']})")
    return "\n".join(lines) if lines else "Market & Place towels, bedding and textiles."


def generate_room_concept_image(room_bytes: bytes, mime_type: str, textiles_request: str, products: pd.DataFrame) -> bytes:
    """
    2-step process:
    1) Describe the uploaded room photo.
    2) Ask gpt-image-1 to render a new image with the same layout but updated textiles
       using the selected Market & Place products as inspiration.
    """
    room_desc = describe_photo_with_vision(
        room_bytes,
        mime_type=mime_type,
        guidance=(
            "You are helping an interior stylist. "
            "Describe this room: layout, walls, furniture/fixtures and where textiles "
            "(towels, bedding, shower curtains, rugs, etc.) appear or could appear. "
            "Keep it under 6 sentences."
        ),
    )

    product_snippet = build_product_snippet(products)

    prompt = f"""
You are generating a **styled concept image** for a customer using Market & Place products.

Room photo description (from the real uploaded photo):
{room_desc}

Customer request for new textiles:
\"\"\"{textiles_request}\"\"\".

Market & Place products that must inspire ALL textiles in the new image:
{product_snippet}

VERY IMPORTANT RULES:
- Keep the same room layout, camera viewpoint, doors, windows, walls and fixed fixtures
  (vanity, toilet, tub, bed, side tables, etc.) as described in the original room photo.
- Only change or add **textiles**: towels, shower curtains, bath mats, bedding, throws, cushions.
- Textiles must match the colors and patterns implied by the Market & Place products above.
- Do NOT invent totally different product types or wild colors that aren't in the list.
- No people. No visible brand logos.

Render a realistic, well-lit scene that looks like the same room, now styled with those Market & Place products.
"""

    return generate_concept_image(prompt)


def generate_store_shelf_concept(shelf_query: str, products: pd.DataFrame) -> bytes:
    base_layout_desc = describe_base_shelf_photo()
    product_snippet = build_product_snippet(products)

    prompt = f"""
You are generating a **retail store shelf concept** for Market & Place.

Base layout description (this comes from a real Market & Place empty shelf photo):
{base_layout_desc}

Customer request for this shelf:
\"\"\"{shelf_query}\"\"\".

Market & Place products that must inspire everything on the shelves:
{product_snippet}

VERY IMPORTANT RULES:
- The scene must clearly show the same type of store shelving as in the base layout description:
  long bays with central gondola and side shelving – not a home bathroom or bedroom.
- Keep the general geometry of the base layout (aisle, number of bays, perspective).
- Populate shelves only with folded or hanging textiles (towels, sheets, quilts, etc.) based on the products.
- Do NOT show beds, toilets, sinks, or other home furniture.
- No people and no brand logos.

Render a clean, well-lit store interior with those Market & Place products displayed.
"""

    return generate_concept_image(prompt)


# =========================
#  HEADER (CONDENSED)
# =========================

header_cols = st.columns([1, 2, 1])
with header_cols[1]:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=300)  # smaller logo
    st.markdown(
        "<h2 style='text-align:center; margin-bottom:0.25rem;'>Market & Place AI Stylist</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; font-size:0.95rem; margin-bottom:0.5rem;'>"
        "Chat with an AI stylist, search the Market & Place catalog, and generate concept "
        "visualizations using your own product file."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; font-size:0.9rem; margin-top:0;'>"
        "<a href='https://marketandplace.co/' style='text-decoration:none;'>"
        "← Return to Market & Place website</a></p>",
        unsafe_allow_html=True,
    )

st.markdown("---")


# =========================
#  MAIN LAYOUT
# =========================

left_col, right_col = st.columns([1, 1], gap="large")

# ---------- LEFT: ASK THE STYLIST + QUICK CATALOG ----------

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


# ---------- RIGHT: AI CONCEPT VISUALIZER ----------

with right_col:
    st.subheader("AI concept visualizer")

    st.write(
        "Generate a **styled concept** using Market & Place products. "
        "You can either style an uploaded room photo or create a store shelf concept."
    )

    mode = st.radio(
        "What do you want to visualize?",
        ["Room concept from my photo", "Store shelf / showroom concept"],
        horizontal=True,
    )

    # ===== ROOM MODE (with AI IMAGE) =====
    if mode == "Room concept from my photo":
        st.markdown("#### Room concept from your photo")

        uploaded_room = st.file_uploader(
            "Upload a photo of your room (bathroom, bedroom, etc.):",
            type=["jpg", "jpeg", "png"],
        )

        textiles_request = st.text_input(
            "What textiles would you like to add or change?",
            placeholder="e.g. luxury towels, cabana stripe shower curtain, white quilt, navy sheets",
            key="room_textiles_request",
        )

        if st.button("Generate room concept"):
            if uploaded_room is None:
                st.error("Please upload a room photo first.")
            else:
                mime_type = uploaded_room.type or "image/jpeg"
                room_bytes = uploaded_room.getvalue()

                with st.spinner("Styling your room with Market & Place products…"):
                    # choose catalog products based on request
                    suggested_products = filter_catalog_by_query(
                        catalog_df,
                        textiles_request or "",
                        max_results=8,
                    )

                    img_bytes = generate_room_concept_image(
                        room_bytes,
                        mime_type=mime_type,
                        textiles_request=textiles_request or "updated textiles",
                        products=suggested_products,
                    )

                st.markdown("##### Original room photo")
                st.image(room_bytes, use_column_width=True)

                st.markdown("##### AI-styled room concept")
                st.image(img_bytes, use_column_width=True)

                st.markdown("##### Market & Place products used as inspiration")
                render_product_list(suggested_products)

    # ===== STORE SHELF MODE =====
    else:
        st.markdown("#### Store shelf / showroom concept")

        if os.path.exists(STORE_SHELF_PHOTO_PATH):
            st.markdown("**Store shelf layout reference (real Market & Place photo)**")
            st.image(STORE_SHELF_PHOTO_PATH, use_column_width=True)
        else:
            st.info("The base store shelf photo was not found in the repo.")

        shelf_query = st.text_input(
            "Describe the shelf you want to visualize:",
            placeholder="e.g. cabana stripe beach towels in aqua and navy, folded stacks and matching mats",
            key="shelf_request",
        )

        if st.button("Generate store shelf concept image"):
            with st.spinner("Designing your Market & Place store shelf…"):
                shelf_products = filter_catalog_by_query(
                    catalog_df, shelf_query or "", max_results=8
                )

                img_bytes = generate_store_shelf_concept(
                    shelf_query or "display Market & Place textiles on the shelf",
                    products=shelf_products,
                )

            st.markdown("##### AI-generated store shelf concept")
            st.image(img_bytes, use_column_width=True)
            st.markdown("##### Market & Place products used as inspiration")
            render_product_list(shelf_products)








