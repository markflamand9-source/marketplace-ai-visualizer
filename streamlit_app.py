import base64
import io
import os
from pathlib import Path
from typing import List, Tuple
from uuid import uuid4

import pandas as pd
import streamlit as st
from openai import OpenAI, BadRequestError

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
LOGO_PATH = "logo.png"
SHELF_PHOTO_PATH = "store shelf.jpg"          # base store shelf image


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
            "Category": "category",
        }
    )

    for col in ["name", "color", "amazon", "image_url", "category"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    df["name_lower"] = df["name"].str.lower()
    df["color_lower"] = df["color"].str.lower()
    df["category_lower"] = df.get("category", "").str.lower()

    return df


catalog_df = load_catalog(DATA_PATH)


# ---------- SIMPLE NLP HELPERS ----------

CATEGORY_KEYWORDS = {
    "towel": ["towel", "bath towel", "hand towel"],
    "beach_towel": ["beach towel", "cabana"],
    "sheet": ["sheet", "sheet set"],
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
    """Return (category_terms, color_terms) from a free-text query."""
    t = text.lower()
    found_cats: List[str] = []
    found_colors: List[str] = []

    # category
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

    # de-duplicate
    found_cats = list(dict.fromkeys(found_cats))
    found_colors = list(dict.fromkeys(found_colors))
    return found_cats, found_colors


def filter_catalog_by_query(df: pd.DataFrame, query: str, max_results: int = 8) -> pd.DataFrame:
    """
    Looser matching:
      * filter by detected category words (towels vs sheets vs quilts, etc.)
      * optionally narrow by color words
      * fall back to keyword search on product name
    """
    if not query:
        return df.head(max_results)

    q = query.lower()
    cat_terms, color_terms = detect_category_terms(q)

    mask = pd.Series(True, index=df.index)

    # category constraints based on product name
    if cat_terms:
        cat_mask = pd.Series(False, index=df.index)
        for cat in cat_terms:
            for kw in CATEGORY_KEYWORDS.get(cat, []):
                cat_mask |= df["name_lower"].str.contains(kw, case=False, na=False)
        mask &= cat_mask

    # color constraints (soft – if they remove everything, skip)
    if color_terms:
        color_mask = pd.Series(False, index=df.index)
        for c in color_terms:
            color_mask |= df["color_lower"].str.contains(c, case=False, na=False)
            color_mask |= df["name_lower"].str.contains(c, case=False, na=False)

        narrowed = df[mask & color_mask]
        if not narrowed.empty:
            mask &= color_mask

    results = df[mask]

    # Fallback: general keyword search
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
        img_src = None

        # prefer local file if it exists, otherwise use image_url (if http)
        if isinstance(row.get("image_url"), str) and row["image_url"].startswith("http"):
            img_src = row["image_url"]
        else:
            # try to find a local image file with a loose match on name
            safe_stem = "".join(
                ch for ch in f"{row['name']} {row['color']}" if ch.isalnum() or ch in (" ", "_", "-")
            )
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                candidate = Path(f"{safe_stem}{ext}")
                if candidate.exists():
                    img_src = str(candidate)
                    break

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


# ---------- IMAGE / VISION HELPERS ----------


def save_uploaded_to_tmp(uploaded_file) -> str:
    """Save a Streamlit UploadedFile to a temp path and return the path."""
    ext = Path(uploaded_file.name).suffix or ".jpg"
    tmp_dir = Path("tmp_uploads")
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / f"upload_{uuid4().hex}{ext}"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(tmp_path)


@st.cache_data(show_spinner=False)
def describe_photo_with_vision(image_path: str, extra_instruction: str = "") -> str:
    """
    Use the Responses API with vision to get a short description of an image.
    NOTE: we pass the local path as `image_url` (the hosting layer maps it to a URL).
    """
    system_prompt = (
        "You are a concise visual describer for interior design. "
        "Describe the room layout, major fixtures (e.g., shower, vanity, bed), "
        "colors, and where textiles currently appear. "
        "Keep it to 4–6 sentences."
    )

    if extra_instruction:
        system_prompt += " " + extra_instruction

    resp = client.responses.create(
        model=OPENAI_MODEL_VISION,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": system_prompt},
                    {"type": "input_image", "image_url": image_path},
                ],
            }
        ],
    )

    return resp.output[0].content[0].text


def build_product_snippet(products: pd.DataFrame) -> str:
    if products.empty:
        return "Market & Place towels, bath mats, quilts, sheets and other textiles."
    lines = []
    for _, r in products.iterrows():
        lines.append(f"- {r['name']} (color: {r['color']}, price: {r['price']})")
    return "\n".join(lines)


def generate_room_concept_image(uploaded_room, textiles_request: str, products: pd.DataFrame) -> bytes:
    """
    1) Describe the uploaded room using vision.
    2) Generate a new concept image using only Market & Place products for textiles.
    Returns raw PNG bytes.
    """
    room_path = save_uploaded_to_tmp(uploaded_room)
    room_desc = describe_photo_with_vision(room_path)

    product_snippet = build_product_snippet(products)

    user_request = textiles_request.strip() or "Tasteful Market & Place textiles that complement the space."

    prompt = f"""
You are an interior design visualizer for Market & Place.

Base room description (from a real customer photo):
{room_desc}

Customer request about textiles:
\"\"\"{user_request}\"\"\".

Market & Place products available (only use these for colors, patterns, and textile types):
{product_snippet}

REQUIREMENTS (very important):
- Keep the same room layout, angle, and fixtures as the described room.
- Only change or add textiles: towels, bath mats, shower curtains, window curtains,
  quilts, comforters, sheets, and decorative pillows.
- Do NOT change the architecture, furniture shapes, or camera angle.
- Do NOT add logos or non-textile products.
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
def describe_shelf_photo() -> str:
    """Describe the base store shelf photo once (cached)."""
    if not Path(SHELF_PHOTO_PATH).exists():
        return "A clean, neutral-colored store shelf with multiple long bays and empty shelves."

    resp = client.responses.create(
        model=OPENAI_MODEL_VISION,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Describe this store shelf photo in detail. "
                            "Focus on the shelving layout, angle, lighting and materials. "
                            "Do NOT mention any brand names."
                        ),
                    },
                    {"type": "input_image", "image_url": SHELF_PHOTO_PATH},
                ],
            }
        ],
    )
    return resp.output[0].content[0].text


def generate_store_shelf_concept(shelf_query: str, products: pd.DataFrame) -> bytes:
    """
    Generate a store shelf concept image based on the base shelf photo description
    and selected Market & Place products.
    """
    base_desc = describe_shelf_photo()
    product_snippet = build_product_snippet(products)

    user_request = shelf_query.strip() or "Neatly merchandised towels and textiles."

    prompt = f"""
You are generating a **store shelf merchandising** concept for Market & Place textiles.

Base shelf photo description:
{base_desc}

Customer shelf request:
\"\"\"{user_request}\"\"\".

Market & Place products to display:
{product_snippet}

REQUIREMENTS:
- Keep the same general shelf layout, angle, and lighting as the described base shelf.
- Populate the shelves neatly with stacks and some hanging pieces of those textiles.
- Only show towels, bath mats, sheets, quilts, or similar textiles.
- Do NOT show beds, bathtubs, sinks, or non-textile products.
- No people and no visible logos.
"""

    img_resp = client.images.generate(
        model=OPENAI_MODEL_IMAGE,
        prompt=prompt,
        size="1024x1024",
        n=1,
    )
    b64 = img_resp.data[0].b64_json
    return base64.b64decode(b64)


# ---------- HEADER (CONDENSED) ----------

def render_header():
    logo_b64 = None
    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode("utf-8")

    if logo_b64:
        st.markdown(
            f"""
        <div style="text-align:center; padding-top:10px; padding-bottom:10px;">
            <img src="data:image/png;base64,{logo_b64}"
                 style="max-width:220px; margin-bottom:6px;" />
            <h1 style="font-size:30px; margin:0;">Market & Place AI Stylist</h1>
            <p style="font-size:15px; margin:4px 0 0 0; color:#555;">
                Chat with an AI stylist, search the Market & Place catalog,
                and generate concept visualizations using your own product file.
            </p>
            <p style="margin-top:4px; font-size:14px;">
                <a href="https://marketandplace.co/"
                   style="text-decoration:none;">← Return to Market & Place website</a>
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.title("Market & Place AI Stylist")
        st.write(
            "Chat with an AI stylist, search the Market & Place catalog, "
            "and generate concept visualizations using your own product file."
        )
        st.markdown(
            "[← Return to Market & Place website](https://marketandplace.co/)"
        )


render_header()
st.markdown("---")


# ---------- MAIN LAYOUT ----------

left_col, right_col = st.columns([1, 1], gap="large")

# ----- LEFT: TEXT STYLIST + QUICK CATALOG PEEK -----

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


# ----- RIGHT: AI CONCEPT VISUALIZER -----

with right_col:
    st.subheader("AI concept visualizer")
    st.write(
        "Generate a **styled concept** using Market & Place products. "
        "Upload a room photo for a room concept, or generate a store shelf concept."
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
        )

        textiles_request = st.text_input(
            "What textiles would you like to add or change?",
            placeholder="e.g. luxury towels, white quilt, navy striped shower curtain",
            key="room_textiles_request",
        )

        if st.button("Generate room concept image"):
            if uploaded_room is None:
                st.error("Please upload a room photo first.")
            else:
                with st.spinner("Analyzing your room and generating a concept…"):
                    try:
                        room_products = filter_catalog_by_query(
                            catalog_df, textiles_request or "", max_results=8
                        )
                        img_bytes = generate_room_concept_image(
                            uploaded_room, textiles_request, room_products
                        )

                        st.markdown("##### AI-generated room concept")
                        st.image(img_bytes, use_column_width=True)

                        st.caption(
                            "Concept image generated using only Market & Place products "
                            "as inspiration for textiles. Layout may not be pixel-perfect, "
                            "but it aims to keep the same room structure."
                        )

                        st.markdown("##### Products used as inspiration")
                        render_product_list(room_products)

                    except BadRequestError as e:
                        st.error(f"Image generation failed: {e}")
                    except Exception as e:  # fallback so the app doesn't crash
                        st.error(f"Something went wrong while generating the image: {e}")

    # --- STORE SHELF / SHOWROOM CONCEPT ---
    else:
        st.markdown("#### Store shelf / showroom concept")

        if Path(SHELF_PHOTO_PATH).exists():
            st.markdown("**Base Market & Place shelf photo (for layout):**")
            st.image(SHELF_PHOTO_PATH, use_column_width=True)
            st.caption(
                "The AI will try to keep this shelf layout while filling it with Market & Place products."
            )

        shelf_query = st.text_input(
            "Describe the shelf you want to visualize:",
            placeholder="e.g. cabana stripe beach towels in aqua and navy, folded stacks and matching mats",
            key="shelf_request",
        )

        if st.button("Generate store shelf concept image"):
            with st.spinner("Generating store shelf concept…"):
                try:
                    shelf_products = filter_catalog_by_query(
                        catalog_df, shelf_query or "", max_results=8
                    )
                    img_bytes = generate_store_shelf_concept(shelf_query, shelf_products)

                    st.markdown("##### AI-generated store shelf concept")
                    st.image(img_bytes, use_column_width=True)

                    st.caption(
                        "Concept image generated using your Market & Place shelf photo "
                        "for layout and Market & Place products for the textiles."
                    )

                    st.markdown("##### Products used as inspiration")
                    render_product_list(shelf_products)

                except BadRequestError as e:
                    st.error(f"Image generation failed: {e}")
                except Exception as e:
                    st.error(f"Something went wrong while generating the image: {e}")










