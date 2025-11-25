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

OPENAI_MODEL_VISION = "gpt-4.1-mini"
OPENAI_MODEL_IMAGE = "gpt-image-1"

client = OpenAI()

DATA_PATH = "market_and_place_products.xlsx"  # same folder as this file
LOGO_PATH = "logo.png"                        # same folder as this file
SHELF_BASE_PATH = "store shelf.jpg"          # your reference shelf photo


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
        "Category": "category",  # optional, if you added it
    })

    for col in ["name", "color", "amazon", "image_url", "category"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    df["name_lower"] = df["name"].str.lower()
    df["color_lower"] = df["color"].str.lower()

    return df


catalog_df = load_catalog(DATA_PATH)


# ---------- SIMPLE SEARCH HELPERS ----------

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


# ---------- RENDER HELPERS ----------

def render_product_card(row: pd.Series):
    cols = st.columns([1, 2.5])
    with cols[0]:
        if row.get("image_url") and str(row["image_url"]).strip().lower().startswith(("http://", "https://")):
            st.image(row["image_url"], use_column_width=True)
        elif row.get("image_url") and os.path.exists(row["image_url"]):
            st.image(row["image_url"], use_column_width=True)
    with cols[1]:
        st.markdown(f"**{row['name']}**")
        st.write(f"• Color: {row['color']}")
        st.write(f"• Price: {row['price']}")
        if row.get("amazon") and isinstance(row["amazon"], str) and row["amazon"].startswith("http"):
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

def bytes_to_data_uri(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def analyze_room_with_vision(image_bytes: bytes) -> Dict[str, str]:
    """
    Ask the vision model to tell us:
    - room type (bedroom, bathroom, etc.)
    - style sentence
    - layout sentence
    """
    data_uri = bytes_to_data_uri(image_bytes)

    prompt = """
You are an interior design expert. Look at this photo and respond in exactly this format:

ROOMTYPE: <one- or two-word room type like "bedroom", "bathroom", "living room">
STYLE: <one short sentence about colors, mood, and materials>
LAYOUT: <one short sentence describing the layout and main furniture positions>

No extra commentary.
""".strip()

    resp = client.responses.create(
        model=OPENAI_MODEL_VISION,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": {"url": data_uri}},
                ],
            }
        ],
    )

    try:
        text = resp.output[0].content[0].text
    except Exception:
        try:
            text = resp.output_text
        except Exception:
            text = str(resp)

    room_type = "room"
    style = "neutral style"
    layout = "standard layout"

    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("ROOMTYPE:"):
            room_type = line.split(":", 1)[1].strip()
        elif line.upper().startswith("STYLE:"):
            style = line.split(":", 1)[1].strip()
        elif line.upper().startswith("LAYOUT:"):
            layout = line.split(":", 1)[1].strip()

    return {"room_type": room_type, "style": style, "layout": layout}


# ---------- IMAGE GENERATION ----------

def generate_room_concept_image(uploaded_room_file, product_row: pd.Series, styling_notes: str) -> bytes:
    """
    1) Describe the uploaded room with the vision model.
    2) Ask the image model to render a new concept of the SAME room type and a similar layout,
       featuring the selected Market & Place product as the main textile.
    """
    image_bytes = uploaded_room_file.getvalue()
    room_info = analyze_room_with_vision(image_bytes)

    product_desc = f"{product_row['name']} (color: {product_row['color']})"

    notes = styling_notes.strip() if styling_notes else "Keep the design cohesive with the existing style."

    prompt = f"""
You are creating a realistic interior concept image.

ROOM TYPE AND LAYOUT TO PRESERVE
- Room type: {room_info['room_type']}. Do NOT change to another type of room (no bathrooms if it is a bedroom, etc.).
- Layout: {room_info['layout']} (keep the main furniture positions and camera angle similar).

EXISTING STYLE
- {room_info['style']}

FEATURED MARKET & PLACE PRODUCT
- {product_desc}

GOAL
- Replace or layer the existing textiles in the room (bedding, towels, curtains, rugs, etc.) with this Market & Place product where it makes sense.
- Keep walls, windows, doors, and main furniture consistent with the described layout.
- Avoid brand logos or text; patterns should look like the Market & Place product but generic.

USER NOTES
- {notes}
""".strip()

    img_resp = client.images.generate(
        model=OPENAI_MODEL_IMAGE,
        prompt=prompt,
        size="1024x1024",
        n=1,
    )

    b64 = img_resp.data[0].b64_json
    return base64.b64decode(b64)


def generate_store_shelf_concept(product_row: pd.Series, styling_notes: str) -> bytes:
    """
    Generate a retail shelf / showroom bay image using only Market & Place products as inspiration.
    This is a fresh concept image (not an exact edit of the reference shelf photo).
    """
    product_desc = f"{product_row['name']} (color: {product_row['color']})"
    notes = styling_notes.strip() if styling_notes else "Neat, well-organized merchandising."

    prompt = f"""
Create a realistic concept image of a Market & Place retail store shelf / showroom bay.

REQUIREMENTS
- It must clearly be a retail store interior, not a home bathroom or bedroom.
- Show long white gondola-style shelves similar to a real store aisle.
- No people, no brand logos.

FEATURED MARKET & PLACE PRODUCT
- {product_desc}

GOAL
- Fill the shelves mainly with stacks and/or hanging displays of this product and closely related colorways.
- Keep the display tidy and visually appealing, like a high-end retailer.

USER NOTES
- {notes}
""".strip()

    img_resp = client.images.generate(
        model=OPENAI_MODEL_IMAGE,
        prompt=prompt,
        size="1024x1024",
        n=1,
    )

    b64 = img_resp.data[0].b64_json
    return base64.b64decode(b64)


# ---------- HEADER (CONDENSED) ----------

header_cols = st.columns([1, 2, 1])
with header_cols[1]:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=260)
    st.markdown(
        "<h2 style='text-align:center; margin-bottom:0.4rem;'>Market & Place AI Stylist</h2>",
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
        "<p style='text-align:center; margin-top:0;'>"
        "<a href='https://marketandplace.co/' style='text-decoration:none;'>"
        "← Return to Market & Place website</a></p>",
        unsafe_allow_html=True,
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
    st.caption("A few random Market & Place products from your catalog:")

    if len(catalog_df) > 0:
        sample_df = catalog_df.sample(
            n=min(5, len(catalog_df)),
            random_state=None,  # new random sample each run
        )
        render_product_list(sample_df)
    else:
        st.info("Catalog appears to be empty.")


# ----- RIGHT: AI CONCEPT VISUALIZER (ROOM IMAGE + STORE SHELF IMAGE) -----

with right_col:
    st.subheader("AI concept visualizer")

    st.write(
        "Generate a **styled concept** using your own photos and Market & Place products. "
        "Choose a product from the catalog, upload a room photo for styling, or "
        "generate a store shelf concept."
    )

    mode = st.radio(
        "What do you want to visualize?",
        ["Room concept image", "Store shelf / showroom concept"],
        horizontal=True,
    )

    # Common: select a product from the catalog
    if len(catalog_df) == 0:
        st.warning("No products found in the catalog file.")
    else:
        if mode == "Room concept image":
            st.markdown("#### Room concept image")

            selected_idx = st.selectbox(
                "Choose a Market & Place product to feature in this room:",
                options=catalog_df.index,
                format_func=lambda i: f"{catalog_df.loc[i,'name']} | {catalog_df.loc[i,'color']} | ${catalog_df.loc[i,'price']}",
                key="room_product_select",
            )
            selected_product = catalog_df.loc[selected_idx]

            styling_notes = st.text_input(
                "Optional: any styling notes?",
                placeholder="e.g. modern spa look, neutrals, add striped accents",
                key="room_style_notes",
            )

            uploaded_room = st.file_uploader(
                "Reference room photo (used as style inspiration):",
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
                            img_bytes = generate_room_concept_image(
                                uploaded_room,
                                selected_product,
                                styling_notes,
                            )
                            st.image(img_bytes, use_column_width=True)
                            st.caption(
                                "AI-generated concept using your room photo as inspiration "
                                "and the selected Market & Place product."
                            )
                        except Exception as e:
                            st.error(f"Image generation failed: {e}")

        else:
            st.markdown("#### Store shelf / showroom concept")

            if os.path.exists(SHELF_BASE_PATH):
                st.image(
                    SHELF_BASE_PATH,
                    caption="Market & Place store shelf reference photo (for inspiration).",
                    use_column_width=True,
                )

            selected_idx_shelf = st.selectbox(
                "Choose a Market & Place product to feature on this shelf:",
                options=catalog_df.index,
                format_func=lambda i: f"{catalog_df.loc[i,'name']} | {catalog_df.loc[i,'color']} | ${catalog_df.loc[i,'price']}",
                key="shelf_product_select",
            )
            shelf_product = catalog_df.loc[selected_idx_shelf]

            shelf_notes = st.text_input(
                "Optional: how would you like the shelf styled?",
                placeholder="e.g. stacks of cabana stripe beach towels in aqua & navy, color-blocked by shelf",
                key="shelf_notes",
            )

            if st.button("Generate store shelf concept image"):
                with st.spinner("Generating AI store shelf concept…"):
                    try:
                        img_bytes = generate_store_shelf_concept(shelf_product, shelf_notes)
                        st.image(img_bytes, use_column_width=True)
                        st.caption(
                            "AI-generated store shelf concept using the selected Market & Place product."
                        )
                    except Exception as e:
                        st.error(f"Image generation failed: {e}")











