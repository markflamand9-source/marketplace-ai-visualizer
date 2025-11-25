import base64
import io
import os
from typing import List, Tuple

import pandas as pd
import streamlit as st
from openai import OpenAI

# ---------- CONFIG ----------

LOGO_PATH = "logo.png"  # in repo root
DATA_PATH = "market_and_place_products.xlsx"
SHELF_BASE_PATH = "store shelf.jpg"  # base shelf photo in repo

# Use logo as page icon if it exists
page_icon = LOGO_PATH if os.path.exists(LOGO_PATH) else "🧵"

st.set_page_config(
    page_title="Market & Place AI Stylist",
    page_icon=page_icon,
    layout="wide",
)

OPENAI_MODEL_VISION = "gpt-4.1-mini"
OPENAI_MODEL_IMAGE = "gpt-image-1"

client = OpenAI()

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
            "Category": "category",  # if present
        }
    )

    for col in ["name", "color", "amazon", "image_url", "category"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    df["name_lower"] = df["name"].str.lower()
    df["color_lower"] = df["color"].str.lower()

    return df


catalog_df = load_catalog(DATA_PATH)

# ---------- HELPER FUNCTIONS ----------


def filter_catalog_by_query(
    df: pd.DataFrame, query: str, max_results: int = 8
) -> pd.DataFrame:
    """Loose keyword matching on name + color."""

    if not query:
        return df.head(max_results)

    q = query.lower()
    words = [w for w in q.split() if w]

    mask = pd.Series(False, index=df.index)
    for w in words:
        mask |= df["name_lower"].str.contains(w, case=False, na=False)
        mask |= df["color_lower"].str.contains(w, case=False, na=False)

    results = df[mask]
    if results.empty:
        results = df[df["name_lower"].str.contains(q, case=False, na=False)]

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
        if row.get("amazon") and str(row["amazon"]).startswith("http"):
            st.markdown(f"[View on Amazon]({row['amazon']})")


def render_product_list(df: pd.DataFrame):
    if df.empty:
        st.info(
            "We couldn’t find matching products in the catalog for that request. "
            "Try adding some detail, like *'navy striped bath towels'* or "
            "*'white quilt for queen bed'*."
        )
        return

    for _, row in df.iterrows():
        render_product_card(row)
        st.markdown("---")


def uploaded_file_to_data_url(uploaded_file) -> str:
    """
    Convert a Streamlit UploadedFile into a data URL that the
    OpenAI Responses API accepts for `image_url.url`.
    """
    file_bytes = uploaded_file.getvalue()
    # uploaded_file.type is e.g. 'image/jpeg'
    mime = uploaded_file.type or "image/jpeg"
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def describe_photo_with_vision(uploaded_file) -> str:
    """
    Use the vision model to get a compact description of the room,
    so we can feed that into the image generator prompt.
    """
    data_url = uploaded_file_to_data_url(uploaded_file)

    resp = client.responses.create(
        model=OPENAI_MODEL_VISION,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": {"url": data_url},
                    },
                    {
                        "type": "text",
                        "text": (
                            "You are an interior designer. Briefly describe this room in one sentence: "
                            "room type (bathroom, bedroom, etc.), overall style, colors, and key furniture. "
                            "Do NOT invent any new objects."
                        ),
                    },
                ],
            }
        ],
    )

    # New Responses API: text is under output[0].content[0].text
    try:
        description = resp.output[0].content[0].text
    except Exception:
        description = str(resp)

    return description


def generate_room_concept_image(uploaded_file, product_row: pd.Series, style_notes: str):
    """
    Generate a NEW concept image (not a strict edit) of the room,
    inspired by the uploaded photo and featuring the chosen catalog product.
    """

    room_desc = describe_photo_with_vision(uploaded_file)

    product_name = product_row["name"]
    product_color = product_row.get("color", "")

    prompt = f"""
You are an interior design CGI artist.

Base room description (from a real photo):
{room_desc}

Generate a photorealistic concept image of THE SAME ROOM TYPE as described above.
Do NOT change the room type (if it is a bedroom, keep it a bedroom; if it is a bathroom, keep it a bathroom).

Feature this exact Market & Place product prominently in the scene:
- Product: {product_name}
- Color: {product_color}

Integrate the product naturally into the textiles (bedding, towels, curtains, etc.) that make sense for that room type.

Additional styling notes from the user:
{style_notes or "Match the existing style of the room."}

Keep the basic layout and architecture similar to the described room, but you may clean up clutter and make the styling cohesive.
No people, no visible brand logos.
"""

    img_resp = client.images.generate(
        model=OPENAI_MODEL_IMAGE,
        prompt=prompt,
        size="1024x1024",
        n=1,
    )

    b64 = img_resp.data[0].b64_json
    return base64.b64decode(b64)


def generate_store_shelf_concept(style_prompt: str):
    """
    Text-only generation of a shelf concept image, inspired by the uploaded shelf photo.
    (We can't actually edit the base photo, so we describe it instead.)
    """

    base_shelf_desc = """
A clean, modern retail aisle with long white metal gondola shelves,
neutral flooring, and bright even lighting. The shelves are mostly empty.
"""

    prompt = f"""
You are creating a concept image for a Market & Place retail store shelf.

Base shelf description:
{base_shelf_desc}

Fill the shelves with folded stacks and hanging displays of Market & Place textiles
(towels, sheets, quilts, etc.) using cohesive, realistic color palettes.

User styling request:
{style_prompt or "Create an attractive, well-organized display of Market & Place textiles."}

Do NOT show beds, bathtubs, or other home furniture — it must clearly be a store aisle.
No people, no visible brand logos.
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
    # Smaller logo & tighter header
    if os.path.exists(LOGO_PATH):
        st.image(
            LOGO_PATH,
            use_column_width=False,
            width=260,
            output_format="PNG",
        )

    st.markdown(
        "<h2 style='text-align:center; margin-top:0.4rem; margin-bottom:0.2rem;'>"
        "Market & Place AI Stylist"
        "</h2>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='text-align:center; font-size:0.95rem; margin-bottom:0.4rem;'>"
        "Chat with an AI stylist, search the Market & Place catalog, and generate concept visualizations using your own product file."
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='text-align:center; font-size:0.9rem; margin-top:0;'>"
        "<a href='https://marketandplace.co/' style='text-decoration:none;'>"
        "← Return to Market & Place website"
        "</a>"
        "</p>",
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

    st.caption("A few products from the Market & Place catalog (refresh the app to see different ones).")

    if len(catalog_df) > 0:
        peek_df = catalog_df.sample(n=min(5, len(catalog_df)), random_state=None)
        render_product_list(peek_df)
    else:
        st.write("No products found in the catalog file.")


# ----- RIGHT: AI CONCEPT VISUALIZER (ROOM IMAGE + STORE SHELF) -----


with right_col:
    st.subheader("AI concept visualizer")

    st.write(
        "Generate a **styled concept** using your own photos and Market & Place products. "
        "You can either upload a room for AI styling, or generate a store shelf concept."
    )

    mode = st.radio(
        "What do you want to visualize?",
        ["Room concept image", "Store shelf / showroom concept"],
        horizontal=True,
    )

    # --- ROOM CONCEPT IMAGE ---
    if mode == "Room concept image":
        st.markdown("#### Room concept image")

        # Product dropdown
        st.write("Choose a Market & Place product to feature in this room:")

        product_options = catalog_df["name"].tolist()
        if not product_options:
            st.error("No products found in the catalog. Please check the Excel file.")
        else:
            default_index = 0
            selected_name = st.selectbox(
                "Product from catalog:",
                product_options,
                index=default_index,
            )

            selected_row = catalog_df[catalog_df["name"] == selected_name].iloc[0]

            style_notes = st.text_input(
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
                                uploaded_room, selected_row, style_notes
                            )
                            st.image(img_bytes, use_column_width=True)
                            st.caption(
                                "Concept image generated using your room photo as inspiration "
                                "and the selected Market & Place product."
                            )
                        except Exception as e:
                            st.error(f"Image generation failed: {e}")

    # --- STORE SHELF CONCEPT ---
    else:
        st.markdown("#### Store shelf / showroom concept")

        if os.path.exists(SHELF_BASE_PATH):
            st.image(
                SHELF_BASE_PATH,
                caption="Market & Place store shelf photo (used as style reference).",
                use_column_width=True,
            )
        else:
            st.info("Base shelf photo not found in the repo (expected 'store shelf.jpg').")

        shelf_request = st.text_input(
            "Describe how you'd like the shelf styled:",
            placeholder="e.g. cabana stripe beach towels in aqua and navy, folded stacks and matching mats",
            key="shelf_request",
        )

        if st.button("Generate store shelf concept image"):
            with st.spinner("Generating AI store shelf concept…"):
                try:
                    img_bytes = generate_store_shelf_concept(shelf_request or "")
                    st.image(img_bytes, use_column_width=True)
                    st.caption(
                        "Concept image generated for a Market & Place store shelf using your styling request."
                    )
                except Exception as e:
                    st.error(f"Image generation failed: {e}")










