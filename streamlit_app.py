import base64
import io
import os
from typing import List, Tuple

import pandas as pd
import streamlit as st
from openai import OpenAI

# ---------- CONFIG ----------

DATA_PATH = "market_and_place_products.xlsx"  # product catalog
LOGO_PATH = "logo.png"                        # logo image
SHELF_BASE_PATH = "store shelf.jpg"          # base shelf photo for concepts

# Use logo as page icon if available
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
            "Category": "category",
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


def product_label(row: pd.Series) -> str:
    parts = [row.get("name", "")]
    if pd.notna(row.get("color", "")) and row["color"]:
        parts.append(f"Color: {row['color']}")
    if pd.notna(row.get("category", "")) and row["category"]:
        parts.append(f"Category: {row['category']}")
    if pd.notna(row.get("price", "")) and row["price"]:
        parts.append(f"${row['price']}")
    return " | ".join(parts)


def filter_catalog_by_query(
    df: pd.DataFrame, query: str, max_results: int = 8
) -> pd.DataFrame:
    """
    Loose matching for the left-hand 'Ask the AI stylist' search.
    """
    if not query:
        return df.head(max_results)

    q = query.lower()
    mask = (
        df["name_lower"].str.contains(q, case=False, na=False)
        | df["color_lower"].str.contains(q, case=False, na=False)
        | df.get("category", pd.Series("", index=df.index))
        .astype(str)
        .str.lower()
        .str.contains(q, case=False, na=False)
    )
    results = df[mask]
    if results.empty:
        # fallback: split into words
        simple_mask = pd.Series(False, index=df.index)
        for word in q.split():
            simple_mask |= df["name_lower"].str.contains(word, case=False, na=False)
        results = df[simple_mask]
    return results.head(max_results)


def render_product_card(row: pd.Series):
    cols = st.columns([1, 2.5])
    with cols[0]:
        if row.get("image_url") and str(row["image_url"]).startswith("http"):
            st.image(row["image_url"], use_column_width=True)
    with cols[1]:
        st.markdown(f"**{row['name']}**")
        if row.get("category") and row["category"] != "nan":
            st.write(f"• Category: {row['category']}")
        st.write(f"• Color: {row['color']}")
        st.write(f"• Price: {row['price']}")
        if row.get("amazon") and row["amazon"] != "nan":
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


def generate_room_concept_image(
    uploaded_room_file, product_row: pd.Series, styling_notes: str
) -> bytes:
    """
    Image-to-image: keep the uploaded room layout and add/replace textiles
    with the selected Market & Place product.
    """
    base_image_bytes = uploaded_room_file.getvalue()

    product_desc = product_label(product_row)

    prompt = f"""
You are an interior stylist doing **image editing** on a photo of a real room.

Requirements:
- Keep the exact room type, camera angle, furniture, windows, walls, and lighting from the original photo.
- Only update textiles (bedding, towels, shower curtains, rugs, window treatments, etc.).
- Use **only this Market & Place product** as the hero textile:

{product_desc}

- Match its color/pattern realistically and apply it where it makes sense in the room.
- Do not change the room to a different type (bedroom stays a bedroom, bathroom stays a bathroom, etc.).
- Do not add text, logos, or extra products from other brands.

Extra styling notes from the customer (if any):
\"\"\"{styling_notes}\"\"\".
"""

    img_resp = client.images.generate(
        model=OPENAI_MODEL_IMAGE,
        prompt=prompt,
        size="1024x1024",
        image=base_image_bytes,  # image-to-image
    )

    b64 = img_resp.data[0].b64_json
    return base64.b64decode(b64)


def generate_store_shelf_concept(product_row: pd.Series, styling_notes: str) -> bytes:
    """
    Image-to-image: keep the real Market & Place store shelf layout and
    fill it with the selected product.
    """
    if not os.path.exists(SHELF_BASE_PATH):
        raise FileNotFoundError(f"Base shelf photo not found at '{SHELF_BASE_PATH}'")

    with open(SHELF_BASE_PATH, "rb") as f:
        shelf_bytes = f.read()

    product_desc = product_label(product_row)

    prompt = f"""
You are creating a **store shelf / showroom** concept for Market & Place.

Requirements:
- Keep the exact shelving layout, perspective, and store environment from the base photo.
- Do NOT turn this into a bathroom or bedroom; it must stay a retail shelf / aisle.
- Fill the shelves with neatly folded/hanging textiles representing **only this product**:

{product_desc}

- Use its real colors and general pattern, but you may tile or repeat it to fill the shelf.
- No extra brands, no additional product types beyond what fits this product.
- No text or logos.

Extra shelf styling notes from the customer (if any):
\"\"\"{styling_notes}\"\"\".
"""

    img_resp = client.images.generate(
        model=OPENAI_MODEL_IMAGE,
        prompt=prompt,
        size="1024x1024",
        image=shelf_bytes,  # image-to-image using fixed shelf photo
    )

    b64 = img_resp.data[0].b64_json
    return base64.b64decode(b64)


# ---------- LAYOUT: HEADER ----------

header_cols = st.columns([1, 2, 1])
with header_cols[1]:
    if os.path.exists(LOGO_PATH):
        st.image(
            LOGO_PATH,
            use_column_width=False,
            width=260,  # roughly half the original size
        )

    st.markdown(
        "<h2 style='text-align:center; margin-bottom:0.15rem;'>"
        "Market & Place AI Stylist"
        "</h2>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='text-align:center; margin-top:0;'>"
        "Chat with an AI stylist, search the Market & Place catalog, and generate "
        "concept visualizations using your own product file."
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='text-align:center; margin-top:0.1rem;'>"
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
            placeholder=(
                "e.g. luxury bath towels in grey, striped beach towels, "
                "queen quilt ideas"
            ),
        )
        submitted = st.form_submit_button("Send")

    if submitted and user_query.strip():
        st.markdown(f"### 🧵 {user_query.strip()}")
        matches = filter_catalog_by_query(catalog_df, user_query, max_results=6)
        render_product_list(matches)

    st.markdown("---")
    st.subheader("Quick catalog peek")

    if catalog_df.empty:
        st.info("Catalog is empty.")
    else:
        # 5 random products each time the app is opened/refreshed
        sample_df = catalog_df.sample(
            n=min(5, len(catalog_df)), replace=False, random_state=None
        )
        render_product_list(sample_df)

# ----- RIGHT: AI CONCEPT VISUALIZER (ROOM & SHELF IMAGE-TO-IMAGE) -----

with right_col:
    st.subheader("AI concept visualizer")

    st.write(
        "Generate a **styled concept** using your own room photo or the Market & "
        "Place store shelf photo, combined with a product from the catalog."
    )

    mode = st.radio(
        "What do you want to visualize?",
        ["Room concept image", "Store shelf / showroom concept"],
        horizontal=True,
    )

    # Pre-build product index list for selectboxes
    product_indices = catalog_df.index.tolist()

    # --- ROOM CONCEPT IMAGE ---
    if mode == "Room concept image":
        st.markdown("#### Room concept image")

        if not product_indices:
            st.info("No products found in the catalog for styling.")
        else:
            selected_idx = st.selectbox(
                "Choose a Market & Place product to feature in this room:",
                product_indices,
                format_func=lambda i: product_label(catalog_df.loc[i]),
                key="room_product_select",
            )
            selected_product = catalog_df.loc[selected_idx]

            styling_notes = st.text_input(
                "Optional: any styling notes?",
                placeholder="e.g. modern spa look, neutrals, add striped accents",
                key="room_styling_notes",
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
                                uploaded_room, selected_product, styling_notes or ""
                            )
                            st.image(img_bytes, use_column_width=True)
                            st.caption(
                                "The AI kept your room layout and overlaid the selected "
                                "Market & Place product as the main textile."
                            )
                        except Exception as e:
                            st.error(f"Image generation failed: {e}")

    # --- STORE SHELF CONCEPT ---
    else:
        st.markdown("#### Store shelf / showroom concept")

        if os.path.exists(SHELF_BASE_PATH):
            st.image(
                SHELF_BASE_PATH,
                caption="Market & Place store shelf photo (base image for concepts).",
                use_column_width=True,
            )
        else:
            st.warning(
                f"Base shelf photo not found at '{SHELF_BASE_PATH}'. "
                "Upload it to the repo to enable shelf concepts."
            )

        if not product_indices:
            st.info("No products found in the catalog for shelf styling.")
        else:
            selected_idx = st.selectbox(
                "Choose a Market & Place product to feature on this shelf:",
                product_indices,
                format_func=lambda i: product_label(catalog_df.loc[i]),
                key="shelf_product_select",
            )
            selected_product = catalog_df.loc[selected_idx]

            shelf_notes = st.text_input(
                "Optional: any shelf styling notes?",
                placeholder="e.g. beachy cabana story, color-blocked stacks",
                key="shelf_styling_notes",
            )

            if st.button("Generate store shelf concept image"):
                if not os.path.exists(SHELF_BASE_PATH):
                    st.error(
                        f"Base shelf photo not found at '{SHELF_BASE_PATH}'. "
                        "Please add it to the repo."
                    )
                else:
                    with st.spinner("Generating AI store shelf concept…"):
                        try:
                            img_bytes = generate_store_shelf_concept(
                                selected_product, shelf_notes or ""
                            )
                            st.image(img_bytes, use_column_width=True)
                            st.caption(
                                "The AI kept the Market & Place store shelf layout and "
                                "filled it with the selected product."
                            )
                        except Exception as e:
                            st.error(f"Image generation failed: {e}")









