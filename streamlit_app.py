import base64
import os
from io import BytesIO
from typing import List

import pandas as pd
import streamlit as st
from openai import OpenAI


# ---------- CONFIG ----------

st.set_page_config(
    page_title="Market & Place AI Stylist",
    layout="wide",
)

# ---------- DATA LOADING ----------

@st.cache_data(show_spinner=False)
def load_catalog(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    # Normalize expected columns
    rename_map = {
        "Product name": "name",
        "Product Name": "name",
        "Color": "color",
        "Price": "price",
        "raw_amazon": "amazon_url",
        "Image URL:": "image_url",
        "Image URL": "image_url",
    }
    df = df.rename(columns=rename_map)

    # Ensure columns exist
    for col in ["name", "color", "price", "amazon_url", "image_url"]:
        if col not in df.columns:
            df[col] = ""

    # Normalize text columns
    df["name"] = df["name"].astype(str)
    df["color"] = df["color"].astype(str)
    df["category_hint"] = (
        df["name"].str.lower()
        + " "
        + df["color"].str.lower()
    )

    return df


CATALOG_PATH = "market_and_place_products.xlsx"
catalog_df = load_catalog(CATALOG_PATH)

# ---------- OPENAI CLIENT FOR STORE-SHELF IMAGES ----------

def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.warning("OPENAI_API_KEY is not set; store shelf image generation will be disabled.")
        return None
    return OpenAI(api_key=api_key)


client = get_openai_client()


# ---------- PRODUCT FILTERING LOGIC (NO HALLUCINATIONS) ----------

def word_list(text: str) -> List[str]:
    return [w.strip().lower() for w in text.split() if w.strip()]


def filter_catalog_by_query(df: pd.DataFrame, query: str, max_results: int = 12) -> pd.DataFrame:
    """
    Strict, deterministic filter that ONLY returns products from the Excel file.
    No AI, no hallucinations.
    """
    if not query:
        return df.head(0)

    q = query.lower()

    # Basic category routing
    # ----------------------------------
    base = df.copy()

    # Towel vs sheet rough routing
    is_towel = "towel" in q or "towels" in q
    is_sheet = any(w in q for w in ["sheet", "sheets", "bedding", "duvet", "comforter"])

    if is_towel and not is_sheet:
        base = base[base["category_hint"].str.contains("towel")]
        # Luxury vs beach towels
        if "beach" in q:
            base = base[base["category_hint"].str.contains("beach")]
        elif "luxury" in q or "spa" in q:
            # prefer things marked luxury/turkish, avoid beach
            luxury_mask = base["category_hint"].str.contains("luxury|turkish", regex=True)
            beach_mask = base["category_hint"].str.contains("beach")
            base = base[luxury_mask & ~beach_mask]
    elif is_sheet:
        base = base[
            base["category_hint"].str.contains(
                "sheet|bedding|duvet|comforter|quilt", regex=True
            )
        ]

    # Token matching on remaining subset
    tokens = word_list(q)
    if not tokens:
        return base.head(max_results)

    mask = pd.Series(True, index=base.index)
    for t in tokens:
        token_mask = (
            base["name"].str.lower().str.contains(t)
            | base["color"].str.lower().str.contains(t)
            | base["category_hint"].str.contains(t)
        )
        mask &= token_mask

    results = base[mask]

    # If we filtered too aggressively, fall back to the base subset
    if results.empty:
        results = base

    return results.head(max_results)


# ---------- DISPLAY HELPERS ----------

def render_product_card(row: pd.Series):
    col_img, col_text = st.columns([1, 3])
    with col_img:
        if isinstance(row.get("image_url", ""), str) and row["image_url"].strip():
            st.image(row["image_url"], use_container_width=True)
    with col_text:
        st.markdown(f"**{row['name']}**")
        if row.get("color", ""):
            st.markdown(f"- Color: `{row['color']}`")
        if row.get("price", ""):
            st.markdown(f"- Price: **{row['price']}**")
        if row.get("amazon_url", ""):
            st.markdown(f"- [View on Amazon]({row['amazon_url']})")


def render_product_list(title: str, df: pd.DataFrame):
    st.markdown(f"### {title}")
    if df.empty:
        st.write("No matching Market & Place products were found for that request.")
        return
    for _, row in df.iterrows():
        st.markdown("---")
        render_product_card(row)


# ---------- STORE SHELF IMAGE GENERATION ----------

def generate_store_shelf_image(prompt_text: str):
    if client is None:
        st.error("Image generation is disabled because OPENAI_API_KEY is not set.")
        return

    full_prompt = (
        "Create a high-quality 3D render of a retail store shelf or showroom wall "
        "filled ONLY with Market & Place style textiles. Emphasize folded towels, "
        "bath textiles, and bedding stacks that match this description: "
        f"\"{prompt_text}\". Neutral, well-lit store environment, straight-on camera view."
    )
    try:
        with st.spinner("Generating store shelf concept image..."):
            img_response = client.images.generate(
                model="gpt-image-1",
                prompt=full_prompt,
                size="1024x1024",
                n=1,
            )
        b64 = img_response.data[0].b64_json
        img_bytes = base64.b64decode(b64)
        st.image(img_bytes, caption="AI-generated store shelf concept", use_container_width=True)
    except Exception as e:
        st.error(f"Image generation failed: {e}")


# ---------- TOP SECTION: LOGO + TAGLINE ----------

def render_header():
    # Centered logo
    logo_path = "logo.png"
    if os.path.exists(logo_path):
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(logo_path, caption="Market & Place logo", use_container_width=True)

    st.markdown(
        "<h1 style='text-align:center;'>Market & Place AI Stylist</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;'>Chat with an AI stylist, search the Market & Place catalog, "
        "and generate concept visualizations using your own product file.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;'><a href='https://marketandplace.co/' target='_blank'>"
        "← Return to Market & Place website</a></p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")


# ---------- LEFT COLUMN: ASK THE STYLIST + QUICK CATALOG PEEK ----------

def render_left_column():
    st.subheader("Ask the AI stylist")

    # Single-shot query; each new query replaces the previous answer
    with st.form("stylist_form"):
        user_query = st.text_input(
            "Describe what you're looking for:",
            placeholder="e.g. luxury bath towels in grey, cabana stripe beach towels, flannel queen sheets…",
        )
        submitted = st.form_submit_button("Send")
    if submitted and user_query.strip():
        results = filter_catalog_by_query(catalog_df, user_query, max_results=10)
        st.markdown(f"#### 🧵 {user_query.strip()}")
        render_product_list("Here are some Market & Place products that could work:", results)

    st.markdown("---")
    st.subheader("Quick catalog peek")

    peek_query = st.text_input(
        "Filter products by keyword:",
        placeholder="e.g. cabana stripe, flannel sheet, navy",
        key="peek_query",
    )
    if peek_query.strip():
        subset = filter_catalog_by_query(catalog_df, peek_query, max_results=25)
        for _, row in subset.iterrows():
            st.markdown("---")
            render_product_card(row)
    else:
        st.write("Type a keyword above to quickly peek at the catalog.")


# ---------- RIGHT COLUMN: AI CONCEPT VISUALIZER ----------

def render_right_column():
    st.subheader("AI concept visualizer")

    st.markdown(
        "Generate ideas using your catalog:\n\n"
        "- **In my room (suggestions only):** upload a photo and describe what textiles you'd like. "
        "The app will suggest Market & Place products that fit the space. No new image is created.\n"
        "- **Store shelf / showroom:** generate an AI image of a retail shelf wall using Market & Place style textiles."
    )

    mode = st.radio(
        "What do you want to visualize?",
        ["In my room (product suggestions)", "Store shelf / showroom"],
        horizontal=True,
    )

    if mode.startswith("In my room"):
        st.markdown("#### In my room – product suggestions")

        room_type = st.selectbox(
            "Room type (for context):",
            ["bathroom", "bedroom", "living room", "kitchen", "other"],
            index=0,
        )

        room_photo = st.file_uploader(
            "Upload a reference photo of your room (optional):",
            type=["jpg", "jpeg", "png"],
        )

        desire = st.text_input(
            "What textiles would you like to see in this room?",
            placeholder="e.g. white luxury bath towels and a matching bath mat",
        )

        if st.button("Get product suggestions", key="room_suggestions_btn"):
            if not desire.strip():
                st.warning("Please describe what you'd like to visualize.")
            else:
                # Currently we just use the description + room type to filter the catalog.
                # The photo is stored / uploaded but not used for AI vision, to keep it simple and stable.
                query = f"{room_type} {desire}"
                results = filter_catalog_by_query(catalog_df, query, max_results=10)
                st.markdown("#### Suggested Market & Place products for your room")
                if room_photo is not None:
                    st.image(room_photo, caption="Uploaded room (reference)", use_container_width=True)
                render_product_list("Based on your description, consider:", results)

    else:
        st.markdown("#### Store shelf / showroom concept image")

        shelf_prompt = st.text_input(
            "Describe the shelf / showroom concept you want:",
            placeholder="e.g. rainbow cabana stripe beach towels, stacked by color",
            key="shelf_prompt",
        )

        if st.button("Generate store shelf concept image", key="store_shelf_btn"):
            if not shelf_prompt.strip():
                st.warning("Please describe the shelf / showroom concept you want.")
            else:
                generate_store_shelf_image(shelf_prompt.strip())


# ---------- MAIN APP ----------

def main():
    render_header()

    col_left, col_right = st.columns([1.2, 1.1])
    with col_left:
        render_left_column()
    with col_right:
        render_right_column()


if __name__ == "__main__":
    main()










