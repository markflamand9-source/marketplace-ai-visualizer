# streamlit_app.py

import base64
import os
from typing import List, Dict

import pandas as pd
import streamlit as st
from openai import OpenAI

# -------------------------------------------------------------------
# CONFIG & GLOBALS
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Market & Place AI Stylist",
    layout="wide",
)

client = OpenAI()

CATALOG_PATH = "market_and_place_products.xlsx"
LOGO_PATH = "logo.png"
BRAND_NAME = "Market & Place"


# -------------------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_catalog(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Catalog file not found: {path}")
        return pd.DataFrame()
    df = pd.read_excel(path)

    # Normalize column names a bit
    df.columns = [str(c).strip() for c in df.columns]

    # Add a simple "Category" guess if not present, based on product name
    if "Category" not in df.columns:
        categories = []
        for name in df.get("Product name", ""):
            name_str = str(name).lower()
            if "towel" in name_str:
                categories.append("towel")
            elif "sheet" in name_str:
                categories.append("sheet")
            elif "quilt" in name_str or "comforter" in name_str or "bedspread" in name_str:
                categories.append("quilt")
            elif "shower curtain" in name_str:
                categories.append("shower curtain")
            else:
                categories.append("other")
        df["Category"] = categories

    return df


catalog_df = load_catalog(CATALOG_PATH)


# -------------------------------------------------------------------
# HELPER FUNCTIONS — SEARCH & MATCHING
# -------------------------------------------------------------------

def detect_requested_category(query: str) -> str | None:
    """
    Very simple heuristic to figure out what kind of product
    the user is probably asking for.
    """
    q = query.lower()
    if any(w in q for w in ["towel", "towels"]):
        # distinguish beach vs bath if we can
        if "beach" in q:
            return "beach towel"
        if "bath" in q:
            return "bath towel"
        return "towel"
    if any(w in q for w in ["sheet", "sheets"]):
        return "sheet"
    if any(w in q for w in ["quilt", "comforter", "duvet", "bedspread", "coverlet"]):
        return "quilt"
    if "shower curtain" in q or ("shower" in q and "curtain" in q):
        return "shower curtain"
    return None


def filter_catalog_for_query(df: pd.DataFrame, query: str, limit: int = 6) -> pd.DataFrame:
    """
    Search the Market & Place catalog based on free-text query.
    Only returns products that actually exist in the Excel file.
    """
    if df.empty or not query.strip():
        return df.head(0)

    q = query.lower()
    requested_cat = detect_requested_category(q)

    # Start with simple keyword match over a few fields
    text_cols = [c for c in ["Product name", "Color", "Category"] if c in df.columns]
    mask = pd.Series(False, index=df.index)

    for col in text_cols:
        mask |= df[col].astype(str).str.lower().str.contains(q, na=False)

    # If that finds nothing, loosen it by splitting into words
    if not mask.any():
        for word in q.split():
            for col in text_cols:
                mask |= df[col].astype(str).str.lower().str.contains(word, na=False)

    results = df[mask].copy()

    # Category-specific filtering
    if requested_cat:
        # Special handling for beach towels vs general towels
        if requested_cat == "beach towel":
            name_mask = results["Product name"].astype(str).str.lower().str.contains("beach")
            cat_mask = results["Category"].astype(str).str.contains("towel", case=False, na=False)
            results = results[name_mask | cat_mask]
        elif requested_cat == "bath towel":
            name_mask = results["Product name"].astype(str).str.lower().str.contains("bath")
            cat_mask = results["Category"].astype(str).str.contains("towel", case=False, na=False)
            results = results[name_mask | cat_mask]
        else:
            results = results[results["Category"].astype(str).str.contains(requested_cat.split()[0], case=False, na=False)]

    # If still nothing, just fall back to all matches
    if results.empty:
        results = df[mask].copy()

    return results.head(limit)


def get_products_for_visualizer(df: pd.DataFrame, room_type: str, notes: str) -> pd.DataFrame:
    """
    Use the same catalog filtering logic, but biased by room type.
    """
    base_query = f"{room_type} {notes}".strip()
    results = filter_catalog_for_query(df, base_query, limit=8)

    # Extra bias: in a bathroom, prefer towels & shower curtains; in a bedroom, sheets & quilts.
    if room_type.lower() == "bathroom":
        if not results.empty:
            mask = results["Category"].astype(str).str.contains("towel|shower curtain", case=False, na=False)
            prioritized = results[mask]
            if not prioritized.empty:
                return prioritized.head(8)
    if room_type.lower() == "bedroom":
        if not results.empty:
            mask = results["Category"].astype(str).str.contains("sheet|quilt", case=False, na=False)
            prioritized = results[mask]
            if not prioritized.empty:
                return prioritized.head(8)

    return results


# -------------------------------------------------------------------
# HELPER FUNCTIONS — UI & RENDERING
# -------------------------------------------------------------------

def render_logo_and_header():
    st.markdown(
        f"<div style='text-align:center; margin-bottom:1rem;'>"
        f"<img src='app://{LOGO_PATH}' alt='{BRAND_NAME} logo' style='max-height:140px; margin-bottom:0.5rem;' />"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<h1 style='text-align:center; margin-bottom:0.25rem;'>{BRAND_NAME} AI Stylist</h1>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='text-align:center;'>"
        "Chat with an AI stylist, search the Market &amp; Place catalog, "
        "and generate concept visualizations using your own product file."
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div style='text-align:center; margin-bottom: 1.5rem;'>"
        f"<a href='https://marketandplace.co/' style='text-decoration:none;'>"
        f"&larr; Return to {BRAND_NAME} website</a></div>",
        unsafe_allow_html=True,
    )


def render_product_card(row: pd.Series):
    """
    Nice-looking product card for suggestions & catalog peek.
    No weird green text; everything neutral & clean.
    """
    with st.container(border=True):
        cols = st.columns([1, 2])
        with cols[0]:
            img_col = None
            for key in ["Image URL:", "Image URL", "image_url"]:
                if key in row and pd.notna(row[key]):
                    img_col = key
                    break
            if img_col:
                try:
                    st.image(row[img_col], use_column_width=True)
                except Exception:
                    pass

        with cols[1]:
            name = row.get("Product name", "Unnamed product")
            st.markdown(f"**{name}**")

            if "Color" in row and pd.notna(row["Color"]):
                st.write(f"**Color:** {row['Color']}")

            if "Price" in row and pd.notna(row["Price"]):
                st.write(f"**Price:** {row['Price']}")

            # Amazon URL – must be used as-is, no changes.
            raw_url_col = None
            for key in ["raw_amazon", "Amazon URL", "amazon_url"]:
                if key in row and pd.notna(row[key]):
                    raw_url_col = key
                    break

            if raw_url_col:
                url = str(row[raw_url_col])
                st.markdown(f"[View on Amazon]({url})")


def generate_store_shelf_image(selected_products: List[Dict], extra_notes: str = "") -> str | None:
    """
    Generate a retail store shelf visualization using Market & Place products.
    Returns base64 image string, or None on failure.
    """

    if not selected_products:
        desc_text = "assorted Market & Place striped towels and bedding products"
    else:
        fragments = []
        for p in selected_products:
            name = p.get("Product name", "")
            color = p.get("Color", "")
            cat = p.get("Category", "")
            frag = f"{color} {cat} from Market & Place called '{name}'"
            fragments.append(frag)
        desc_text = "; ".join(fragments)

    prompt = f"""
You are generating a *visual merchandising mockup* for a textile brand called Market & Place.

Create a **realistic retail store shelf / aisle** scene, **not** a home bathroom, bedroom,
or living room. It must clearly look like a **store**:
- long metal gondola shelves or wall fixtures
- neutral retail floor and ceiling
- no bathtubs, toilets, sinks, beds, sofas, or domestic furniture.

On the shelves, show folded stacks and hanging displays of textiles that match:

{desc_text}

Rules:
- Use only textile products inspired by Market & Place (towels, sheets, quilts, shower curtains, etc.).
- Use stripe patterns, textures, and colors consistent with Market & Place products
  (cabana stripes, coastal colors, etc.).
- Do **not** invent non-textile products (no bottles, toys, food, electronics).
- The focus is the textile presentation on store shelves.
{extra_notes}
""".strip()

    try:
        img_resp = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024",
            n=1,
        )
        b64 = img_resp.data[0].b64_json
        return b64
    except Exception as e:
        st.error(f"Image generation failed: {e}")
        return None


def show_base64_image(b64: str, caption: str):
    try:
        img_bytes = base64.b64decode(b64)
        st.image(img_bytes, caption=caption, use_column_width=True)
    except Exception as e:
        st.error(f"Could not display image: {e}")


# -------------------------------------------------------------------
# MAIN APP LAYOUT
# -------------------------------------------------------------------

def main():
    render_logo_and_header()

    if catalog_df.empty:
        st.stop()

    left_col, right_col = st.columns([1, 1.1], gap="large")

    # ---------------------------------------------------------------
    # LEFT: ASK THE AI STYLIST + QUICK CATALOG PEEK
    # ---------------------------------------------------------------
    with left_col:
        st.subheader("Ask the AI stylist")

        with st.form("stylist_form", clear_on_submit=False):
            user_query = st.text_input(
                "Describe what you're looking for:",
                placeholder="e.g. luxury bath towels in grey, cabana beach towels, queen sheet set...",
            )
            submit = st.form_submit_button("Send")

        if submit and user_query.strip():
            st.markdown(f"### 🧵 {user_query}")
            results = filter_catalog_for_query(catalog_df, user_query, limit=8)

            if results.empty:
                st.info(
                    "I couldn't find any matching products in the Market & Place catalog for that request. "
                    "Try rephrasing (for example, include the product type like 'luxury bath towels in grey')."
                )
            else:
                st.write(
                    "Here are some Market & Place products that match your request and could work well in your space:"
                )
                for _, row in results.iterrows():
                    render_product_card(row)

        st.markdown("---")
        st.subheader("Quick catalog peek")

        peek_query = st.text_input(
            "Filter products by keyword:",
            placeholder="e.g. cabana stripe, flannel sheet, navy",
            key="peek_query",
        )

        if peek_query.strip():
            peek_results = filter_catalog_for_query(catalog_df, peek_query, limit=20)
        else:
            peek_results = catalog_df.head(0)

        if peek_results.empty and peek_query.strip():
            st.info("No catalog items match that keyword.")
        else:
            for _, row in peek_results.iterrows():
                render_product_card(row)

    # ---------------------------------------------------------------
    # RIGHT: AI CONCEPT VISUALIZER
    # ---------------------------------------------------------------
    with right_col:
        st.subheader("AI concept visualizer")

        st.write(
            "Generate a **styled concept** using Market & Place products.\n\n"
            "- **In my room photo**: upload a photo and describe what textiles you’d like. "
            "The AI will suggest products from the catalog (no image editing).\n"
            "- **Store shelf / showroom**: generate a retail shelf/aisle view with Market & Place products."
        )

        mode = st.radio(
            "What do you want to visualize?",
            options=["In my room photo", "Store shelf / showroom"],
            horizontal=False,
        )

        if mode == "In my room photo":
            room_type = st.selectbox(
                "Room type:",
                options=["bathroom", "bedroom", "living room", "other"],
                index=0,
            )

            uploaded_room = st.file_uploader(
                "Upload a reference photo of your room (optional):",
                type=["jpg", "jpeg", "png"],
                key="room_uploader",
            )

            notes = st.text_input(
                "What textiles would you like to add or change?",
                placeholder="e.g. striped bath towels and a matching bath mat",
                key="room_notes",
            )

            if st.button("Get product suggestions", key="room_suggestions_btn"):
                if not notes.strip():
                    st.warning("Please describe what you'd like to visualize (towels, sheets, colors, etc.).")
                else:
                    st.markdown("### Suggested Market & Place products for your room")
                    if uploaded_room is not None:
                        st.caption("Reference photo uploaded – the suggestions are still catalog-only, not image edits.")

                    vis_results = get_products_for_visualizer(catalog_df, room_type, notes)

                    if vis_results.empty:
                        st.info(
                            "I couldn't find matching products in the catalog for that request. "
                            "Try adding more detail, like 'navy striped bath towels' or 'white quilt for queen bed'."
                        )
                    else:
                        for _, row in vis_results.iterrows():
                            render_product_card(row)

        else:  # Store shelf / showroom
            st.markdown("#### Store shelf / showroom view")

            shelf_notes = st.text_input(
                "What should the shelf focus on?",
                value="cabana stripe beach towels in aqua and navy",
                key="shelf_notes",
            )

            if st.button("Generate store shelf concept image", key="shelf_generate_btn"):
                if not shelf_notes.strip():
                    st.warning("Please describe what products the shelf should focus on.")
                else:
                    with st.spinner("Finding matching Market & Place products..."):
                        shelf_products_df = filter_catalog_for_query(catalog_df, shelf_notes, limit=8)

                    if shelf_products_df.empty:
                        st.info(
                            "I couldn't find specific products for that description, "
                            "but I'll still generate a Market & Place–style shelf using generic textiles."
                        )
                        selected_products = []
                    else:
                        selected_products = [
                            shelf_products_df.iloc[i].to_dict() for i in range(len(shelf_products_df))
                        ]

                        st.markdown("**Products used as inspiration for this shelf:**")
                        for _, row in shelf_products_df.iterrows():
                            render_product_card(row)

                    with st.spinner("Generating AI store shelf visualization..."):
                        b64_img = generate_store_shelf_image(selected_products, extra_notes=shelf_notes)

                    if b64_img:
                        show_base64_image(
                            b64_img,
                            caption="AI-generated Market & Place store shelf concept",
                        )


if __name__ == "__main__":
    main()










