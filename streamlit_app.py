import base64
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import streamlit as st
from openai import OpenAI


# =========================
# CONFIG & INITIALISATION
# =========================

st.set_page_config(
    page_title="Market & Place AI Stylist",
    layout="wide",
)

# Paths (logo & catalog live next to this script)
APP_DIR = Path(__file__).parent
LOGO_PATH = APP_DIR / "logo.png"
CATALOG_PATH = APP_DIR / "market_and_place_products.xlsx"

# OpenAI client – expects OPENAI_API_KEY in the Streamlit Cloud secrets
client = OpenAI()


# =========================
# DATA LOADING & HELPERS
# =========================

@st.cache_data(show_spinner="Loading Market & Place catalog…")
def load_catalog() -> pd.DataFrame:
    """
    Load the Market & Place Excel catalog and build a simple
    search index over all text columns.
    """
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Catalog file not found at {CATALOG_PATH}. "
            "Make sure 'market_and_place_products.xlsx' is in the app folder."
        )

    df = pd.read_excel(CATALOG_PATH)

    # Normalise column names a bit and keep everything as string for safety
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    for col in df.columns:
        df[col] = df[col].astype(str).fillna("")

    # Build a big text field for keyword matching
    text_cols = [c for c in df.columns]
    df["__search_text"] = df[text_cols].agg(" ".join, axis=1).str.lower()

    # Very lightweight product type classifier based on product name
    def classify_type(name: str) -> str:
        n = name.lower()
        if "beach towel" in n:
            return "beach_towel"
        if "towel" in n:
            return "towel"
        if "sheet" in n or "bedding" in n or "duvet" in n:
            return "sheet"
        if "quilt" in n or "comforter" in n:
            return "quilt"
        if "bath rug" in n or "rug" in n or "mat" in n:
            return "rug"
        return "other"

    name_col = "Product name" if "Product name" in df.columns else df.columns[0]
    df["__type"] = df[name_col].apply(classify_type)

    return df


def keyword_tokens(text: str) -> List[str]:
    return [t.strip().lower() for t in text.split() if t.strip()]


def search_catalog(df: pd.DataFrame, query: str, preferred_type: str = None) -> pd.DataFrame:
    """
    Very simple keyword AND search over the pre-built __search_text field.
    Optionally bias towards a product type (towel/sheet/etc).
    """
    q = query.strip().lower()
    if not q:
        return df.iloc[0:0].copy()

    tokens = keyword_tokens(q)
    if not tokens:
        return df.iloc[0:0].copy()

    mask = df["__search_text"].apply(
        lambda txt: all(t in txt for t in tokens)
    )

    candidates = df[mask].copy()

    # If nothing matched strictly, fall back to OR search
    if candidates.empty:
        mask = df["__search_text"].apply(
            lambda txt: any(t in txt for t in tokens)
        )
        candidates = df[mask].copy()

    if candidates.empty:
        return candidates

    # Simple relevance score: number of token hits
    def score_row(txt: str) -> int:
        return sum(txt.count(t) for t in tokens)

    candidates["__score"] = candidates["__search_text"].apply(score_row)

    # Optional type bias (towels vs sheets, etc.)
    if preferred_type:
        # small boost to matching types
        candidates["__score"] += (candidates["__type"] == preferred_type).astype(int) * 3

    candidates = candidates.sort_values("__score", ascending=False)

    return candidates


def infer_preferred_type(user_query: str) -> str:
    q = user_query.lower()
    if "beach" in q and "towel" in q:
        return "beach_towel"
    if "towel" in q or "bathroom" in q or "bath" in q:
        return "towel"
    if "sheet" in q or "bedding" in q or "duvet" in q:
        return "sheet"
    if "quilt" in q or "comforter" in q:
        return "quilt"
    if "rug" in q or "mat" in q:
        return "rug"
    return None


def top_n_products(df: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    return df.head(n).copy()


def build_product_prompt_snippet(df: pd.DataFrame, max_items: int = 8) -> str:
    """
    Create a short natural-language snippet summarising a handful of products
    for the image generator prompts.
    """
    if df.empty:
        return ""

    name_col = "Product name" if "Product name" in df.columns else df.columns[0]
    color_col = "Color" if "Color" in df.columns else None

    lines = []
    for _, row in df.head(max_items).iterrows():
        name = row.get(name_col, "")
        color = row.get(color_col, "")
        if color:
            lines.append(f"- {name} in {color}")
        else:
            lines.append(f"- {name}")

    if not lines:
        return ""

    return (
        "Use only Market & Place textiles inspired by the following real products:\n"
        + "\n".join(lines)
    )


def generate_image(prompt: str, size: str = "1024x1024") -> bytes:
    """
    Call OpenAI's image generation API and return raw image bytes.
    """
    resp = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size=size,
        n=1,
    )

    b64 = resp.data[0].b64_json
    return base64.b64decode(b64)


# =========================
# PAGE HEADER
# =========================

# Logo row, centered
logo_cols = st.columns([1, 2, 1])
with logo_cols[1]:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_column_width=False, width=420)
    else:
        st.markdown("### Market & Place")

st.markdown(
    "<h1 style='text-align:center; margin-bottom:0;'>Market & Place AI Stylist</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='text-align:center;'>"
    "Chat with an AI stylist, search the Market & Place catalog, "
    "and generate concept visualizations using your own product file."
    "</p>",
    unsafe_allow_html=True,
)

st.markdown(
    "[← Return to Market & Place website](https://marketandplace.co/)",
)

st.markdown("---")

# =========================
# LOAD CATALOG
# =========================

try:
    catalog_df = load_catalog()
except Exception as e:
    st.error(f"Could not load catalog: {e}")
    st.stop()

name_col = "Product name" if "Product name" in catalog_df.columns else catalog_df.columns[0]
color_col = "Color" if "Color" in catalog_df.columns else None
price_col = "Price" if "Price" in catalog_df.columns else None
amazon_col = "raw_amazon" if "raw_amazon" in catalog_df.columns else None
image_url_col = "Image URL:" if "Image URL:" in catalog_df.columns else None


# =========================
# LAYOUT: LEFT (STYLIST) / RIGHT (IMAGES)
# =========================

left_col, right_col = st.columns([1.15, 1])

# ---- LEFT: Ask the AI stylist ----
with left_col:
    st.subheader("Ask the AI stylist", anchor="stylist")

    with st.form("stylist_form"):
        user_query = st.text_input(
            "Describe what you're looking for:",
            placeholder="e.g. neutral queen bedding under $80 for a bright room",
            key="stylist_query",
        )
        submitted = st.form_submit_button("Send")

    stylist_results = None
    if submitted and user_query.strip():
        preferred_type = infer_preferred_type(user_query)
        stylist_results = search_catalog(catalog_df, user_query, preferred_type=preferred_type)

        st.markdown(f"### 🧵 {user_query.strip()}")

        if stylist_results is None or stylist_results.empty:
            st.info(
                "I couldn't find any matching products in the Market & Place catalog for that request. "
                "Try being more specific — for example, include product type and colour like "
                "“luxury bath towels in grey” or “navy flannel sheet set”."
            )
        else:
            st.write(
                "Here are some Market & Place products that match your request "
                "and could work well in your space:"
            )

            # Show top 6 nicely
            for idx, (_, row) in enumerate(top_n_products(stylist_results, 6).iterrows(), start=1):
                name = row.get(name_col, "")
                color = row.get(color_col, "")
                price = row.get(price_col, "")
                amazon = row.get(amazon_col, "")
                img_url = row.get(image_url_col, "")

                st.markdown(f"**{idx}. {name}**")

                bullet_lines = []
                if color:
                    bullet_lines.append(f"- Color: {color}")
                if price:
                    bullet_lines.append(f"- Price: {price}")
                if amazon:
                    bullet_lines.append(f"- [View on Amazon]({amazon})")

                if bullet_lines:
                    st.markdown("\n".join(bullet_lines))

                if isinstance(img_url, str) and img_url.startswith("http"):
                    st.image(img_url, use_column_width=True)

                st.markdown("---")

    # ---- Quick catalog peek (always under the search) ----
    st.subheader("Quick catalog peek")
    peek_query = st.text_input(
        "Filter products by keyword:",
        placeholder="e.g. cabana stripe, flannel sheet, navy",
        key="peek_query",
    )

    peek_df = search_catalog(catalog_df, peek_query) if peek_query.strip() else catalog_df.iloc[0:0]

    if peek_query.strip() and peek_df.empty:
        st.info("No products matched that keyword in the catalog.")
    elif not peek_df.empty:
        st.caption("Top matching products:")
        for idx, (_, row) in enumerate(top_n_products(peek_df, 10).iterrows(), start=1):
            name = row.get(name_col, "")
            color = row.get(color_col, "")
            price = row.get(price_col, "")
            amazon = row.get(amazon_col, "")
            img_url = row.get(image_url_col, "")

            with st.container():
                cols = st.columns([1, 2])
                with cols[0]:
                    if isinstance(img_url, str) and img_url.startswith("http"):
                        st.image(img_url, use_column_width=True)
                with cols[1]:
                    st.markdown(f"**{idx}. {name}**")
                    if color:
                        st.write(f"Color: {color}")
                    if price:
                        st.write(f"Price: {price}")
                    if amazon:
                        st.markdown(f"[View on Amazon]({amazon})")
            st.markdown("---")


# ---- RIGHT: AI concept visualizer ----
with right_col:
    st.subheader("AI concept visualizer")

    st.markdown(
        "Generate a **styled concept image** using Market & Place products.\n\n"
        "- **In my room photo**: upload a photo and describe what textiles you'd like. "
        "The AI will try to keep the same layout and only change textiles.\n"
        "- **Store shelf / showroom**: skip the photo and generate a retail shelf/aisle view "
        "with Market & Place products."
    )

    mode = st.radio(
        "What do you want to visualize?",
        ["In my room photo", "Store shelf / showroom"],
        horizontal=True,
    )

    if mode == "In my room photo":
        room_type = st.selectbox("Room type:", ["bathroom", "bedroom", "living room", "kitchen", "other"])

        uploaded_img = st.file_uploader(
            "Upload a reference photo of your room:",
            type=["jpg", "jpeg", "png"],
        )

        concept_text = st.text_input(
            "What would you like to visualize?",
            placeholder="e.g. add navy cabana stripe towels and a matching bath rug",
            key="room_visualize_text",
        )

        if uploaded_img is not None:
            st.image(uploaded_img, caption="Uploaded room (reference)", use_column_width=True)

        if st.button("Generate concept image", key="generate_room"):
            if uploaded_img is None:
                st.error("Please upload a room image first, or switch to 'Store shelf / showroom'.")
            elif not concept_text.strip():
                st.error("Please describe what you'd like to visualize (towels, bedding, colours, etc.).")
            else:
                with st.spinner("Generating concept image…"):
                    try:
                        # Use the stylist results, if any, as product inspiration; otherwise, whole catalog
                        source_df = stylist_results if isinstance(stylist_results, pd.DataFrame) and not stylist_results.empty else catalog_df
                        product_snippet = build_product_prompt_snippet(source_df)

                        prompt = f"""
High quality interior design concept rendering of a {room_type}.
The layout, camera angle and major fixtures should stay consistent with the user's reference photo
(the photo is not edited directly – this is a conceptual re-imagining).

Only adjust or add **textiles** that make sense for a {room_type}:
- towels, bath rugs and shower curtains in a bathroom
- bedding, throws and decorative pillows in a bedroom
- throws, cushions and rugs in a living room
- tea towels, floor mats and runners in a kitchen

Follow the user's request exactly:
\"\"\"{concept_text.strip()}\"\"\".

Hard rules:
- Do NOT change the type of room (bathroom stays bathroom, bedroom stays bedroom, etc.).
- Do NOT add new windows, doors, toilets, sinks, bathtubs, or structural elements.
- Do NOT add random décor or furniture that is not textiles.
- Textiles must be inspired only by real Market & Place products.
{product_snippet}
"""

                        img_bytes = generate_image(prompt)
                        st.image(img_bytes, caption="AI-generated style concept", use_column_width=True)
                    except Exception as e:
                        st.error(f"Image generation failed: {e}")

    else:  # Store shelf / showroom
        aisle_type = st.selectbox(
            "Shelf / showroom focus:",
            ["towels", "beach towels", "bedding & sheets", "mixed textiles"],
        )

        shelf_text = st.text_input(
            "Optional details (colours, vibe, etc.):",
            placeholder="e.g. bright cabana stripes in aqua and navy",
            key="store_visualize_text",
        )

        if st.button("Generate concept image", key="generate_store"):
            with st.spinner("Generating store shelf concept…"):
                try:
                    # Pick relevant subset to bias the prompt
                    if aisle_type == "towels":
                        sub = catalog_df[catalog_df["__type"].isin(["towel"])]
                    elif aisle_type == "beach towels":
                        sub = catalog_df[catalog_df["__type"].isin(["beach_towel"])]
                    elif aisle_type == "bedding & sheets":
                        sub = catalog_df[catalog_df["__type"].isin(["sheet", "quilt"])]
                    else:
                        sub = catalog_df

                    product_snippet = build_product_prompt_snippet(sub)

                    prompt = f"""
Ultra realistic photograph of a retail store shelf / aisle display featuring Market & Place textiles only.
Focus on neatly folded stacks and hanging products on simple white shelves.

Aisle focus: {aisle_type}.
Extra styling notes from the user (if any):
\"\"\"{shelf_text.strip()}\"\"\".

Hard rules:
- This must clearly be a store shelf or showroom – not a bathroom or bedroom.
- Do NOT show any other brands or logos.
- Do NOT invent fictional products – textiles must be inspired only by real Market & Place items.
- Emphasise clear product organisation and colour blocking.

{product_snippet}
"""

                    img_bytes = generate_image(prompt)
                    st.image(img_bytes, caption="AI-generated store shelf concept", use_column_width=True)
                except Exception as e:
                    st.error(f"Image generation failed: {e}")












