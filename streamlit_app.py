import os
import base64
from io import BytesIO

import pandas as pd
import streamlit as st
from openai import OpenAI

# ---------- OPENAI CLIENT ----------
client = OpenAI()  # uses OPENAI_API_KEY from env

# ---------- LOAD PRODUCT CATALOG ----------
@st.cache_data
def load_catalog():
    df = pd.read_excel("market_and_place_products.xlsx")
    # Normalise column names once
    df.columns = [c.strip() for c in df.columns]
    return df

catalog_df = load_catalog()

# Convenience accessors (adjust these if your column names differ)
NAME_COL = "Product name"
COLOR_COL = "Color"
PRICE_COL = "Price"
AMAZON_COL = "raw_amazon"
IMAGE_COL = "Image URL:"


# ---------- PRODUCT SEARCH / FILTERING ----------

def score_product(row, query: str) -> int:
    """
    Very simple keyword scoring. Higher = more relevant.
    This is what controls "luxury towels vs beach towels".
    """
    q = query.lower()
    name = str(row.get(NAME_COL, "")).lower()

    score = 0

    # Basic matching
    if "towel" in q and "towel" in name:
        score += 5
    if "sheet" in q or "bedding" in q or "bed" in q:
        if any(word in name for word in ["sheet", "bedding", "quilt", "comforter"]):
            score += 5

    # Luxury vs beach logic
    if "luxury" in q:
        if "luxury" in name:
            score += 8
        if "beach" in name:
            score -= 8  # strongly down-rank beach if asking for luxury

    if "beach" in q:
        if "beach" in name:
            score += 8

    if "beach" not in q and "beach" in name:
        score -= 4  # do not prefer beach towels unless asked

    # Room-type hints
    if any(k in q for k in ["bathroom", "bath", "shower"]):
        if "towel" in name or "bath" in name:
            score += 3
    if any(k in q for k in ["bedroom", "bed", "duvet"]):
        if any(word in name for word in ["sheet", "bedding", "quilt", "comforter"]):
            score += 3

    # Fallback mild score if name loosely matches query words
    for token in q.split():
        if token and token in name:
            score += 1

    return score


def find_products_for_query(query: str, max_results: int = 6):
    scores = []
    for idx, row in catalog_df.iterrows():
        s = score_product(row, query)
        if s > 0:  # only keep things that look at least somewhat relevant
            scores.append((s, idx))

    if not scores:
        # If nothing scored >0, fall back to showing nothing; better than hallucinating
        return []

    scores.sort(reverse=True, key=lambda x: x[0])
    top_indices = [idx for _, idx in scores[:max_results]]

    return catalog_df.loc[top_indices]


# ---------- UTILS: IMAGE DISPLAY FROM B64 ----------

def display_base64_image(b64_str: str, caption: str = None):
    image_bytes = base64.b64decode(b64_str)
    st.image(image_bytes, use_column_width=True, caption=caption)


# ---------- PAGE CONFIG ----------

st.set_page_config(
    page_title="Market & Place AI Stylist",
    layout="wide",
)

# ---------- LOGO, TITLE, TAGLINE ----------

# Centered logo
logo_cols = st.columns([1, 2, 1])
with logo_cols[1]:
    try:
        st.image("logo.png", use_column_width=True)
    except Exception:
        st.markdown(
            "<h2 style='text-align:center;'>Market & Place</h2>",
            unsafe_allow_html=True,
        )

# Title
st.markdown(
    "<h1 style='text-align:center; margin-top:0.3rem;'>Market & Place AI Stylist</h1>",
    unsafe_allow_html=True,
)

# Tagline
st.markdown(
    "<p style='text-align:center;'>"
    "Chat with an AI stylist, search the Market & Place catalog, "
    "and generate concept visualizations using your own product file."
    "</p>",
    unsafe_allow_html=True,
)

# Return link
st.markdown(
    "[← Return to Market & Place website](https://marketandplace.co/)",
)

st.markdown("---")

# ---------- LAYOUT: LEFT = CHAT + CATALOG, RIGHT = IMAGE ----------

left_col, right_col = st.columns([1.2, 1])

# ---------- LEFT: ASK THE AI STYLIST & CATALOG PEEK ----------

with left_col:
    st.subheader("Ask the AI stylist")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Form so Enter submits
    with st.form("stylist_form", clear_on_submit=True):
        user_query = st.text_input(
            "Describe what you're looking for:",
            placeholder="e.g. luxury bath towels in neutral colours",
        )
        submitted = st.form_submit_button("Send")

    if submitted and user_query.strip():
        products = find_products_for_query(user_query)
        st.session_state.chat_history.insert(
            0,
            {
                "query": user_query.strip(),
                "products": products,
            },
        )

    # Render history (newest first, stays below the search bar)
    for entry in st.session_state.chat_history:
        query = entry["query"]
        products = entry["products"]

        st.markdown(f"### 🧵 {query}")

        if products is None or len(products) == 0:
            st.info(
                "I couldn't find anything in the current Market & Place catalog that "
                "clearly matches this request. Try rephrasing or being more specific."
            )
            continue

        st.write(
            "Here are some Market & Place products that match your request and could "
            "work well in your space:"
        )

        for i, (_, row) in enumerate(products.iterrows(), start=1):
            name = str(row.get(NAME_COL, "Unnamed product"))
            color = str(row.get(COLOR_COL, ""))
            price = row.get(PRICE_COL, "")
            amazon_url = str(row.get(AMAZON_COL, "")).strip()
            image_url = str(row.get(IMAGE_COL, "")).strip()

            st.markdown(f"**{i}. {name}**")
            if color:
                st.write(f"- Color: {color}")
            if price != "":
                st.write(f"- Price: {price}")
            if amazon_url:
                # IMPORTANT: show URL exactly as stored, no modifications
                st.markdown(f"- [View on Amazon]({amazon_url})")

            if image_url and image_url.lower().startswith("http"):
                st.image(image_url, width=160)

            st.markdown("---")

    # Quick catalog peek directly under chat, moving down as chat grows
    st.subheader("Quick catalog peek")

    peek_query = st.text_input(
        "Filter products by keyword:",
        placeholder="e.g. striped towel, flannel sheet, navy",
        key="peek_query",
    )

    if peek_query.strip():
        mask = catalog_df[NAME_COL].str.contains(
            peek_query.strip(), case=False, na=False
        )
        peek_df = catalog_df[mask].head(10)
    else:
        peek_df = catalog_df.head(10)

    if peek_df.empty:
        st.info("No products matched that filter in the Market & Place file.")
    else:
        for _, row in peek_df.iterrows():
            name = str(row.get(NAME_COL, "Unnamed product"))
            color = str(row.get(COLOR_COL, ""))
            price = row.get(PRICE_COL, "")
            amazon_url = str(row.get(AMAZON_COL, "")).strip()
            image_url = str(row.get(IMAGE_COL, "")).strip()

            row_cols = st.columns([1, 3])
            with row_cols[0]:
                if image_url and image_url.lower().startswith("http"):
                    st.image(image_url, use_column_width=True)
            with row_cols[1]:
                st.markdown(f"**{name}**")
                if color:
                    st.write(f"- Color: {color}")
                if price != "":
                    st.write(f"- Price: {price}")
                if amazon_url:
                    st.markdown(f"- [View on Amazon]({amazon_url})")
            st.markdown("---")

# ---------- RIGHT: IMAGE CONCEPT VISUALIZER ----------

with right_col:
    st.subheader("🖼️ Your image")

    st.write(
        "Upload a photo of your room (bathroom, bedroom, etc.), or, if you're a "
        "distributor, leave it empty and generate a store shelf / showroom view."
    )

    room_type = st.selectbox(
        "Room type (used to enforce strict styling rules):",
        ["bathroom", "bedroom", "store shelf / showroom"],
        index=0,
    )

    viz_mode = st.radio(
        "What do you want to visualize?",
        ["In my room photo", "Store shelf view"],
        index=0,
    )

    uploaded_image = st.file_uploader(
        "Upload room / shelf reference photo (optional for store shelf view):",
        type=["jpg", "jpeg", "png"],
    )

    change_description = st.text_input(
        "Describe what you want the AI to change (textiles only):",
        placeholder=(
            "e.g. Replace all towels with our luxury white towel collection, "
            "add a matching bath mat and shower curtain."
        ),
        key="change_description",
    )

    generate_clicked = st.button("Generate concept image")

    if generate_clicked:
        # Safety checks
        if viz_mode == "In my room photo" and uploaded_image is None:
            st.error("Please upload a room photo to visualize textiles in your space.")
        else:
            with st.spinner("Generating concept image using Market & Place products..."):
                try:
                    # Build the ultra-strict prompt
                    core_rules = [
                        "You are generating a concept image for Market & Place textiles.",
                        "You must ONLY show textiles that could reasonably match products "
                        "from the Market & Place catalog (towels, bath mats, shower curtains, "
                        "bedding, quilts, decorative pillows, curtains).",
                        "Do NOT invent or display fictional brands or logos.",
                        "Do NOT change the architecture or furniture layout of the room.",
                        "Do NOT add bathtubs, beds, sinks, toilets, mirrors, or shelving "
                        "that are not already implied by the scene type.",
                        "Your job is ONLY to change or showcase textiles and colours.",
                    ]

                    if room_type == "bathroom":
                        core_rules.append(
                            "Bathroom stays a bathroom: never turn it into a bedroom or living room."
                        )
                    if room_type == "bedroom":
                        core_rules.append(
                            "Bedroom stays a bedroom: never add showers, toilets or bathroom fixtures."
                        )

                    if viz_mode == "Store shelf view":
                        core_rules.append(
                            "Render a clean store-shelf / showroom wall full of neatly folded and "
                            "hung Market & Place towels and related textiles. "
                            "Use simple retail shelving. Do NOT show a bath, shower, sink, or bed."
                        )
                    else:
                        core_rules.append(
                            "Use the uploaded image as a visual reference: keep walls, tiles, "
                            "furniture and fixtures identical. Only change the textiles."
                        )

                    if change_description.strip():
                        user_change = change_description.strip()
                    else:
                        user_change = (
                            "Update the textiles to showcase a cohesive Market & Place collection."
                        )

                    # A compact summary of catalog styles to bias colours/patterns
                    sample_names = ", ".join(
                        catalog_df[NAME_COL].head(25).astype(str).tolist()
                    )

                    image_prompt = (
                        "STRICT MARKET & PLACE TEXTILE RENDER INSTRUCTIONS:\n"
                        + "\n".join(f"- {rule}" for rule in core_rules)
                        + "\n\nUser request for textiles:\n"
                        f"{user_change}\n\n"
                        "Market & Place catalog style hints (names only, not URLs):\n"
                        f"{sample_names}\n"
                    )

                    img_resp = client.images.generate(
                        model="gpt-image-1",
                        prompt=image_prompt,
                        size="1024x1024",
                    )

                    b64 = img_resp.data[0].b64_json
                    display_base64_image(
                        b64,
                        caption="AI-generated concept using Market & Place-inspired textiles",
                    )
                except Exception as e:
                    st.error(
                        "Could not generate an edited version of your room image. "
                        f"Technical details: {e}"
                    )












