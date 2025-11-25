# streamlit_app.py
import base64
import io
import os
from typing import List, Tuple

import pandas as pd
import streamlit as st
from PIL import Image
from openai import OpenAI


# ---------- CONFIG ----------

st.set_page_config(
    page_title="Market & Place AI Stylist",
    page_icon="🧵",
    layout="wide",
)

client = OpenAI()


# ---------- DATA LOADING ----------

@st.cache_data
def load_catalog() -> pd.DataFrame:
    """
    Load the Market & Place catalog from the Excel file
    and pre-compute a simple 'category' for each product.
    """
    df = pd.read_excel("market_and_place_products.xlsx")

    # Normalise column names just in case
    df.columns = [c.strip() for c in df.columns]

    if "Product name" not in df.columns:
        raise ValueError("Excel file must contain a 'Product name' column.")

    def infer_category(name: str) -> str:
        n = str(name).lower()
        if "towel" in n:
            # More specific towel sub-types
            if "beach" in n or "cabana" in n:
                return "beach_towel"
            if "luxury" in n or "turkish" in n or "spa" in n or "hotel" in n:
                return "luxury_towel"
            return "towel"
        if any(x in n for x in ["sheet", "bedding", "duvet", "quilt", "comforter"]):
            return "bedding"
        if "rug" in n or "mat" in n:
            return "rug"
        return "other"

    df["category"] = df["Product name"].apply(infer_category)
    return df


CATALOG = load_catalog()


# ---------- HELPER: CATALOG SEARCH (NO MORE DF ERRORS) ----------

def score_product(query: str, name: str, color: str = "") -> int:
    """
    Very simple keyword scoring. Pure Python integers -> no pandas shape issues.
    Higher score means more relevant.
    """
    q = query.lower()
    name_l = str(name).lower()
    color_l = str(color).lower()

    score = 0

    # Base: keyword hits in the name
    for token in q.split():
        if token in name_l:
            score += 4
        if token in color_l:
            score += 1

    # Some special boosts
    if "luxury" in q and ("luxury" in name_l or "turkish" in name_l or "spa" in name_l):
        score += 8
    if "beach" in q and ("beach" in name_l or "cabana" in name_l):
        score += 8

    if "towel" in q and "towel" in name_l:
        score += 6
    if any(x in q for x in ["sheet", "bedding", "quilt", "duvet", "comforter"]) and any(
        x in name_l for x in ["sheet", "bedding", "quilt", "duvet", "comforter"]
    ):
        score += 6

    # Light bias towards shorter names that match (usually more specific SKUs)
    score -= max(0, len(name_l) // 40)

    return score


def detect_query_categories(query: str) -> List[str]:
    q = query.lower()

    # Towels vs bedding vs rugs
    if "towel" in q or "washcloth" in q:
        if "beach" in q:
            return ["beach_towel"]
        if any(x in q for x in ["luxury", "hotel", "spa"]):
            return ["luxury_towel"]
        return ["towel", "luxury_towel", "beach_towel"]

    if any(x in q for x in ["sheet", "bedding", "duvet", "quilt", "comforter"]):
        return ["bedding"]

    if "rug" in q or "mat" in q:
        return ["rug"]

    # Fallback: anything
    return []


def search_catalog(query: str, top_k: int = 5) -> pd.DataFrame:
    """
    Strictly search inside the Excel catalog.
    No hallucinations, no outside products.
    """
    if query is None:
        query = ""
    query = query.strip()
    df = CATALOG.copy()

    allowed_cats = detect_query_categories(query)
    if allowed_cats:
        df = df[df["category"].isin(allowed_cats)].copy()

    # If we filtered everything out, fall back to full catalog
    if df.empty:
        df = CATALOG.copy()

    scores = [
        score_product(query, name=row["Product name"], color=row.get("Color", ""))
        for _, row in df.iterrows()
    ]

    df["score"] = scores
    df = df.sort_values("score", ascending=False)

    # If everything scored 0, just take the first few rows
    if df["score"].max() <= 0:
        return df.head(top_k)

    df = df[df["score"] > 0]
    return df.head(top_k)


# ---------- HELPER: RECOMMENDATION TEXT ----------

def render_recommendations(query: str, df: pd.DataFrame):
    """
    Render the nice layout you liked: heading with emoji + list of products.
    """
    spool_emoji = "🧵"
    heading = f"{spool_emoji} {query.lower()}"
    st.markdown(f"### {heading}")

    if df.empty:
        st.info("I couldn’t find anything in the Market & Place catalog that matches that request.")
        return

    st.write(
        "Here are some Market & Place products that match your request and only come from the official catalog:"
    )

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        name = str(row.get("Product name", "")).strip()
        color = str(row.get("Color", "")).strip()
        price = row.get("Price", "")
        url = str(row.get("raw_amazon", "")).strip()  # RAW – no edits, no tracking
        img_url = str(row.get("Image URL:", "")).strip()

        st.markdown(f"**{idx}. {name}**")
        info_lines = []
        if color:
            info_lines.append(f"- **Color:** {color}")
        if price != "" and not pd.isna(price):
            info_lines.append(f"- **Price:** {price}")
        if info_lines:
            st.markdown("\n".join(info_lines))

        if url:
            # Amazon nuclear rule: show the raw string, exactly.
            st.markdown(f"- [View on Amazon]({url})")

        if img_url:
            st.image(img_url, width=220)


# ---------- HELPER: IMAGE GENERATION ----------

def build_image_prompt(
    mode: str,
    room_type: str,
    user_instruction: str,
    matched_products: pd.DataFrame,
) -> str:
    """
    Construct a super-strict prompt for the image model.
    We DO NOT allow it to invent random products; it must use only the given list as inspiration.
    """
    lines = []

    if mode == "room":
        lines.append(
            f"Generate a realistic {room_type} concept image. "
            "The layout, architecture and hard fixtures (walls, windows, doors, vanity, toilet, shower, tub, shelves, lighting, mirrors) "
            "must remain simple and consistent – do NOT add new furniture or structural elements. "
            "Focus on textiles only."
        )
        lines.append(
            "You are ONLY allowed to change or add soft home textiles: towels, shower curtain, bath rug, bath mat, bathrobe, and similar items."
        )
    else:
        # store / showroom
        lines.append(
            "Generate a realistic store shelf / showroom view featuring neatly folded and hanging textiles. "
            "Show long runs of shelves with stacks of towels and possibly matching rugs. "
            "Avoid non-textile products."
        )

    if user_instruction.strip():
        lines.append(f"User request: {user_instruction.strip()}")

    # Product list (strict catalog only)
    if not matched_products.empty:
        lines.append(
            "Use ONLY the following Market & Place catalog products as inspiration for colors, stripes and patterns. "
            "Do NOT invent any new products or brands."
        )
        for _, row in matched_products.iterrows():
            name = str(row.get("Product name", "")).strip()
            color = str(row.get("Color", "")).strip()
            lines.append(f"- {name} (Color: {color})")

    lines.append(
        "Overall style: natural lighting, clean photography, no logos, no text overlays, no extra decorative objects beyond simple plants or jars."
    )

    return "\n".join(lines)


def generate_concept_image(
    mode: str,
    room_type: str,
    user_instruction: str,
    ref_image: Image.Image | None,
) -> Tuple[Image.Image | None, str | None]:
    """
    Call gpt-image-1 for a concept render.

    NOTE: we currently use the uploaded photo only as a reference ON SCREEN;
    the model gets a very strict text prompt instead of a true pixel edit so that
    nothing crashes and the behaviour stays predictable.
    """
    try:
        # For the image model, we just give a tight textual prompt and
        # our catalog products – NO hallucinated SKUs.
        matched = search_catalog(user_instruction or room_type, top_k=6)
        prompt = build_image_prompt(
            mode=mode,
            room_type=room_type,
            user_instruction=user_instruction,
            matched_products=matched,
        )

        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024",
            n=1,
            response_format="b64_json",
        )

        b64_data = response.data[0].b64_json
        image_bytes = base64.b64decode(b64_data)
        img = Image.open(io.BytesIO(image_bytes))
        return img, None

    except Exception as e:
        return None, str(e)


# ---------- UI: HEADER & LOGO ----------

# Centered logo
logo_cols = st.columns([1, 2, 1])
with logo_cols[1]:
    st.image("logo.png", use_column_width=True)

st.markdown(
    "<h1 style='text-align:center; margin-top: 0.4rem;'>Market & Place AI Stylist</h1>",
    unsafe_allow_html=True,
)
st.write(
    "Chat with an AI stylist, search the Market & Place catalog, and generate concept visualizations using your own product file."
)

st.markdown(
    "[← Return to Market & Place website](https://marketandplace.co/)",
)


# ---------- SECTION 1: ASK THE AI STYLIST ----------

st.markdown("## Ask the AI stylist")

with st.form("stylist_form", clear_on_submit=False):
    user_query = st.text_input(
        "Describe what you're looking for:",
        placeholder="e.g. luxury towels for a modern bathroom, neutral queen bedding under $80, etc.",
        key="stylist_query",
    )
    submitted = st.form_submit_button("Send")

if submitted and user_query.strip():
    # Clear previous result by simply re-rendering only the latest query
    try:
        recs = search_catalog(user_query, top_k=6)
        render_recommendations(user_query, recs)
    except Exception as e:
        st.error(f"Could not search the catalog: {e}")


# ---------- SECTION 2: QUICK CATALOG PEEK ----------

st.markdown("---")
st.markdown("## Quick catalog peek")

peek_keyword = st.text_input(
    "Filter products by keyword (optional):",
    placeholder="e.g. towel, sheet, cabana stripe, grey, queen",
    key="peek_keyword",
)

peek_df = CATALOG.copy()
if peek_keyword.strip():
    k = peek_keyword.lower()
    mask = peek_df["Product name"].str.lower().str.contains(k)
    if "Color" in peek_df.columns:
        mask |= peek_df["Color"].astype(str).str.lower().str.contains(k)
    peek_df = peek_df[mask]

peek_df = peek_df.head(20)

for _, row in peek_df.iterrows():
    cols = st.columns([1, 3])
    img_url = str(row.get("Image URL:", "")).strip()
    if img_url:
        cols[0].image(img_url, use_column_width=True)
    name = str(row.get("Product name", "")).strip()
    color = str(row.get("Color", "")).strip()
    price = row.get("Price", "")
    url = str(row.get("raw_amazon", "")).strip()

    cols[1].markdown(f"**{name}**")
    if color:
        cols[1].markdown(f"- **Color:** {color}")
    if price != "" and not pd.isna(price):
        cols[1].markdown(f"- **Price:** {price}")
    if url:
        cols[1].markdown(f"- [View on Amazon]({url})")

st.markdown("---")


# ---------- SECTION 3: AI CONCEPT VISUALIZER ----------

st.markdown("## 🛋️ AI concept visualizer")

st.write(
    "Generate a styled version of a room or a store shelf using *only* Market & Place textiles "
    "as inspiration. The AI is explicitly told **not** to invent products that aren’t in the catalog "
    "and to avoid changing walls, furniture, or fixtures."
)

mode_choice = st.radio(
    "What do you want to visualize?",
    options=["In my room photo", "Store shelf / showroom"],
    horizontal=False,
)

room_mode = "room" if mode_choice == "In my room photo" else "store"

if room_mode == "room":
    room_type = st.selectbox(
        "Room type:",
        options=["bathroom", "bedroom", "living room", "kids' room"],
        index=0,
    )
else:
    room_type = "store / showroom"

uploaded_file = st.file_uploader(
    "Upload a reference photo of your room (or store aisle):",
    type=["jpg", "jpeg", "png"],
)

ref_image = None
if uploaded_file is not None:
    ref_image = Image.open(uploaded_file)
    st.image(ref_image, caption="Uploaded room (reference)", use_column_width=True)

image_instruction = st.text_input(
    "What would you like to visualize?",
    placeholder="e.g. bold striped towels in blue and white on the towel rack",
    key="image_instruction",
)

generate_clicked = st.button("Generate concept image")

if generate_clicked:
    if room_mode == "room" and ref_image is None:
        st.error("Please upload a room image first, or switch to 'Store shelf / showroom'.")
    else:
        with st.spinner("Generating concept image..."):
            img, err = generate_concept_image(
                mode=room_mode,
                room_type=room_type,
                user_instruction=image_instruction,
                ref_image=ref_image,
            )
        if err:
            st.error(f"Image generation failed: {err}")
        elif img is not None:
            st.image(img, caption="AI-generated style concept", use_column_width=True)
        else:
            st.error("Image generation failed for an unknown reason.")












