import base64
import io
from typing import List, Tuple

import pandas as pd
import streamlit as st
from PIL import Image
from openai import OpenAI

# --------- BASIC CONFIG ---------

st.set_page_config(
    page_title="Market & Place AI Stylist",
    page_icon="🧵",
    layout="wide",
)

client = OpenAI()


# --------- LOAD CATALOG ---------

@st.cache_data
def load_catalog() -> pd.DataFrame:
    df = pd.read_excel("market_and_place_products.xlsx")
    df.columns = [c.strip() for c in df.columns]

    if "Product name" not in df.columns:
        raise ValueError("Excel must contain 'Product name' column")

    def infer_category(name: str) -> str:
        n = str(name).lower()
        if "towel" in n:
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


# --------- CATALOG SEARCH HELPERS ---------

def score_product(query: str, name: str, color: str = "") -> int:
    q = query.lower()
    name_l = str(name).lower()
    color_l = str(color).lower()
    score = 0

    for token in q.split():
        if token in name_l:
            score += 4
        if token in color_l:
            score += 1

    if "luxury" in q and ("luxury" in name_l or "turkish" in name_l or "spa" in name_l or "hotel" in name_l):
        score += 8
    if "beach" in q and ("beach" in name_l or "cabana" in name_l):
        score += 8

    if "towel" in q and "towel" in name_l:
        score += 6
    if any(x in q for x in ["sheet", "bedding", "duvet", "quilt", "comforter"]) and any(
        x in name_l for x in ["sheet", "bedding", "duvet", "quilt", "comforter"]
    ):
        score += 6

    score -= max(0, len(name_l) // 40)
    return score


def detect_query_categories(query: str) -> List[str]:
    q = query.lower()
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
    return []


def search_catalog(query: str, top_k: int = 5) -> pd.DataFrame:
    if query is None:
        query = ""
    query = query.strip()

    df = CATALOG.copy()
    allowed = detect_query_categories(query)
    if allowed:
        df = df[df["category"].isin(allowed)].copy()
        if df.empty:
            df = CATALOG.copy()

    scores = [
        score_product(query, row["Product name"], color=row.get("Color", ""))
        for _, row in df.iterrows()
    ]
    df["score"] = scores
    df = df.sort_values("score", ascending=False)

    if df["score"].max() <= 0:
        return df.head(top_k)

    df = df[df["score"] > 0]
    return df.head(top_k)


def render_recommendations(query: str, df: pd.DataFrame):
    spool = "🧵"
    heading = f"{spool} {query.lower()}"
    st.markdown(f"### {heading}")

    if df.empty:
        st.info("I couldn’t find anything in the Market & Place catalog that matches that request.")
        return

    st.write(
        "Here are some Market & Place products that match your request "
        "and come only from the official catalog:"
    )

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        name = str(row.get("Product name", "")).strip()
        color = str(row.get("Color", "")).strip()
        price = row.get("Price", "")
        url = str(row.get("raw_amazon", "")).strip()
        img_url = str(row.get("Image URL:", "")).strip()

        st.markdown(f"**{idx}. {name}**")
        lines = []
        if color:
            lines.append(f"- **Color:** {color}")
        if price != "" and not pd.isna(price):
            lines.append(f"- **Price:** {price}")
        if lines:
            st.markdown("\n".join(lines))
        if url:
            st.markdown(f"- [View on Amazon]({url})")
        if img_url:
            st.image(img_url, width=220)


# --------- IMAGE GENERATION HELPERS ---------

def build_image_prompt(
    mode: str,
    room_type: str,
    user_instruction: str,
    matched_products: pd.DataFrame,
) -> str:
    lines: List[str] = []

    if mode == "room":
        lines.append(
            f"Generate a realistic {room_type} concept image. "
            "The layout and hard fixtures (walls, windows, doors, vanity, toilet, shower, tub, shelves, mirrors, lighting) "
            "must remain simple and consistent. Do NOT add new furniture or change the architecture."
        )
        lines.append(
            "You are ONLY allowed to change or add soft textiles: towels, shower curtain, bath rug or mat, bedding, pillows, throws, and similar items."
        )
    else:
        lines.append(
            "Generate a realistic store shelf / showroom view featuring neatly folded and hanging textiles. "
            "Use long runs of shelves with stacks of towels or other textiles. Avoid non-textile products."
        )

    if user_instruction.strip():
        lines.append(f"User request: {user_instruction.strip()}")

    if not matched_products.empty:
        lines.append(
            "Use ONLY the following Market & Place catalog products as inspiration for colors, stripes and patterns. "
            "Do NOT invent brands or SKUs."
        )
        for _, row in matched_products.iterrows():
            name = str(row.get("Product name", "")).strip()
            color = str(row.get("Color", "")).strip()
            lines.append(f"- {name} (Color: {color})")

    lines.append(
        "Style: clean photography, natural lighting, no logos, no text overlays, no random extra objects."
    )

    return "\n".join(lines)


def generate_concept_image(
    mode: str,
    room_type: str,
    user_instruction: str,
) -> Tuple[Image.Image | None, str | None]:
    """
    Call the image model with a very strict prompt.
    Wrapped in try/except so if catalog search ever breaks, it still returns an image or a clear error.
    """
    try:
        try:
            matched = search_catalog(user_instruction or room_type, top_k=6)
        except Exception:
            matched = pd.DataFrame()

        prompt = build_image_prompt(
            mode=mode,
            room_type=room_type,
            user_instruction=user_instruction,
            matched_products=matched,
        )

        resp = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024",
            n=1,
            response_format="b64_json",
        )
        b64_data = resp.data[0].b64_json
        image_bytes = base64.b64decode(b64_data)
        img = Image.open(io.BytesIO(image_bytes))
        return img, None
    except Exception as e:
        return None, str(e)


# --------- HEADER & LOGO ---------

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
st.markdown("[← Return to Market & Place website](https://marketandplace.co/)")

st.markdown("---")

# MAIN TWO-COLUMN LAYOUT
left_col, right_col = st.columns([1.3, 1.2])


# --------- LEFT: ASK STYLIST + CATALOG ---------

with left_col:
    st.markdown("## Ask the AI stylist")

    with st.form("stylist_form", clear_on_submit=False):
        user_query = st.text_input(
            "Describe what you're looking for:",
            placeholder="e.g. luxury towels for a modern bathroom, neutral queen bedding under $80...",
            key="stylist_query",
        )
        submitted = st.form_submit_button("Send")

    if submitted and user_query.strip():
        try:
            recs = search_catalog(user_query, top_k=6)
            render_recommendations(user_query, recs)
        except Exception as e:
            st.error(f"Could not search the catalog: {e}")

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
        c1, c2 = st.columns([1, 3])
        img_url = str(row.get("Image URL:", "")).strip()
        if img_url:
            c1.image(img_url, use_column_width=True)

        name = str(row.get("Product name", "")).strip()
        color = str(row.get("Color", "")).strip()
        price = row.get("Price", "")
        url = str(row.get("raw_amazon", "")).strip()

        c2.markdown(f"**{name}**")
        if color:
            c2.markdown(f"- **Color:** {color}")
        if price != "" and not pd.isna(price):
            c2.markdown(f"- **Price:** {price}")
        if url:
            c2.markdown(f"- [View on Amazon]({url})")


# --------- RIGHT: AI CONCEPT VISUALIZER ---------

with right_col:
    st.markdown("## 🛋️ AI concept visualizer")

    st.write(
        "Generate a styled version of a room or a store shelf using **only** Market & Place textiles as inspiration. "
        "The AI is told not to invent extra furniture or non-catalog products."
    )

    mode_choice = st.radio(
        "What do you want to visualize?",
        options=["In my room photo", "Store shelf / showroom"],
        horizontal=True,
        key="mode_choice",
    )

    room_mode = "room" if mode_choice == "In my room photo" else "store"

    if room_mode == "room":
        room_type = st.selectbox(
            "Room type:",
            options=["bathroom", "bedroom", "living room", "kids' room"],
            index=0,
            key="room_type",
        )
    else:
        room_type = "store / showroom"

    uploaded_file = st.file_uploader(
        "Upload a photo of your room (optional for store shelves):",
        type=["jpg", "jpeg", "png"],
        key="room_image",
    )

    if uploaded_file is not None:
        ref_img = Image.open(uploaded_file)
        st.image(ref_img, caption="Uploaded room (reference)", use_column_width=True)
    else:
        ref_img = None

    image_instruction = st.text_input(
        "What would you like to visualize?",
        placeholder="e.g. navy and white striped towels on the rack",
        key="image_instruction",
    )

    generate_clicked = st.button("Generate concept image", key="generate_image")

    if generate_clicked:
        if room_mode == "room" and ref_img is None:
            st.error("Please upload a room image first, or switch to 'Store shelf / showroom'.")
        else:
            with st.spinner("Generating concept image..."):
                img, err = generate_concept_image(
                    mode=room_mode,
                    room_type=room_type,
                    user_instruction=image_instruction,
                )
            if err:
                st.error(f"Image generation failed: {err}")
            elif img is not None:
                st.image(img, caption="AI-generated style concept", use_column_width=True)
            else:
                st.error("Image generation failed for an unknown reason.")













