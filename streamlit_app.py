import base64
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image
from openai import OpenAI

# ---------- CONFIG ----------
st.set_page_config(
    page_title="Market & Place AI Stylist",
    layout="wide",
)

client = OpenAI()


# ---------- DATA ----------
@st.cache_data(show_spinner=False)
def load_catalog():
    # Make sure this filename matches your Excel file in the repo
    df = pd.read_excel("market_and_place_products.xlsx")
    # Normalise column names just once
    df.columns = [str(c).strip() for c in df.columns]
    return df


df_catalog = load_catalog()

# Convenience helpers for known column names
COL_NAME = "Product name"
COL_COLOR = "Color"
COL_PRICE = "Price"
COL_URL = "raw_amazon"
COL_IMG = "Image URL:"


# ---------- UI: HEADER ----------
header_col1, header_col2, header_col3 = st.columns([1, 2, 1])

with header_col2:
    try:
        st.image(
            "logo.png",
            caption="Market & Place logo",
            use_column_width=True,
        )
    except Exception:
        st.write("")  # fail silently if logo missing

st.markdown(
    "<h1 style='text-align:center; margin-top: 0;'>Market & Place AI Stylist</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='text-align:center;'>Chat with an AI stylist, search the Market & Place "
    "catalog, and generate concept visualizations using your own product file.</p>",
    unsafe_allow_html=True,
)

st.markdown(
    "[← Return to Market & Place website](https://marketandplace.co/)",
)


# ---------- SEARCH / CHAT LOGIC ----------
def smart_filter_for_query(df, query: str) -> pd.DataFrame:
    """
    Simple heuristic: if user asks for towels vs sheets vs bedding,
    we prioritise matching product types.
    """
    q = query.lower()

    df_work = df.copy()

    # Start with all products
    mask = pd.Series(True, index=df_work.index)

    name_series = df_work.get(COL_NAME, pd.Series("", index=df_work.index)).astype(str)

    # Towel vs sheet logic
    if "towel" in q and "sheet" not in q and "bedding" not in q:
        mask &= name_series.str.contains("towel", case=False, na=False)

        # If they did NOT say "beach", prefer non-beach / luxury towels
        if "beach" not in q:
            # keep beach towels only if explicitly luxury-related words missing
            non_beach = ~name_series.str.contains("beach", case=False, na=False)
            luxury_like = name_series.str.contains(
                "luxury|turkish|spa|bath", case=False, na=False
            )
            mask &= (non_beach | luxury_like)

    if any(w in q for w in ["sheet", "sheets", "bedding", "duvet", "comforter", "quilt"]):
        mask &= name_series.str.contains(
            "sheet|bedding|duvet|comforter|quilt|coverlet",
            case=False,
            na=False,
        )

    # Score by keyword matches across multiple columns
    cols_to_search = [
        c
        for c in df_work.columns
        if c
        in [
            COL_NAME,
            COL_COLOR,
            "Category",
            "Subcategory",
            "Keywords",
            "Tags",
            "Description",
        ]
    ]

    score = pd.Series(0, index=df_work.index)
    for c in cols_to_search:
        col_series = df_work[c].astype(str).str.lower()
        score += col_series.str.contains(q, na=False).astype(int)

    df_scored = df_work[mask].copy()
    df_scored["score"] = score[mask]
    df_scored = df_scored[df_scored["score"] > 0].sort_values(
        by="score", ascending=False
    )

    return df_scored


def render_product_list(df_results: pd.DataFrame, title: str):
    if df_results.empty:
        st.write("No matching Market & Place products were found for that request.")
        return

    st.markdown(f"### {title}")
    for idx, row in df_results.iterrows():
        name = str(row.get(COL_NAME, ""))
        color = str(row.get(COL_COLOR, "") or "")
        price = row.get(COL_PRICE, "")
        url = str(row.get(COL_URL, "") or "")
        img_url = str(row.get(COL_IMG, "") or "")

        st.markdown(f"**{name}**")
        if color:
            st.write(f"- Color: {color}")
        if price != "" and pd.notna(price):
            st.write(f"- Price: {price}")
        if url:
            # MUST output raw_amazon EXACTLY
            st.markdown(f"- [View on Amazon]({url})")

        if img_url and img_url.lower().startswith("http"):
            st.image(img_url, width=240)

        st.markdown("---")


def ai_style_message(user_query: str, df_results: pd.DataFrame) -> str:
    """
    Use the Responses API to craft a nice explanation message.
    """
    if df_results.empty:
        return (
            "I couldn't find any matching products in the Market & Place catalog "
            "for that request. Try rephrasing (for example, include the product "
            "type like 'luxury bath towels in grey')."
        )

    # Build a short bullet summary for the model
    examples = []
    for _, row in df_results.head(5).iterrows():
        examples.append(
            f"- {row.get(COL_NAME, '')} — Color: {row.get(COL_COLOR, '')}, "
            f"Price: {row.get(COL_PRICE, '')}"
        )
    catalog_snippet = "\n".join(examples)

    prompt = f"""
You are an AI stylist for Market & Place.

The user asked: "{user_query}".

Here are some matching catalog items:

{catalog_snippet}

Write a short, friendly explanation (2–4 sentences) of why these items could work
for their request. Keep it simple, no markdown headings, no emojis.
"""

    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
        )
        return resp.output[0].content[0].text
    except Exception:
        # Fallback text if API has an issue
        return (
            "Here are some Market & Place products that match your request and "
            "could work well in your space."
        )


# ---------- IMAGE GENERATOR ----------
def decode_and_show_image(b64_data: str, caption: str):
    image_bytes = base64.b64decode(b64_data)
    image = Image.open(BytesIO(image_bytes))
    st.image(image, caption=caption, use_column_width=True)


def build_room_prompt(room_type: str, request: str):
    """
    NOTE: This cannot truly 'edit' the uploaded photo. It builds a strong
    prompt so the new image *resembles* a simple version of that room type
    and only swaps textiles.
    """
    base = f"""Ultra-realistic {room_type} interior concept using ONLY Market & Place textiles.

Rules:
- Keep a simple, believable {room_type} layout (no new furniture shapes, no extra walls).
- Do NOT invent non-textile decor like new furniture pieces, plants, or wall art.
- Focus ONLY on towels, shower curtains, bath mats, bedding, and other Market & Place textiles.
- Do NOT show any product that couldn't plausibly be from Market & Place.
- No people, logos, or brand names.

User request about the textiles: {request}
"""
    return base


def build_store_prompt(request: str):
    base = f"""Ultra-realistic photo of store shelves with ONLY Market & Place products.

Rules:
- Simple store aisle with plain shelving.
- Show neatly folded stacks and/or hanging Market & Place textiles (towels, bedding, etc.).
- No other brands, no random products, no people.
- Background should stay minimal and not distracting.
- Focus on how the products look on the shelves.

User request about the display: {request}
"""
    return base


def generate_concept_image(mode: str, room_type: str, request: str):
    if mode == "Store shelf / showroom":
        prompt = build_store_prompt(request)
    else:
        prompt = build_room_prompt(room_type, request)

    try:
        resp = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024",
            n=1,
        )
        b64_data = resp.data[0].b64_json
        decode_and_show_image(b64_data, "AI-generated style concept")
    except Exception as e:
        st.error(f"Image generation failed: {e}")


# ---------- LAYOUT: LEFT (CHAT + CATALOG) / RIGHT (VISUALIZER) ----------
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.markdown("## Ask the AI stylist")

    # Form so hitting Enter submits
    with st.form(key="stylist_form"):
        user_query = st.text_input(
            "Describe what you're looking for:",
            placeholder="e.g. luxury bath towels in grey, or neutral queen bedding under $80",
        )
        send = st.form_submit_button("Send")

    if send and user_query.strip():
        df_results = smart_filter_for_query(df_catalog, user_query.strip())
        explanation = ai_style_message(user_query.strip(), df_results)

        # Clear previous response simply by re-rendering this block
        st.markdown(f"### 🧵 {user_query.strip()}")
        st.write(explanation)
        st.write("")
        render_product_list(df_results, "Suggested Market & Place products")

    st.markdown("---")
    st.markdown("## Quick catalog peek")

    search_term = st.text_input(
        "Filter products by keyword:",
        key="quick_filter",
        placeholder="e.g. cabana stripe, flannel sheet, navy",
    )

    if search_term.strip():
        peek_results = smart_filter_for_query(df_catalog, search_term.strip())
        if not peek_results.empty:
            for _, row in peek_results.head(10).iterrows():
                st.markdown(f"- **{row.get(COL_NAME, '')}** — {row.get(COL_COLOR, '')}")
        else:
            st.write("No matches in the catalog for that keyword.")
    else:
        st.write("Type a keyword above to quickly peek at the catalog.")


with right_col:
    st.markdown("## AI concept visualizer")
    st.caption(
        "Note: this generates a **concept image** based on your request. "
        "It cannot perfectly edit or overlay your exact photo, "
        "but it will try to keep the layout simple and only change textiles."
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

        uploaded_file = st.file_uploader(
            "Upload a reference photo of your room (used as inspiration only):",
            type=["jpg", "jpeg", "png"],
        )

        request_text = st.text_input(
            "What would you like to visualize?",
            placeholder="e.g. add luxury white towels and a neutral bath mat",
            key="room_request",
        )

        if uploaded_file:
            # Just show the reference; not used directly by the image API
            try:
                img = Image.open(uploaded_file)
                st.image(img, caption="Uploaded room (reference)", use_column_width=True)
            except Exception:
                st.write("Couldn't preview this image, but it will still be used as reference.")

        if st.button("Generate concept image", key="room_generate"):
            if not uploaded_file:
                st.error("Please upload a room image first, or switch to 'Store shelf / showroom'.")
            elif not request_text.strip():
                st.error("Tell the AI what you want to change in the room (textiles, colors, etc.).")
            else:
                generate_concept_image("room", room_type, request_text.strip())

    else:  # Store shelf / showroom
        request_text = st.text_input(
            "What would you like to visualize on the shelves?",
            placeholder="e.g. cabana stripe beach towels in navy and white",
            key="store_request",
        )

        if st.button("Generate concept image", key="store_generate"):
            if not request_text.strip():
                st.error("Describe what you want to see on the shelves.")
            else:
                generate_concept_image("Store shelf / showroom", "store", request_text.strip())












