import base64
import io
import os

import pandas as pd
import streamlit as st
from openai import OpenAI

# ---------- CONFIG ----------

st.set_page_config(
    page_title="Market & Place AI Stylist",
    layout="wide",
)

OPENAI_MODEL_VISION = "gpt-4.1-mini"
OPENAI_MODEL_IMAGE = "gpt-image-1"

client = OpenAI()

# ---------- DATA LOADING & CATEGORY LOGIC (OPTION C) ----------


@st.cache_data
def load_catalog(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    # Normalise column names a bit
    df.columns = [c.strip() for c in df.columns]

    # Some expected columns
    # "Product name", "Color", "Price", "raw_amazon", "Image URL:"
    if "Product name" not in df.columns:
        raise ValueError("Expected a 'Product name' column in the Excel file.")

    def infer_category(name: str) -> str:
        n = str(name).lower()
        if "beach towel" in n:
            return "beach_towel"
        if "towel" in n:
            return "towel"
        if any(
            kw in n
            for kw in [
                "sheet",
                "sheets",
                "bedding",
                "duvet",
                "comforter",
                "quilt",
                "coverlet",
                "bed set",
            ]
        ):
            return "sheet"
        return "other"

    df["category"] = df["Product name"].apply(infer_category)
    return df


def detect_product_family(query: str) -> str | None:
    """Figure out what kind of thing the user is asking for."""
    q = query.lower()

    towel_keywords = [
        "towel",
        "towels",
        "bath towel",
        "bath towels",
        "hand towel",
        "hand towels",
        "luxury towel",
        "luxury towels",
        "spa towel",
        "spa towels",
    ]
    if any(k in q for k in towel_keywords):
        if "beach" in q:
            return "beach_towel"
        return "towel"

    sheet_keywords = [
        "sheet",
        "sheets",
        "bed sheet",
        "bed sheets",
        "bedding",
        "bed set",
        "comforter",
        "duvet",
        "quilt",
        "coverlet",
    ]
    if any(k in q for k in sheet_keywords):
        return "sheet"

    return None


def get_matching_products(user_query: str, df: pd.DataFrame, max_results: int = 4):
    """
    Use BOTH:
      - what the user asked for (towel vs sheet, beach vs luxury)
      - the inferred product 'category' from the catalog
    to choose good matches.

    This is the 'option C' logic.
    """
    family = detect_product_family(user_query)

    if family == "beach_towel":
        candidates = df[df["category"] == "beach_towel"]
    elif family == "towel":
        candidates = df[df["category"] == "towel"]
    elif family == "sheet":
        candidates = df[df["category"] == "sheet"]
    else:
        # If we can't tell, keep everything and just do a soft search later
        candidates = df.copy()

    # If they mention "luxury", favour products with "luxury" in the name
    if "luxury" in user_query.lower():
        lux_mask = candidates["Product name"].str.contains(
            "luxury", case=False, na=False
        )
        if lux_mask.any():
            candidates = candidates[lux_mask]

    # Very simple fuzzy-ish filter: keep rows whose name contains any
    # non-stopword from the query. If nothing matches, just keep the candidates.
    words = [w for w in user_query.lower().split() if len(w) > 3]
    if words:
        mask = pd.Series([False] * len(candidates))
        for w in words:
            mask = mask | candidates["Product name"].str.contains(
                w, case=False, na=False
            )
        if mask.any():
            candidates = candidates[mask]

    return candidates.head(max_results)


# ---------- UTILS ----------


def encode_image_to_data_url(uploaded_file) -> str:
    """Turn a Streamlit uploaded file into a data URL."""
    bytes_data = uploaded_file.getvalue()
    b64 = base64.b64encode(bytes_data).decode("utf-8")
    # Let the model assume it's jpeg; for our purposes type doesn't matter much
    return f"data:image/jpeg;base64,{b64}"


def describe_room_with_vision(uploaded_file, room_type: str, user_prompt: str) -> str:
    """
    Use GPT-4.1-mini with vision to get a short description of the uploaded room.
    We feed this text into the image generator as 'reference layout'.
    """
    data_url = encode_image_to_data_url(uploaded_file)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "You are helping describe a reference photo for an AI stylist. "
                        f"The room type is: {room_type}. "
                        "Describe the layout, main fixtures (vanity, shelves, shower, bed, etc.), "
                        "and where the textiles (towels, bedding, rugs, curtains) are located. "
                        "Keep it under 120 words."
                    ),
                },
                {
                    "type": "input_image",
                    "image_url": {"url": data_url},
                },
            ],
        }
    ]

    resp = client.chat.completions.create(
        model=OPENAI_MODEL_VISION,
        messages=messages,
        max_tokens=180,
    )
    return resp.choices[0].message.content.strip()


def build_product_summary_for_prompt(products_df: pd.DataFrame) -> str:
    """
    Turn a small slice of the catalog into a one-paragraph summary for the image prompt.
    """
    if products_df is None or len(products_df) == 0:
        return (
            "Focus on Market & Place style textiles in realistic colors and patterns "
            "(stripes, solids, and subtle textures)."
        )

    snippets = []
    for _, row in products_df.iterrows():
        name = str(row.get("Product name", "")).strip()
        color = str(row.get("Color", "")).strip()
        piece = name
        if color:
            piece += f" in color {color}"
        snippets.append(piece)

    joined = "; ".join(snippets[:6])
    return (
        "Use textiles clearly inspired by these Market & Place products: "
        f"{joined}. Do not invent brands or collections."
    )


def generate_concept_image(
    mode: str,
    room_type: str,
    uploaded_file,
    user_visual_prompt: str,
    matching_products: pd.DataFrame,
):
    """
    mode: "room" (use uploaded room photo as reference) or "shelf"
    room_type: 'bathroom', 'bedroom', 'store / showroom', etc.
    uploaded_file: Streamlit UploadedFile or None
    user_visual_prompt: text user entered about what they'd like to see
    matching_products: slice of df for context
    """
    product_summary = build_product_summary_for_prompt(matching_products)

    if mode == "room":
        if uploaded_file is None:
            st.error("Please upload a room image first.")
            return None

        room_description = describe_room_with_vision(
            uploaded_file, room_type, user_visual_prompt
        )

        prompt = f"""
You generate concept images for Market & Place, a textile brand.

REFERENCE ROOM DESCRIPTION (from the uploaded photo):
{room_description}

ROOM TYPE: {room_type}

WHAT THE USER WANTS:
{user_visual_prompt or "Style the textiles in this room using Market & Place products."}

STRICT RULES:
- Treat the reference description as the true layout of the room.
- Keep the same room type (do NOT turn a bathroom into a bedroom or vice versa).
- Keep the same permanent structure and fixtures: walls, windows, doors, mirrors, vanity, shelving, shower, tub, toilet, bed frame, nightstands, lamps, etc.
- Do not change the camera angle or perspective.
- Do not add or remove large furniture or architectural elements.
- You may ONLY change removable textiles:
  - bathroom: towels, bath mats, shower curtain, bath rug;
  - bedroom: sheets, duvet/comforter, pillows, shams, throw blanket, curtains;
  - store/showroom: towels or bedding displayed on the shelves.
- Do not invent new decorative items (no extra plants, art, or props).
- Textiles must be consistent with Market & Place products and color stories.

TEXTILE GUIDELINES:
{product_summary}

Generate a single, photo-realistic concept image that clearly follows these rules and uses Market & Place style textiles in the existing layout.
        """.strip()

    else:  # mode == "shelf"
        prompt = f"""
Create a photo-realistic image of a retail shelf in a store or showroom displaying Market & Place textiles.

STRICT RULES:
- Show one or more shelving bays, like in a big-box store or well-organized showroom.
- The main focus must be stacks or folds of towels and/or bedding on shelves.
- Do NOT add other random products (no food, clothes, electronics, furniture, etc.).
- The background should remain a simple store or showroom interior.
- Avoid adding bathtubs, sofas, beds or extra furniture; just shelving and textiles.

WHAT THE USER WANTS:
{user_visual_prompt or "Show how Market & Place textiles would look neatly arranged on store shelves."}

TEXTILE GUIDELINES:
{product_summary}

Generate a single, photo-realistic image.
        """.strip()

    with st.spinner("Generating concept image..."):
        img_resp = client.images.generate(
            model=OPENAI_MODEL_IMAGE,
            prompt=prompt,
            size="1024x1024",
        )

    b64 = img_resp.data[0].b64_json
    return b64


# ---------- SESSION STATE ----------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- LAYOUT: HEADER ----------

logo_cols = st.columns([1, 2, 1])
with logo_cols[1]:
    try:
        st.image("logo.png", use_column_width=True)
    except Exception:
        # If logo fails, just show text title
        st.markdown(
            "<h1 style='text-align:center;'>Market & Place</h1>",
            unsafe_allow_html=True,
        )

st.markdown(
    "<h1 style='text-align:center; margin-top: 0;'>Market & Place AI Stylist</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    """
<p style='text-align:center;'>
Chat with an AI stylist, search the Market & Place catalog, and generate concept visualizations using your own product file.
</p>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "[← Return to Market & Place website](https://marketandplace.co/)",
)

st.markdown("---")

# ---------- MAIN LAYOUT: LEFT (CHAT) / RIGHT (IMAGE) ----------

left_col, right_col = st.columns([1.1, 1])

# ---------- LEFT: ASK THE AI STYLIST ----------

with left_col:
    st.subheader("Ask the AI stylist")

    catalog_df = load_catalog("market_and_place_products.xlsx")

    with st.form("chat_form", clear_on_submit=False):
        user_query = st.text_input(
            "Describe what you're looking for:",
            placeholder="e.g. luxury bath towels for a spa-like bathroom",
            key="user_query_input",
        )
        submitted = st.form_submit_button("Send")

    if submitted and user_query.strip():
        matching = get_matching_products(user_query, catalog_df, max_results=6)

        # Build nice markdown block for this response
        if len(matching) == 0:
            answer_md = (
                "I couldn't find matching products in the Market & Place file for this request, "
                "but the AI stylist will still try to help with general ideas."
            )
        else:
            # Choose an emoji based on family
            fam = detect_product_family(user_query)
            if fam in ("towel", "beach_towel"):
                title_emoji = "🧵"
            elif fam == "sheet":
                title_emoji = "🛏️"
            else:
                title_emoji = "🛍️"

            title = f"{title_emoji} {user_query.strip().lower()}"
            lines = [f"### {title}\n"]
            lines.append(
                "Here are some Market & Place products that match your request and could work well in your space:\n"
            )

            for i, (_, row) in enumerate(matching.iterrows(), start=1):
                name = str(row.get("Product name", "")).strip()
                color = str(row.get("Color", "")).strip()
                price = row.get("Price", "")
                link = str(row.get("raw_amazon", "")).strip()
                img_url = str(row.get("Image URL:", "")).strip()

                lines.append(f"**{i}. {name}**  ")
                if color:
                    lines.append(f"- Color: {color}  ")
                if price != "" and not pd.isna(price):
                    lines.append(f"- Price: {price}  ")
                if link:
                    lines.append(f"- [View on Amazon]({link})  ")
                if img_url and img_url.lower().startswith("http"):
                    # Show image under the bullet points
                    lines.append(f"\n<img src='{img_url}' width='260'>\n")

                lines.append("")  # blank line between products

            answer_md = "\n".join(lines)

        st.session_state.chat_history.insert(
            0, {"query": user_query.strip(), "answer_md": answer_md}
        )

    # Render chat history (newest first)
    for item in st.session_state.chat_history:
        st.markdown("---")
        st.markdown(f"**🧵 {item['query']}**")
        st.markdown(item["answer_md"], unsafe_allow_html=True)

# ---------- RIGHT: IMAGE VISUALIZER ----------

with right_col:
    st.subheader("🖼️ Your image")

    st.write(
        "Upload a photo of your room (bathroom, bedroom, etc.) or, if you're a distributor, "
        "leave it empty and generate a store-shelf / showroom view."
    )

    # Visualization mode
    mode = st.radio(
        "What do you want to visualize?",
        options=["In my room photo", "Store shelf / showroom"],
        index=0,
        horizontal=False,
    )
    mode_key = "room" if mode == "In my room photo" else "shelf"

    # Room type
    room_type = st.selectbox(
        "Room type:",
        options=["bathroom", "bedroom", "living room", "store / showroom", "other"],
        index=0,
    )

    # Upload
    uploaded_room = st.file_uploader(
        "Upload a reference photo of your room (or store aisle):",
        type=["jpg", "jpeg", "png"],
        key="room_uploader",
    )

    if uploaded_room is not None:
        st.image(uploaded_room, caption="Uploaded room (reference)", use_column_width=True)

    # What user wants changed
    user_visual_prompt = st.text_area(
        "What would you like to visualize?",
        placeholder=(
            "e.g. Replace all orange towels with luxury white towels and grey bath mats, "
            "or show a shelf with the blue and aqua beach towels stacked by color."
        ),
        height=100,
    )

    # Optionally reuse last matching products from chat for image styling context
    if st.session_state.chat_history:
        # Use products from the most recent answer, if we still have them
        # (We don't have the df slice stored, so re-run matching based on last query)
        last_query = st.session_state.chat_history[0]["query"]
        image_products = get_matching_products(last_query, catalog_df, max_results=6)
    else:
        image_products = catalog_df.head(6)

    if st.button("Generate concept image"):
        if mode_key == "room" and uploaded_room is None:
            st.error("Please upload a room image first, or switch to 'Store shelf / showroom'.")
        else:
            try:
                b64_img = generate_concept_image(
                    mode=mode_key,
                    room_type=room_type,
                    uploaded_file=uploaded_room,
                    user_visual_prompt=user_visual_prompt,
                    matching_products=image_products,
                )
                if b64_img:
                    img_bytes = base64.b64decode(b64_img)
                    st.image(io.BytesIO(img_bytes), caption="AI-generated style concept", use_column_width=True)
            except Exception as e:
                st.error("There was a problem generating the image. Please try again.")
                st.error(str(e))










