# streamlit_app.py

import io
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import streamlit as st
from openai import OpenAI

# ---------- CONFIG ----------

st.set_page_config(
    page_title="Market & Place AI Stylist",
    layout="wide",
)

client = OpenAI()  # uses OPENAI_API_KEY from env / Streamlit secrets

CATALOG_PATH = "market_and_place_products.xlsx"
LOGO_PATH = "logo.png"
MARKET_AND_PLACE_URL = "https://marketandplace.co/"

# ---------- DATA LOADING ----------


@st.cache_data
def load_catalog() -> pd.DataFrame:
    df = pd.read_excel(CATALOG_PATH)
    df.columns = [c.strip() for c in df.columns]
    # Normalized helper columns
    df["name_lower"] = df["Product name"].str.lower()
    return df


catalog = load_catalog()


# ---------- SMALL HELPERS ----------


def ensure_session_defaults() -> None:
    """Initialize session_state keys safely (no module-level writes)."""
    st.session_state.setdefault("chat_history", [])  # list[dict]
    st.session_state.setdefault("last_products", [])  # for image prompt
    st.session_state.setdefault("room_type", "bathroom")


ensure_session_defaults()


def find_products_from_query(query: str, max_results: int = 8) -> pd.DataFrame:
    """
    Very simple keyword matcher, with special handling so
    asking for 'towels' actually returns towel SKUs.
    """
    q = (query or "").lower()

    # Nuclear rules: explicit category routing first
    if any(w in q for w in ["towel", "towels"]):
        mask = catalog["name_lower"].str.contains("towel", na=False)
        results = catalog[mask]
    elif any(w in q for w in ["sheet", "sheets", "bedding", "bed set", "sheet set"]):
        mask = catalog["name_lower"].str.contains("sheet|bedding|bed set", na=False)
        results = catalog[mask]
    elif any(w in q for w in ["quilt", "comforter", "coverlet"]):
        mask = catalog["name_lower"].str.contains(
            "quilt|comforter|coverlet", na=False
        )
        results = catalog[mask]
    else:
        # Generic keyword OR search
        tokens = [t for t in q.replace(",", " ").split() if len(t) > 2]
        if not tokens:
            return catalog.head(max_results)

        mask = False
        for t in tokens:
            mask = mask | catalog["name_lower"].str.contains(t, na=False)
        results = catalog[mask]

    if results.empty:
        # Last resort: just show a few highlight products
        return catalog.head(max_results)

    return results.head(max_results)


def format_product_for_text(row: pd.Series, idx: int) -> str:
    name = row.get("Product name", "Unnamed product")
    color = row.get("Color", "N/A")
    price = row.get("Price", "N/A")
    url = row.get("raw_amazon", "")
    line = (
        f"{idx}. **{name}**  \n"
        f"   • Color: **{color}**  \n"
        f"   • Price: **{price}**"
    )
    if isinstance(url, str) and url.strip():
        line += f"  \n   • [View on Amazon]({url.strip()})"
    return line


def render_product_card(row: pd.Series) -> None:
    img_url = row.get("Image URL:", None)
    name = row.get("Product name", "Unnamed product")
    color = row.get("Color", "N/A")
    price = row.get("Price", "N/A")
    url = row.get("raw_amazon", "")

    col_img, col_txt = st.columns([1, 2])
    with col_img:
        if isinstance(img_url, str) and img_url.strip():
            st.image(img_url, use_column_width=True)
        else:
            st.empty()
    with col_txt:
        st.markdown(f"**{name}**")
        st.markdown(f"- Color: **{color}**")
        st.markdown(f"- Price: **{price}**")
        if isinstance(url, str) and url.strip():
            st.markdown(f"[View on Amazon]({url.strip()})")


def build_image_prompt(
    room_type: str,
    visualization_mode: str,
    description: str,
    example_products: List[pd.Series],
) -> str:
    """
    We CANNOT literally force the model to only use catalog images
    or perfectly keep the layout – that’s a real limitation of
    generative image models.

    But we can give it an extremely strong "nuclear" instruction.
    """
    short_examples = []
    for p in example_products[:4]:
        short_examples.append(
            f"{p.get('Product name','Unnamed')} (color {p.get('Color','N/A')})"
        )
    examples_txt = "; ".join(short_examples) or "Market & Place towels, sheets and quilts"

    base_desc = description.strip() or "a clean, modern, realistic styling"

    mode_clause = (
        "Show these textiles installed in a REALISTIC retail store shelf / aisle, "
        "with stacks of folded product on shelves, packaging subtle and not branded."
        if visualization_mode == "Store shelf / showroom"
        else f"Show the *same* {room_type} layout as the input photo."
    )

    prompt = f"""
NUCLEAR SAFETY RULES – YOU MUST OBEY ALL:

1. Room type is: **{room_type}**. The room type MUST NOT change.
   - If the user uploads a bathroom, the output MUST still be a bathroom.
   - If it's a bedroom, it MUST still be a bedroom.
   - Never turn bathrooms into bedrooms or add bathtubs where none exist.

2. Layout, architecture and hard fixtures MUST stay the same:
   - Keep walls, windows, doors, floors, mirrors, sinks, toilets, showers,
     shelving, and cabinets in the same positions and shapes.
   - You are allowed to change ONLY soft textiles (shower curtain,
     towels, bath mat, bedding, decorative pillows, etc.).
   - DO NOT move or remove furniture, plumbing, cabinetry, or mirrors.

3. Textiles must be INSPIRED ONLY by Market & Place products:
   - Use colors, stripes, and patterns inspired by these examples:
     {examples_txt}
   - DO NOT invent random designs that look unrelated.
   - DO NOT show other brands or logos.

4. Absolutely FORBIDDEN:
   - Changing the room type.
   - Adding non-textile objects that change the architecture.
   - Adding characters, people, or pets.
   - Using any brand that is not Market & Place inspired.

Now create a high-resolution, photorealistic image.

Scene instructions:
- {mode_clause}
- Apply Market & Place textiles that fit this description from the user:
  "{base_desc}"
- Lighting should feel natural and inviting.
"""
    return prompt


# ---------- UI: HEADER & LAYOUT ----------

# Centered logo
logo_cols = st.columns([1, 2, 1])
with logo_cols[1]:
    logo_file = Path(LOGO_PATH)
    if logo_file.exists():
        st.image(str(logo_file), use_column_width=True)
    else:
        st.write("")  # no logo available, avoid broken icon

st.markdown(
    "<h1 style='text-align:center; margin-top: 0.5rem;'>Market &amp; Place AI Stylist</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='text-align:center;'>Chat with an AI stylist, search the Market &amp; Place "
    "catalog, and generate concept visualizations using your own product file.</p>",
    unsafe_allow_html=True,
)

# Return link
st.markdown(
    f"[← Return to Market & Place website]({MARKET_AND_PLACE_URL})",
)

st.write("")  # small spacer

# Two main columns: left = chat & catalog, right = image visualizer
left_col, right_col = st.columns([1.1, 1])

# ---------- LEFT: CHAT WITH AI STYLIST + CATALOG PEEK ----------

with left_col:
    st.subheader("Ask the AI stylist")

    # Keep the search bar pinned here by using a form
    with st.form("query_form", clear_on_submit=False):
        user_query = st.text_input(
            "Describe what you're looking for:",
            placeholder="e.g. neutral queen bedding under $80 for a bright room",
            key="query_input",
        )
        submitted = st.form_submit_button("Send")

    if submitted and user_query.strip():
        # Fetch products
        matches = find_products_from_query(user_query, max_results=8)
        st.session_state["last_products"] = matches.to_dict(orient="records")

        # Build textual reply
        if matches.empty:
            reply_text = (
                "I couldn't find any matching Market & Place products in the catalog "
                "for that request. Try adjusting the description or keywords."
            )
        else:
            intro = (
                "Here are some Market & Place products that match your request "
                "and could work well in your space:\n\n"
            )
            lines = [
                format_product_for_text(row, i + 1)
                for i, (_, row) in enumerate(matches.iterrows())
            ]
            reply_text = intro + "\n\n".join(lines)

        # Store in chat history
        st.session_state.chat_history.insert(
            0,
            {
                "query": user_query.strip(),
                "response": reply_text,
            },
        )

    # Render chat history (newest first), but the input stays at the top
    for message in st.session_state.chat_history:
        st.markdown(
            f"""<div style="padding:0.5rem 0;"><strong>🧵 {message['query']}</strong></div>""",
            unsafe_allow_html=True,
        )
        st.markdown(message["response"])
        st.markdown("---")

    # Quick catalog peek that moves down with chat
    st.subheader("Quick catalog peek")
    catalog_filter = st.text_input(
        "Filter products by keyword",
        placeholder="e.g. towel, flannel, wildlife, taupe",
        key="catalog_filter",
    )

    if catalog_filter.strip():
        peek_matches = find_products_from_query(catalog_filter, max_results=6)
    else:
        peek_matches = catalog.head(6)

    for _, row in peek_matches.iterrows():
        render_product_card(row)
        st.markdown("---")


# ---------- RIGHT: IMAGE VISUALIZER ----------

with right_col:
    st.subheader("🖼️ Your image")

    st.write(
        "Upload a photo of your room (bathroom, bedroom, etc.) or, if you're "
        "a distributor, leave it empty and generate a store shelf / showroom view."
    )

    room_type = st.selectbox(
        "Room type (for nuclear rules):",
        ["bathroom", "bedroom", "living room", "store / showroom"],
        index=["bathroom", "bedroom", "living room", "store / showroom"].index(
            st.session_state.get("room_type", "bathroom")
        ),
        help="This MUST match the real type of the uploaded photo.",
    )
    st.session_state["room_type"] = room_type

    visualization_mode = st.radio(
        "What do you want to visualize?",
        [
            "In my room photo",
            "Store shelf / showroom",
        ],
        help="Distributors can pick 'Store shelf / showroom' to see product on shelves.",
    )

    uploaded_file = st.file_uploader(
        "Drag and drop file here",
        type=["jpg", "jpeg", "png"],
        help="Optional. If empty and you choose 'Store shelf / showroom', a generic store scene is generated.",
    )

    image_description = st.text_area(
        "Describe what you’d like to see (colors, patterns, mood, etc.):",
        placeholder=(
            "e.g. Use cabana-stripe towels in navy and white with a coastal feel, "
            "but keep the rest of the bathroom exactly the same."
        ),
    )

    if st.button("Generate concept image", type="primary"):
        # Use the last product suggestions as inspiration for the image
        last_products_records = st.session_state.get("last_products", [])
        last_products = [pd.Series(r) for r in last_products_records]

        prompt = build_image_prompt(
            room_type=room_type,
            visualization_mode=visualization_mode,
            description=image_description,
            example_products=last_products or list(catalog.head(6).apply(lambda r: r)),
        )

        try:
            # NOTE: Streamlit Cloud currently only supports plain image generation.
            # Actual image *editing* APIs may not be available, so we treat the
            # uploaded photo as conceptual reference and do not pass it to the API.
            #
            # This avoids the previous "Images object has no attribute 'edits'" error.
            result = client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                size="1024x1024",
            )

            b64 = result.data[0].b64_json
            import base64

            img_bytes = base64.b64decode(b64)
            st.image(io.BytesIO(img_bytes), caption="AI-generated style concept")

            st.info(
                "Note: This is a generative concept image. The model is strongly "
                " instructed to keep the same room type and only change textiles "
                "using Market & Place–inspired products, but it cannot perfectly "
                "copy your exact room photo or catalog photos."
            )

        except Exception as e:
            st.error(
                "Could not generate a concept image. The image API on this environment "
                "may not support advanced edits yet.\n\n"
                f"Technical details: {e}"
            )












