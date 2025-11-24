import os
import io
import base64

import streamlit as st
import pandas as pd
from PIL import Image
from openai import OpenAI


# ================== PAGE CONFIG ==================

st.set_page_config(
    page_title="Market & Place AI Stylist",
    layout="wide",
)

# Centered logo + title
LOGO_PATH = "logo.png"

logo_cols = st.columns([1, 3, 1])
with logo_cols[1]:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, caption=None, use_column_width=True)
    else:
        st.write("**Market & Place**")  # fallback

st.markdown(
    "<h1 style='text-align: center; margin-top: 0.5rem;'>Market & Place AI Stylist</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center;'>Chat with an AI stylist, search the Market & Place catalog, "
    "and generate concept visualizations using your own product file.</p>",
    unsafe_allow_html=True,
)

# Return link to main website
st.markdown(
    "[← Return to Market & Place website](https://marketandplace.co/)",
)

st.markdown("---")


# ================== OPENAI CLIENT ==================

def get_openai_client() -> OpenAI:
    """Get OpenAI client using OPENAI_API_KEY from env or Streamlit secrets."""
    api_key = os.environ.get("OPENAI_API_KEY")

    # try Streamlit secrets if env var not set
    try:
        if not api_key and "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

    if not api_key:
        st.error(
            "❌ OPENAI_API_KEY not found.\n\n"
            "Go to Streamlit → **⋯ → Settings → Secrets** and add:\n\n"
            '`OPENAI_API_KEY = "sk-...."`'
        )
        st.stop()

    return OpenAI(api_key=api_key)


client = get_openai_client()


# ================== DATA LOADING ==================

@st.cache_data
def load_data() -> pd.DataFrame:
    """Load Market & Place catalog from Excel in the repo."""
    df = pd.read_excel("market_and_place_products.xlsx")

    # Normalise column names just in case
    df.columns = [c.strip() for c in df.columns]
    return df


try:
    df = load_data()
except Exception as e:
    st.error(
        "❌ Could not load `market_and_place_products.xlsx`.\n\n"
        "Make sure the file is in the repo root and has these columns:\n"
        "`Product name`, `Color`, `Price`, `raw_amazon`, `Image URL:`.\n\n"
        f"Technical details: {e}"
    )
    st.stop()

DESC_COL = None  # we don't currently have a description column


# ================== PRODUCT HELPERS ==================

def find_relevant_products(query: str, max_results: int = 6) -> pd.DataFrame:
    """Simple keyword search over product name and color."""
    if not query:
        return df.head(max_results)

    q = query.lower()

    mask = (
        df["Product name"].fillna("").str.lower().str.contains(q)
        | df["Color"].fillna("").str.lower().str.contains(q)
    )

    results = df[mask].copy()

    # fallback so AI always has something
    if results.empty:
        results = df.sample(min(max_results, len(df)), random_state=0)

    return results.head(max_results)


def format_products_for_prompt(products: pd.DataFrame) -> str:
    """Compact text block describing products for the model (for chat + image prompts)."""
    lines = []
    for _, row in products.iterrows():
        line = (
            f"- Name: {row.get('Product name', '')}\n"
            f"  Color: {row.get('Color', '')}\n"
            f"  Price: {row.get('Price', '')}\n"
            f"  Amazon URL: {row.get('raw_amazon', '')}"
        )
        lines.append(line)
    return "\n\n".join(lines)


# ================== OPENAI CALLS (CHAT) ==================

def call_stylist_model(user_message: str, room_context: str,
                       products: pd.DataFrame) -> str:
    """
    Ask GPT-4.1-mini for styling suggestions based on the catalog.

    HARD / NUCLEAR RULES:
    - Only use the provided catalog.
    - Never invent product names, prices, or URLs.
    - Bathroom stays bathroom, bedroom stays bedroom, etc.
    """
    product_block = format_products_for_prompt(products)
    room_part = room_context or "The user did not describe the room."

    system_prompt = """
You are an interior stylist working for the home-textile brand **Market & Place**.

***NUCLEAR RULES – VIOLATING THESE IS NOT ALLOWED:***
1. You ONLY recommend products from the product list provided in this message.
2. You MUST NOT invent, hallucinate, or rename any products, colors, prices, or URLs.
3. You MUST copy the Amazon URL text EXACTLY as given for each product, with no changes.
4. When you refer to the room, respect the type of space the user shows or describes:
   - A bathroom stays a bathroom (no beds, sofas, or random furniture).
   - A bedroom stays a bedroom (no toilets or sinks).
5. You never claim that the AI image generator can perfectly match reality. You refer to it as an approximate concept render only.

For each suggested product, you MUST clearly output:
- Product name
- Color
- Price
- Short explanation of why it fits
- Amazon URL (copy EXACTLY from the data, do not change, trim, or add parameters)

Group recommendations into a short numbered list (3–5 items).
Do NOT invent products or URLs that are not in the list.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"User request:\n{user_message}\n\n"
                f"Room description:\n{room_part}\n\n"
                "Here is the ONLY catalog you may use:\n\n"
                f"{product_block}"
            ),
        },
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.2,  # low temp to reduce hallucinations
            max_tokens=800,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error calling OpenAI chat model: `{e}`"


# ================== OPENAI CALLS (IMAGES) ==================

def build_image_prompt(
    room_description: str,
    products: pd.DataFrame,
    mode: str,
) -> str:
    """
    Build a VERY strict text prompt for the image generator.

    NOTE: images.generate() cannot literally edit the uploaded pixels or guarantee
    exact Market & Place SKUs. This prompt just pushes it as hard as possible.
    """
    product_block = format_products_for_prompt(products)

    if not room_description.strip():
        room_text = "The user did not provide extra text. Infer the room type from the uploaded photo."
    else:
        room_text = room_description

    base_rules = f"""
You are generating a **concept visualization** for the textile brand Market & Place.

***ABSOLUTE / NUCLEAR RULES (DO NOT BREAK THESE):***
- You MUST treat the uploaded photo as the base environment.
- You MUST keep the same type of space:
  - If the uploaded photo is a bathroom, the result MUST be a bathroom (no bed, sofa, etc.).
  - If it is a bedroom, the result MUST be a bedroom (no toilet, sink, etc.).
  - If it is a living room, keep it a living room.
- You MUST keep the room architecture and fixtures the same:
  - Keep windows, walls, doors, mirrors, sinks, toilets, showers, tubs, and furniture in the same style and approximate layout.
  - You are NOT allowed to change or move the plumbing fixtures, doors, or major furniture.
- You are ONLY allowed to change **textiles**:
  - Towels, bath mats, shower curtains, bath rugs (for bathrooms).
  - Sheets, quilts, comforters, duvet covers, shams, pillowcases (for bedrooms).
  - Throw pillows, throws, curtains, and rugs (for living areas).
- Any textiles you show must look like they come from this Market & Place catalog:

{product_block}

- You MUST use patterns, colors, and textures that clearly match these catalog items.
- You MUST NOT introduce new made-up products or fantasy patterns that are not consistent with this catalog.
- The result is a photorealistic concept render, not a perfect pixel edit. Never add floating text, logos, or UI elements.
"""

    if mode == "Store shelf view":
        mode_text = """
The user wants a **store shelf / retail aisle** visualization for distributors.

Generate a photorealistic image of store shelves in a clean, modern retail environment.
- Show neatly arranged stacks of towels, bedding, or related textiles that visually match the Market & Place catalog.
- Use packaging, folds, and stacks that a real retailer might see.
- Do NOT show an entire bedroom or bathroom; the focus is on shelves in a store or showroom.
"""
    else:
        mode_text = f"""
The user wants a **styled version of the uploaded room**, using Market & Place textiles only.
Room description from the user (if any):
{room_text}
"""

    return base_rules + "\n" + mode_text


def generate_concept_image(
    room_description: str,
    products: pd.DataFrame,
    mode: str,
) -> Image.Image | None:
    """
    Generate a concept visualization using gpt-image-1.

    IMPORTANT: In this environment we only have access to images.generate(),
    so this is *not* a true image edit. The uploaded image is shown to the user
    for reference, and the prompt strongly tells the model to mimic it.
    """
    prompt = build_image_prompt(room_description, products, mode)

    try:
        img_resp = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024",
        )

        b64_data = img_resp.data[0].b64_json
        image_bytes = base64.b64decode(b64_data)
        img = Image.open(io.BytesIO(image_bytes))
        return img

    except Exception as e:
        st.warning(
            "Could not generate concept image. "
            "This can be due to org verification, billing, or API limits.\n\n"
            f"Details: {e}"
        )
        return None


# ================== STREAMLIT STATE ==================

if "messages" not in st.session_state:
    st.session_state.messages = []   # chat history

if "last_products" not in st.session_state:
    st.session_state.last_products = df.head(4)

if "room_description" not in st.session_state:
    st.session_state.room_description = ""

if "concept_image_bytes" not in st.session_state:
    st.session_state.concept_image_bytes = None

if "viz_mode" not in st.session_state:
    st.session_state.viz_mode = "Style my room"


# ================== LAYOUT ==================

left_col, right_col = st.columns([2, 2])


# ----- RIGHT: IMAGE SIDE FIRST (to keep layout tidy) -----
with right_col:
    st.subheader("🖼️ Your image")

    st.write(
        "Upload a photo of your room (bathroom, bedroom, etc.) or leave empty "
        "if you just want a store shelf visualization for distributors."
    )

    uploaded_image = st.file_uploader(
        "Drag and drop a file here",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded room / reference", use_column_width=True)

    viz_mode = st.radio(
        "Visualization mode",
        ["Style my room", "Store shelf view"],
        index=0 if st.session_state.viz_mode == "Style my room" else 1,
    )
    st.session_state.viz_mode = viz_mode

    room_description = st.text_area(
        "Optional: Describe your room or what you want:",
        value=st.session_state.room_description,
        placeholder=(
            "e.g. Small bathroom with white tiles and wood vanity, want colorful Market & Place towels "
            "and bathmat that match SKU X, Y, Z..."
        ),
    )
    st.session_state.room_description = room_description

    st.markdown(
        "> ⚠️ *Concept images are approximations.* The AI is instructed to keep the same room type "
        "and only change textiles using the Market & Place catalog, but it cannot perfectly copy your photo "
        "or guarantee exact SKUs."
    )

    if st.button("Generate concept image"):
        with st.spinner("Asking OpenAI to generate a concept visualization..."):
            products_for_image = st.session_state.last_products

            img = generate_concept_image(
                st.session_state.room_description,
                products_for_image,
                st.session_state.viz_mode,
            )
            if img is not None:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                st.session_state.concept_image_bytes = buf.getvalue()

    if st.session_state.concept_image_bytes is not None:
        st.image(
            st.session_state.concept_image_bytes,
            caption=f"AI-generated style concept ({st.session_state.viz_mode})",
            use_column_width=True,
        )

    st.markdown("---")
    st.subheader("🔎 Quick catalog peek (Market & Place only)")

    quick_query = st.text_input("Filter products by keyword", key="quick_query")
    if quick_query:
        preview = find_relevant_products(quick_query, max_results=10)
    else:
        preview = df.head(10)

    # product cards with images
    for _, row in preview.iterrows():
        with st.container():
            cols = st.columns([1, 3])
            with cols[0]:
                img_url = row.get("Image URL:", "")
                if isinstance(img_url, str) and img_url.strip():
                    try:
                        st.image(img_url, use_column_width=True)
                    except Exception:
                        st.empty()
                else:
                    st.empty()
            with cols[1]:
                st.markdown(f"**{row.get('Product name', '')}**")
                st.markdown(f"- Color: **{row.get('Color', '')}**")
                st.markdown(f"- Price: **{row.get('Price', '')}**")
                url = row.get("raw_amazon", "")
                if isinstance(url, str) and url.strip():
                    st.markdown(f"[View on Amazon]({url})")

        st.markdown("---")


# ----- LEFT: CHAT + CATALOG LINKED TO CHAT -----
with left_col:
    st.subheader("Ask the AI stylist")

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input(
        "Describe what you're looking for (e.g. 'neutral queen bedding under $80 for a bright room', "
        "'colorful towels for a grey bathroom')."
    )

    if user_input:
        # user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # pick candidate products for this question
        candidate_products = find_relevant_products(user_input, max_results=6)
        st.session_state.last_products = candidate_products.copy()

        # use stored room description from right column just as extra context
        room_desc = st.session_state.room_description

        # AI reply
        reply = call_stylist_model(user_input, room_desc, candidate_products)
        st.session_state.messages.append({"role": "assistant", "content": reply})

        st.rerun()

    # Products tied to the *current conversation* (under chat)
    products = st.session_state.last_products
    if products is not None and not products.empty:
        st.markdown("### Recommended Market & Place products for this conversation")

        for _, row in products.iterrows():
            with st.container():
                cols = st.columns([1, 3])

                with cols[0]:
                    img_url = row.get("Image URL:", "")
                    if isinstance(img_url, str) and img_url.strip():
                        try:
                            st.image(img_url, use_column_width=True)
                        except Exception:
                            st.empty()
                    else:
                        st.empty()

                with cols[1]:
                    st.markdown(f"**{row.get('Product name', '')}**")
                    st.markdown(f"- Color: **{row.get('Color', '')}**")
                    st.markdown(f"- Price: **{row.get('Price', '')}**")

                    if DESC_COL:
                        desc_val = row.get(DESC_COL, "")
                        if isinstance(desc_val, str) and desc_val.strip():
                            st.markdown(f"- Description: {desc_val}")

                    url = row.get("raw_amazon", "")
                    if isinstance(url, str) and url.strip():
                        st.markdown(f"[View on Amazon]({url})")

            st.markdown("---")












