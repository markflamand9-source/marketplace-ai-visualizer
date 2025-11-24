import os
import io
import base64

import streamlit as st
import pandas as pd
from PIL import Image
from openai import OpenAI


# ================== PAGE CONFIG & GLOBAL STYLING ==================

st.set_page_config(
    page_title="Market & Place AI Stylist",
    layout="wide",
    page_icon="🧵",
)

# Global styling: soft background + nicer padding
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f6fa;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ================== HEADER WITH LOGO ==================

# Centered logo (logo.png must be in repo root)
logo_cols = st.columns([1, 2, 1])
with logo_cols[1]:
    st.image("logo.png", use_column_width=False)

st.markdown(
    """
    <h1 style="text-align:center; margin-top:0.5rem; margin-bottom:0.2rem;">
        Market &amp; Place AI Stylist
    </h1>
    <p style="text-align:center; font-size:1.05rem; color:#555;">
        Chat with an AI stylist, search the Market &amp; Place catalog, and generate
        concept visualizations using your own product file.
    </p>
    """,
    unsafe_allow_html=True,
)

# ---- SEARCH / CHAT BAR DIRECTLY UNDER HEADER ----
st.markdown("### Ask the AI stylist")

with st.form("stylist_form"):
    user_input = st.text_input(
        "Describe what you're looking for:",
        placeholder="e.g. neutral queen bedding under $80 for a bright room",
        key="main_query",
    )
    submitted = st.form_submit_button("Send")


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
    return pd.read_excel("market_and_place_products.xlsx")


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

# There is no dedicated description column in this file.
DESC_COL = None


# ================== PRODUCT HELPERS ==================

def find_relevant_products(query: str, max_results: int = 6) -> pd.DataFrame:
    """
    Keyword search over product name and color.

    Special handling:
    - If the user mentions towels (towel, towels, bath towel, bath sheet, etc.),
      first try to return ONLY towel products.
    """
    if not query:
        return df.head(max_results)

    q = query.lower()

    # detect towel intent
    towel_terms_in_query = [
        "towel", "towels", "bath towel", "bath towels",
        "hand towel", "hand towels", "washcloth", "wash cloth",
        "bath sheet", "bath sheets",
    ]
    wants_towel = any(t in q for t in towel_terms_in_query)

    # towel-specific filter
    towel_pattern = "towel|bath sheet|washcloth|wash cloth"

    if wants_towel:
        name_series = df["Product name"].fillna("").str.lower()
        mask_towel = name_series.str.contains(towel_pattern)
        results = df[mask_towel].copy()

        # if we actually have towels, return those first
        if not results.empty:
            return results.head(max_results)

    # generic fallback search (name + color)
    mask_generic = (
        df["Product name"].fillna("").str.lower().str.contains(q)
        | df["Color"].fillna("").str.lower().str.contains(q)
    )
    results = df[mask_generic].copy()

    # fallback so AI always has something
    if results.empty:
        results = df.sample(min(max_results, len(df)), random_state=0)

    return results.head(max_results)


def format_products_for_prompt(products: pd.DataFrame) -> str:
    """Compact text block describing products for the model."""
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


# ================== OPENAI CALLS ==================

def call_stylist_model(user_message: str, room_context: str,
                       products: pd.DataFrame) -> str:
    """Ask GPT-4.1-mini for styling suggestions based on the catalog."""
    product_block = format_products_for_prompt(products)
    room_part = room_context or "The user did not describe the room."

    system_prompt = """
You are an interior stylist working for the home-textile brand **Market & Place**.

You must follow these rules carefully:

- You ONLY recommend products from the product list provided.
- For each suggested product, you MUST clearly output:
  • Product name  
  • Color  
  • Price  
  • Short explanation of why it fits  
  • Amazon URL (copy EXACTLY from the data, do not change, trim, or add parameters)  
- Group recommendations into a short numbered list (3–5 items).
- If the user asks for towels, treat all of these as valid towel terms:
  "towel", "towels", "bath towel", "bath sheet", "hand towel", "washcloth".
- If no exact towel products exist in the provided list, say this clearly and then
  suggest the closest suitable alternatives from the list (but do NOT invent towels).
- Do NOT invent products or URLs that are not in the list.
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
            temperature=0.7,
            max_tokens=800,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error calling OpenAI chat model: `{e}`"


def generate_concept_image(
    room_description: str,
    products: pd.DataFrame,
    room_image_bytes: bytes | None = None,
) -> Image.Image | None:
    """
    Try to generate a concept visualization of the SAME room, styled with
    Market & Place products.

    We FIRST attempt a true image edit (using the uploaded room photo).
    If the API / SDK does not support edits, we fall back to a pure
    concept render and warn the user.
    """

    top_names = ", ".join(products["Product name"].head(4).tolist())
    # ultra-strict "nuclear" instruction
    prompt = (
        "You are doing STRICT PHOTO EDITING for the brand Market & Place.\n"
        "You receive a real photo of a bedroom. Your job is ONLY to restyle "
        "TEXTILES in that exact photo. Follow these rules as NON-NEGOTIABLE HARD "
        "CONSTRAINTS:\n\n"
        "1. The room architecture, camera angle, bed shape, windows, doors, floor, "
        "   wall color, artwork, shelves, decor objects, plants, lamps, nightstands, "
        "   picture frames and all furniture MUST remain IDENTICAL.\n"
        "2. You are NOT allowed to move, resize, add, or remove any furniture or "
        "   decor items. Their position, shape and style must stay the same.\n"
        "3. You are NOT allowed to change the wall color, lighting, perspective, "
        "   or crop. The photo should look like the same camera shot.\n"
        "4. The ONLY things you are allowed to modify are TEXTILES:\n"
        "   - bedding (duvet, quilt, sheets, pillowcases, throws)\n"
        "   - decorative pillows\n"
        "   - blankets/throws\n"
        "   - curtains\n"
        "   - towels\n"
        "5. If a change would affect anything other than textiles, DO NOT make it.\n\n"
        "Restyle ONLY the textiles so they look like these Market & Place products: "
        f"{top_names}.\n"
        "Everything else must be left exactly as in the original photo."
    )

    try:
        if room_image_bytes is not None:
            # Try a true edit of the uploaded room photo
            img_file = io.BytesIO(room_image_bytes)
            img_file.name = "room.png"

            img_resp = client.images.edits(
                model="gpt-image-1",
                image=img_file,
                prompt=prompt,
                size="1024x1024",
            )
        else:
            # No base image: fall back to concept (brand-new room)
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
        # If edits() isn't available in the SDK or fails, we warn and fall back
        st.warning(
            "Could not perform a true photo edit on your room image. "
            "Your OpenAI setup may not support image edits yet, so the result "
            "below might be a brand-new concept room instead of your exact photo.\n\n"
            f"Technical details: {e}"
        )
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
        except Exception as e2:
            st.error(f"Image generation also failed: {e2}")
            return None


# ================== STREAMLIT STATE ==================

if "messages" not in st.session_state:
    st.session_state.messages = []   # chat history

if "last_products" not in st.session_state:
    st.session_state.last_products = df.head(4)

if "room_description" not in st.session_state:
    st.session_state.room_description = ""

if "uploaded_room_image_bytes" not in st.session_state:
    st.session_state.uploaded_room_image_bytes = None

if "concept_image_bytes" not in st.session_state:
    st.session_state.concept_image_bytes = None


# ================== HANDLE TOP SEARCH / CHAT SUBMIT ==================

if submitted and user_input:
    # user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # pick candidate products for this question
    candidate_products = find_relevant_products(user_input, max_results=6)
    st.session_state.last_products = candidate_products.copy()

    # use stored room description from right column
    room_desc = st.session_state.room_description

    # AI reply
    reply = call_stylist_model(user_input, room_desc, candidate_products)
    st.session_state.messages.append({"role": "assistant", "content": reply})

    st.rerun()


# ================== LAYOUT ==================

col_chat, col_side = st.columns([2.2, 1.3])

# ----- LEFT: CHAT + PRODUCT IMAGES -----
with col_chat:
    st.subheader("Chat with the AI stylist")

    products = st.session_state.last_products
    msgs = st.session_state.messages
    rev_msgs = list(reversed(msgs))  # newest first for display

    # Show newest conversation at the top,
    # and put the product cards right under the newest AI answer.
    for i, msg in enumerate(rev_msgs):
        avatar = "🙂" if msg["role"] == "user" else "🧵"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

        # Immediately under the newest AI message, show the product cards
        if i == 0 and msg["role"] == "assistant":
            if products is not None and not products.empty:
                st.markdown("#### Recommended products for this conversation")

                for _, row in products.iterrows():
                    with st.container():
                        cols = st.columns([1, 3])

                        # image
                        with cols[0]:
                            img_url = row.get("Image URL:", "")
                            if isinstance(img_url, str) and img_url.strip():
                                try:
                                    st.image(img_url, use_column_width=True)
                                except Exception:
                                    st.empty()
                            else:
                                st.empty()

                        # info + light description
                        with cols[1]:
                            name = row.get("Product name", "")
                            color = row.get("Color", "")
                            price = row.get("Price", "")

                            st.markdown(f"**{name}**")
                            st.markdown(f"- Color: **{color}**")
                            st.markdown(f"- Price: **{price}**")

                            desc_text = (
                                f"This {str(color).lower() if isinstance(color, str) else ''} "
                                f"{name} helps tie the room together with Market & Place’s "
                                "soft, cozy textile look."
                            )
                            st.markdown(f"- Description: {desc_text}")

                            url = row.get("raw_amazon", "")
                            if isinstance(url, str) and url.strip():
                                st.markdown(f"[View on Amazon]({url})")

                    st.markdown("---")


# ----- RIGHT: ROOM + CONCEPT VISUALIZER -----
with col_side:
    st.subheader("🛏️ AI concept visualizer")

    st.write(
        "Generates a **styled version of your uploaded room**, using Market & Place "
        "products as inspiration. The AI is instructed to keep layout/furniture the "
        "same and only change textiles. If image edits are not supported in your "
        "OpenAI setup, results may be a new concept room instead of an exact edit."
    )

    uploaded_image = st.file_uploader(
        "Upload a photo of your room (used as the base for styling where possible)",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded_image is not None:
        img_bytes = uploaded_image.getvalue()
        st.session_state.uploaded_room_image_bytes = img_bytes
        st.image(uploaded_image, caption="Uploaded room", use_column_width=True)

    room_description = st.text_area(
        "Describe your room & what you want:",
        value=st.session_state.room_description,
        placeholder="e.g. Small bedroom, white walls, light wood floors, want cozy neutral bedding...",
    )
    st.session_state.room_description = room_description

    if st.button("Generate concept image"):
        with st.spinner("Asking OpenAI to restyle your room..."):
            products_for_image = st.session_state.last_products
            img = generate_concept_image(
                st.session_state.room_description,
                products_for_image,
                st.session_state.uploaded_room_image_bytes,
            )
            if img is not None:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                st.session_state.concept_image_bytes = buf.getvalue()

    if st.session_state.concept_image_bytes is not None:
        st.image(
            st.session_state.concept_image_bytes,
            caption="AI-generated style concept",
            use_column_width=True,
        )

        # 🔹 Products used in this concept (shop the look)
        st.markdown("#### Products used in this concept")

        concept_products = st.session_state.last_products
        if concept_products is not None and not concept_products.empty:
            for _, row in concept_products.iterrows():
                with st.container():
                    cols = st.columns([1, 3])

                    # product image
                    with cols[0]:
                        img_url = row.get("Image URL:", "")
                        if isinstance(img_url, str) and img_url.strip():
                            try:
                                st.image(img_url, use_column_width=True)
                            except Exception:
                                st.empty()
                        else:
                            st.empty()

                    # product info + link
                    with cols[1]:
                        name = row.get("Product name", "")
                        color = row.get("Color", "")
                        price = row.get("Price", "")
                        url = row.get("raw_amazon", "")

                        st.markdown(f"**{name}**")
                        st.markdown(f"- Color: **{color}**")
                        st.markdown(f"- Price: **{price}**")
                        if isinstance(url, str) and url.strip():
                            # IMPORTANT: Amazon URL exactly as stored
                            st.markdown(f"[View on Amazon]({url})")

                st.markdown("---")

    st.markdown("---")
    st.subheader("🔎 Quick catalog peek")

    quick_query = st.text_input("Filter products by keyword", key="quick_query")
    if quick_query:
        preview = find_relevant_products(quick_query, max_results=10)
    else:
        preview = df.head(10)

    if not preview.empty:
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
                    name = row.get("Product name", "")
                    color = row.get("Color", "")
                    price = row.get("Price", "")
                    url = row.get("raw_amazon", "")

                    st.markdown(f"**{name}**")
                    st.markdown(f"- Color: **{color}**")
                    st.markdown(f"- Price: **{price}**")
                    if isinstance(url, str) and url.strip():
                        st.markdown(f"[View on Amazon]({url})")

            st.markdown("---")






