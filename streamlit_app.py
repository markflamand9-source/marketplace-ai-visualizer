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

# Soft background + centered content
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f6fa;
    }
    .block-container {
        padding-top: 1.5rem !important;   /* overall top spacing */
        padding-bottom: 1.5rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ================== HEADER WITH CENTERED LOGO ==================

# Three columns; put logo in the middle to guarantee centering
logo_cols = st.columns([1, 2, 1])
with logo_cols[1]:
    st.image("logo.png", use_column_width=False)

# Small spacer between logo and title
st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

st.markdown(
    """
    <h1 style="text-align:center; margin-top:0rem; margin-bottom:0.15rem;">
        Market &amp; Place AI Stylist
    </h1>
    <p style="text-align:center; font-size:1.05rem; color:#555; margin-top:0;">
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

DESC_COL = None  # there is no separate description column


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

    towel_terms_in_query = [
        "towel", "towels", "bath towel", "bath towels",
        "hand towel", "hand towels", "washcloth", "wash cloth",
        "bath sheet", "bath sheets",
    ]
    wants_towel = any(t in q for t in towel_terms_in_query)

    towel_pattern = "towel|bath sheet|washcloth|wash cloth"

    if wants_towel:
        name_series = df["Product name"].fillna("").str.lower()
        mask_towel = name_series.str.contains(towel_pattern)
        results = df[mask_towel].copy()
        if not results.empty:
            return results.head(max_results)

    mask_generic = (
        df["Product name"].fillna("").str.lower().str.contains(q)
        | df["Color"].fillna("").str.lower().str.contains(q)
    )
    results = df[mask_generic].copy()

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


# ================== OPENAI CALLS (TEXT) ==================

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


# ================== OPENAI CALLS (IMAGES) ==================

def generate_concept_image(
    room_description: str,
    products: pd.DataFrame,
    room_image_bytes: bytes | None = None,
) -> Image.Image | None:
    """
    Generate a visualization styled with Market & Place products.

    BEHAVIOR:
    - If a room image is provided:
        * Use OpenAI images.edits on THAT PHOTO ONLY.
        * Do NOT fall back to a random concept if editing fails.
    - If NO room image is provided:
        * Generate a new concept room from scratch.
    """

    top_names = ", ".join(products["Product name"].head(4).tolist())

    prompt_base = (
        "You are doing STRICT PHOTO EDITING for the brand Market & Place.\n"
        "Your job is to ONLY restyle TEXTILES while keeping the base room and "
        "furniture exactly the same.\n\n"
        "NON-NEGOTIABLE RULES:\n"
        "1. The room architecture, camera angle, bed shape, windows, doors, floor, "
        "   wall color, artwork, shelves, decor objects, plants, lamps, nightstands, "
        "   picture frames and all furniture MUST remain IDENTICAL.\n"
        "2. You may NOT move, resize, add, or remove furniture or decor items. Their "
        "   position, shape and style must stay the same.\n"
        "3. You may NOT change wall color, lighting, perspective, or crop. It must "
        "   look like the same photo.\n"
        "4. The ONLY things you may modify are TEXTILES:\n"
        "   - bedding (duvet, quilt, sheets, pillowcases, throws)\n"
        "   - decorative pillows\n"
        "   - blankets/throws\n"
        "   - curtains\n"
        "   - towels\n"
        "5. Towels must appear only in realistic towel locations: bathroom racks, "
        "   hooks, shelves, benches, or neatly folded. Do NOT put towels on the bed.\n"
        "6. If a change would affect anything other than textiles, DO NOT make it.\n\n"
        "Restyle ONLY the textiles so they look like these Market & Place products: "
        f"{top_names}.\n"
    )

    try:
        # ----- CASE 1: User uploaded a real room photo → TRUE EDIT ONLY -----
        if room_image_bytes is not None:
            prompt = (
                prompt_base
                + "Use the uploaded room photo as the base image. Copy the room "
                  "layout exactly and only update the textiles as described."
            )

            img_file = io.BytesIO(room_image_bytes)
            img_file.name = "room.png"

            img_resp = client.images.edits(
                model="gpt-image-1",
                image=img_file,
                prompt=prompt,
                size="1024x1024",
            )

            b64_data = img_resp.data[0].b64_json
            image_bytes = base64.b64decode(b64_data)
            img = Image.open(io.BytesIO(image_bytes))
            return img

        # ----- CASE 2: No room photo → concept from scratch is allowed -----
        else:
            prompt = (
                prompt_base
                + "There is no base photo. Generate a NEW bedroom that showcases "
                  "these textiles in a realistic way. Do NOT place towels on the bed."
            )

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
        # For uploaded rooms we DO NOT fall back to a random room – we just show an error
        if room_image_bytes is not None:
            st.error(
                "Could not generate an edited version of your room image. "
                "The image API may not support this type of edit yet.\n\n"
                f"Technical details: {e}"
            )
        else:
            st.error(f"Image generation failed: {e}")
        return None


# ================== STREAMLIT STATE ==================

if "messages" not in st.session_state:
    st.session_state.messages = []

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
    st.session_state.messages.append({"role": "user", "content": user_input})

    candidate_products = find_relevant_products(user_input, max_results=6)
    st.session_state.last_products = candidate_products.copy()

    room_desc = st.session_state.room_description

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
    rev_msgs = list(reversed(msgs))  # newest first on top

    for i, msg in enumerate(rev_msgs):
        avatar = "🙂" if msg["role"] == "user" else "🧵"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

        # Under the NEWEST AI message, show product cards
        if i == 0 and msg["role"] == "assistant":
            if products is not None and not products.empty:
                st.markdown("#### Recommended products for this conversation")

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
        "products as inspiration.\n\n"
        "- When you upload a room photo, the AI is instructed to **copy your room** "
        "and **only change textiles** (bedding, pillows, throws, curtains, towels).\n"
        "- It is explicitly told **not to move furniture, change walls, or put towels "
        "on the bed**.\n"
        "- If the edit is not supported by the image API, you will see an error instead "
        "of a random new room."
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

        st.markdown("#### Products used in this concept")

        concept_products = st.session_state.last_products
        if concept_products is not None and not concept_products.empty:
            for _, row in concept_products.iterrows():
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









