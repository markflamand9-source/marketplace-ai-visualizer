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

# Top hero with logo
st.markdown(
    """
    <style>
    .mp-logo-container {
        display: flex;
        justify-content: center;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .mp-logo {
        max-width: 520px;
        height: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container():
    st.markdown('<div class="mp-logo-container">', unsafe_allow_html=True)
    try:
        st.image("logo.png", use_container_width=False, output_format="PNG")
    except Exception:
        # Fallback if logo is missing
        st.title("Market & Place")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align:center; margin-bottom: 0.25rem;'>Market & Place AI Stylist</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='text-align:center; color:#555;'>"
    "Chat with an AI stylist, search the Market & Place catalog, "
    "and generate concept visualizations using your own product file."
    "</p>",
    unsafe_allow_html=True,
)


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


def format_products_for_prompt(products: pd.DataFrame, max_items: int = 10) -> str:
    """Compact text block describing products for the model."""
    lines = []
    for idx, (_, row) in enumerate(products.iterrows()):
        if idx >= max_items:
            break
        line = (
            f"- Name: {row.get('Product name', '')}\n"
            f"  Color: {row.get('Color', '')}\n"
            f"  Price: {row.get('Price', '')}\n"
            f"  Amazon URL (internal ref only): {row.get('raw_amazon', '')}"
        )
        lines.append(line)
    return "\n\n".join(lines)


# ================== OPENAI CHAT CALL ==================

def call_stylist_model(user_message: str, room_context: str,
                       products: pd.DataFrame) -> str:
    """Ask GPT-4.1-mini for styling suggestions based on the catalog."""
    product_block = format_products_for_prompt(products)
    room_part = room_context or "The user did not describe the room."

    system_prompt = """
You are an interior stylist working for the home-textile brand **Market & Place**.

You must follow these rules VERY STRICTLY (NUCLEAR PRIORITY):

1. You ONLY recommend products from the product list provided below in the prompt.
2. For each suggested product, you MUST clearly output:
   • Product name  
   • Color  
   • Price  
   • Short explanation of why it fits  
   • Amazon URL (copy EXACTLY from the data, do not change, trim, or add parameters)  
3. Never invent new products, colors, patterns, or URLs that are not explicitly given.
4. If the user asks for towels, you should prefer towel products. If they ask for bedding, prefer bedding, etc.
5. If nothing matches perfectly, choose the closest matches from the catalog and SAY that you are picking the closest options.
6. All Amazon URLs must be copied character-for-character from the catalog.

Respond in a friendly, concise way and group recommendations into a numbered list (3–5 items).
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"User request:\n{user_message}\n\n"
                f"Room description:\n{room_part}\n\n"
                "Here is the ONLY catalog you may use (Market & Place products):\n\n"
                f"{product_block}"
            ),
        },
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.6,
            max_tokens=800,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error calling OpenAI chat model: `{e}`"


# ================== IMAGE GENERATION ==================

def build_image_prompt(
    mode: str,
    room_description: str,
    products: pd.DataFrame,
) -> str:
    """
    Build a VERY strict prompt for the image model.

    mode = "room" or "store"
    """

    product_block = format_products_for_prompt(products, max_items=8)

    base_rules = f"""
You are generating a concept visualization for the brand **Market & Place**.

***NUCLEAR RULES (DO NOT BREAK):***
- You must visually stay as close as possible to the uploaded image's layout, camera angle, and architecture.
- The room TYPE must never change. A bathroom must stay a bathroom, a bedroom must stay a bedroom, etc.
- You are only allowed to change **removable textiles**:
  • towels, bath mats, shower curtains  
  • bedding, duvets, quilts, sheets, pillowcases, throw blankets  
  • decorative pillows, rugs, and window curtains  
- You are NOT allowed to:
  • move or remove walls, windows, doors, mirrors, vanities, sinks, toilets, showers, tubs, shelves, cabinets, or lighting  
  • change wall color, tile, flooring material, or major furniture pieces  
  • add new furniture or large objects  
  • put towels on beds or bedding in a bathroom, etc. (respect correct textile usage)
- All textiles should look like **Market & Place** products. You may only take inspiration from the catalog below.
- Do NOT invent fictional brands or random patterns. Use colors/patterns that feel realistic for the listed Market & Place items.

Here is a description of the space from the user (if any):
{room_description or "The user did not provide extra description."}

Here is the Market & Place product catalog you must take inspiration from:
{product_block}
"""

    if mode == "room":
        mode_text = """
Task: Restyle the **same room** shown in the uploaded image by updating ONLY the textiles.
Do not redesign the architecture. Do not change the room type. Focus on towels/bedding/rugs/curtains depending on what is visible.
"""
    else:
        # store shelves mode
        mode_text = """
Task: Show how Market & Place products would look merchandised on **store shelves**.

- If a shelf/store photo is uploaded, KEEP the same shelving layout, perspective, walls, and lighting.
- Only change what is on the shelves: add folded towels, stacked bedding, and packaged textiles that look like Market & Place items.
- If no photo is uploaded, create a realistic retail shelf scene with neutral background and shelves filled with Market & Place textiles.
"""

    return base_rules + "\n" + mode_text


def generate_concept_image(
    mode: str,
    room_description: str,
    products: pd.DataFrame,
    room_image_bytes: bytes | None = None,
) -> Image.Image | None:
    """
    Generate a concept visualization:

    - mode="room": restyle the user's room photo (only textiles).
    - mode="store": show products on store shelves (distributor view).

    We FIRST try to edit the uploaded image (if any) using images.edit().
    If that is not available or fails, we fall back to images.generate().
    """

    prompt = build_image_prompt(mode, room_description, products)

    try:
        # Try to use the uploaded image as a base, if we have it and the client supports `edit`.
        if room_image_bytes is not None and hasattr(client.images, "edit"):
            img_file = io.BytesIO(room_image_bytes)
            img_file.name = "base.png"

            img_resp = client.images.edit(
                model="gpt-image-1",
                image=img_file,
                prompt=prompt,
                size="1024x1024",
            )
        else:
            # No base image or edit not supported: generate a concept from scratch
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
            "Could not generate a concept image. "
            "This can be due to org verification, billing, API limits, or image-edit support.\n\n"
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

if "uploaded_room_image_bytes" not in st.session_state:
    st.session_state.uploaded_room_image_bytes = None

if "concept_image_bytes" not in st.session_state:
    st.session_state.concept_image_bytes = None

if "image_mode" not in st.session_state:
    st.session_state.image_mode = "Style my room"


# ================== LAYOUT ==================

col_chat, col_side = st.columns([2.2, 1.6])

# ----- LEFT: CHAT + PRODUCT IMAGES + CATALOG PEEK -----
with col_chat:
    # Return link to main site
    st.markdown(
        "[← Return to Market & Place website](https://marketandplace.co/)",
        unsafe_allow_html=False,
    )

    st.subheader("Ask the AI stylist")

    prompt_label = "Describe what you're looking for:"
    st.write("")  # small spacing
    query_placeholder = (
        "e.g. neutral queen bedding under $80 for a bright room, or 'towel ideas for a blue bathroom'"
    )

    # Simple text input chat (so the catalog section can sit under it)
    user_text = st.text_input(prompt_label, placeholder=query_placeholder)

    send_clicked = st.button("Send", type="primary")

    st.markdown("---")
    st.subheader("Chat with the AI stylist")

    # show full chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 🔹 Product cards ALWAYS directly under the last AI answer
    products = st.session_state.last_products

    if products is not None and not products.empty:
        st.markdown("#### Recommended products for this conversation")

        for _, row in products.iterrows():
            with st.container():
                cols = st.columns([1, 3])

                # image (from Image URL:, which points to Amazon-hosted image)
                with cols[0]:
                    img_url = row.get("Image URL:", "")
                    if isinstance(img_url, str) and img_url.strip():
                        try:
                            st.image(img_url, use_container_width=True)
                        except Exception:
                            st.empty()
                    else:
                        st.empty()

                # info
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
                        # IMPORTANT: show Amazon URL EXACTLY as stored
                        st.markdown(f"[View on Amazon]({url})")

            st.markdown("---")

    # 🔹 Quick catalog peek now lives UNDER the chat so it scrolls with everything
    st.subheader("🔍 Quick catalog peek")

    quick_query = st.text_input(
        "Filter products by keyword",
        key="quick_query",
        placeholder="e.g. towel, sheet set, flannel, queen, grey...",
    )
    if quick_query:
        preview = find_relevant_products(quick_query, max_results=12)
    else:
        preview = df.head(12)

    for _, row in preview.iterrows():
        with st.container():
            cols = st.columns([1, 3])
            with cols[0]:
                img_url = row.get("Image URL:", "")
                if isinstance(img_url, str) and img_url.strip():
                    try:
                        st.image(img_url, use_container_width=True)
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

    # ----- Handle new user message -----
    if send_clicked and user_text.strip():
        # user message
        st.session_state.messages.append({"role": "user", "content": user_text})

        # pick candidate products for this question
        candidate_products = find_relevant_products(user_text, max_results=6)
        st.session_state.last_products = candidate_products.copy()

        # use stored room description from right column
        room_desc = st.session_state.room_description

        # AI reply
        reply = call_stylist_model(user_text, room_desc, candidate_products)
        st.session_state.messages.append({"role": "assistant", "content": reply})

        st.rerun()


# ----- RIGHT: ROOM / STORE + CONCEPT VISUALIZER -----
with col_side:
    st.subheader("🖼️ Your image")

    # Mode selector: room vs store shelves
    mode_label = st.selectbox(
        "What would you like to visualize?",
        ["Style my room", "Show products on store shelves"],
        index=0,
    )
    st.session_state.image_mode = mode_label

    uploaded_image = st.file_uploader(
        "Upload a photo (room or store shelves). The AI will TRY to keep layout and only change textiles.",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded_image is not None:
        img_bytes = uploaded_image.getvalue()
        st.session_state.uploaded_room_image_bytes = img_bytes
        st.image(uploaded_image, caption="Uploaded image", use_column_width=True)

    room_description = st.text_area(
        "Describe your room / store & what you want:",
        value=st.session_state.room_description,
        placeholder=(
            "e.g. Small bathroom with white tiles, want striped towels and bath mat "
            "using Market & Place colors...\n"
            "or: Grocery aisle shelves, want stacks of colorful Market & Place towels."
        ),
    )
    st.session_state.room_description = room_description

    st.markdown("---")
    st.subheader("🛋️ AI concept visualizer")

    st.write(
        """
Generates a **styled version of your uploaded image**, using Market & Place
products as inspiration.

**Important:**
- The AI is instructed to keep the same room or store layout and **only change textiles**.
- It is also told to **only** take inspiration from the Market & Place catalog, not invent random products.
- Results are still generative concept visuals – not pixel-perfect overlays of actual SKUs.
"""
    )

    if st.button("Generate concept image"):
        with st.spinner("Asking OpenAI to restyle your image..."):
            products_for_image = st.session_state.last_products
            mode = "room" if st.session_state.image_mode == "Style my room" else "store"

            img = generate_concept_image(
                mode,
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

        # Small reminder that the products used correspond to the current last_products
        if (
            st.session_state.last_products is not None
            and not st.session_state.last_products.empty
        ):
            st.markdown("##### Products used as inspiration for this image")
            for _, row in st.session_state.last_products.iterrows():
                st.markdown(f"- **{row.get('Product name', '')}** — {row.get('Color', '')}")











