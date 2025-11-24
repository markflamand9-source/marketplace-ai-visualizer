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

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f6fa;
    }
    .block-container {
        max-width: 1200px;
        margin: 0 auto;
        padding-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ================== HEADER WITH CENTERED LOGO ==================

# small top spacer
st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

logo_cols = st.columns([1, 2, 1])
with logo_cols[1]:
    # logo.png must be in repo root
    st.image("logo.png", use_column_width=False)

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


# ================== ASK-THE-STYLIST SEARCH BAR ==================

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

DESC_COL = None  # no dedicated description column in this file


# ================== PRODUCT HELPERS ==================

def find_relevant_products(query: str, max_results: int = 6) -> pd.DataFrame:
    """
    Simple keyword search over product name and color.

    Special handling:
    - If the user mentions towels, we first try to return ONLY towel-ish products
      (towels, bath rugs/mats, bath sheets, etc.).
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

    towel_pattern = "towel|bath sheet|bath rug|bathmat|bath mat|washcloth|wash cloth"

    if wants_towel:
        name_series = df["Product name"].fillna("").str.lower()
        mask_towel = name_series.str.contains(towel_pattern)
        results = df[mask_towel].copy()
        if not results.empty:
            return results.head(max_results)

    # generic search fallback
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
    """Ask GPT for styling suggestions based on the catalog."""
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


# ================== ROOM-TYPE INFERENCE FOR IMAGES ==================

def infer_room_type(room_description: str, last_query: str, products: pd.DataFrame) -> str:
    """
    Very simple heuristic: decide if we're styling a bathroom vs bedroom vs generic room.
    """
    text = (room_description or "") + " " + (last_query or "")
    text = text.lower()

    name_series = products["Product name"].fillna("").str.lower() if products is not None else pd.Series([])

    # Bathroom signals
    bath_keywords = [
        "bathroom", "shower", "toilet", "wc", "sink",
        "towel", "bath towel", "hand towel", "bath rug", "bath mat", "bathmat",
    ]
    if any(k in text for k in bath_keywords) or name_series.str.contains("towel|bath rug|bathmat|bath mat").any():
        return "bathroom"

    # Bedroom signals
    bed_keywords = [
        "bedroom", "bed", "duvet", "comforter", "quilt", "sheet set", "pillowcase",
    ]
    if any(k in text for k in bed_keywords) or name_series.str.contains("quilt|sheet set|duvet|comforter").any():
        return "bedroom"

    return "room"


# ================== OPENAI CALLS (IMAGES) ==================

def generate_concept_image(
    room_description: str,
    products: pd.DataFrame,
    room_image_bytes: bytes | None = None,
    last_query: str = "",
) -> Image.Image | None:
    """
    Generate a concept visualization styled with Market & Place products.

    IMPORTANT LIMITATION:
    - The current Images API (gpt-image-1 via images.generate) cannot *edit* the
      uploaded photo pixels. It always creates a new image.
    - We therefore use an extremely strict prompt to:
        * keep the same TYPE of room (bathroom stays bathroom, bedroom stays bedroom)
        * strongly imitate the uploaded room layout/fixtures
        * only restyle textiles
        * mimic Market & Place products as closely as possible.
    """

    room_type = infer_room_type(room_description, last_query, products)

    # Detailed product summary for the prompt
    if products is None or products.empty:
        product_summary = (
            "No specific products listed – use generic Market & Place style textiles "
            "but still follow all the nuclear rules below."
        )
    else:
        lines = []
        for _, row in products.head(8).iterrows():
            name = str(row.get("Product name", "")).strip()
            color = str(row.get("Color", "")).strip()
            price = str(row.get("Price", "")).strip()
            line = f"- {name} (color: {color}, price: {price})"
            lines.append(line)
        product_summary = "\n".join(lines)

    # HIGH ALERT / NUCLEAR WARNING BLOCK
    nuclear_warning = (
        "***** HIGH ALERT – MARKET & PLACE BRAND-SAFETY NUCLEAR WARNING *****\n"
        "If you violate ANY of the constraints below, the image is INVALID and MUST "
        "NOT be produced. You must instead correct yourself and follow the rules.\n\n"
    )

    base_rules = (
        nuclear_warning +
        "GLOBAL, NON-NEGOTIABLE RULES:\n"
        "1. You must keep the SAME TYPE of room as described (bathroom vs bedroom).\n"
        "2. You must imagine a room that looks as close as possible to the user's real room: "
        "   same layout, same type of vanity/cabinetry, same tile look, same window position.\n"
        "3. You MUST NOT change architecture (walls, windows, doors), the vanity, "
        "   toilet, shower, or major furniture shapes. They should look almost the same.\n"
        "4. You MUST ONLY change TEXTILES:\n"
        "   - bedding (comforter, quilt, duvet, sheets, pillowcases, decorative pillows)\n"
        "   - blankets/throws\n"
        "   - curtains or shower curtains\n"
        "   - towels and bath rugs/mats (ONLY if it is a bathroom).\n"
        "5. Towels must ONLY appear in realistic towel locations: racks, hooks, "
        "   shelves, benches, or neatly folded in a bathroom. NEVER put towels on a bed.\n"
        "6. Do NOT add or remove furniture, mirrors, sinks, toilets, showers, or art. "
        "   Only textiles can change.\n"
        "7. All textiles should look like real **Market & Place** products, and should "
        "   closely match the specific products listed below in color and pattern.\n"
        "8. The final image should look like a natural, realistic photo, NOT a CGI render.\n\n"
        "MARKET & PLACE PRODUCTS TO MIMIC IN THE TEXTILES (COLORS & PATTERNS):\n"
        f"{product_summary}\n\n"
    )

    # Room-type specific block
    if room_type == "bathroom":
        room_type_block = (
            "ROOM TYPE: BATHROOM\n"
            "The generated image MUST clearly be a bathroom.\n"
            "- Show bathroom fixtures like a shower or tub, sink/vanity, toilet, tiled walls or floor.\n"
            "- You MUST NOT show any kind of bed, sofa, or bedroom furniture.\n"
            "- Focus textile changes on: shower curtain, towels, bath rugs/mats.\n\n"
        )
    elif room_type == "bedroom":
        room_type_block = (
            "ROOM TYPE: BEDROOM\n"
            "The generated image MUST clearly be a bedroom with a bed as the main furniture.\n"
            "- You MUST NOT show a toilet, shower, or bathroom fixtures.\n"
            "- Focus textile changes on: duvet/comforter, pillows, sheets, throws, and curtains.\n\n"
        )
    else:
        room_type_block = (
            "ROOM TYPE: GENERIC LIVING SPACE\n"
            "Keep the same general type of room as described by the user. Do not convert "
            "it into a bathroom or bedroom unless it obviously is one from the description.\n\n"
        )

    # Uploaded-photo context
    if room_image_bytes is not None:
        upload_context = (
            "The user has uploaded a photo of their real room. You cannot directly edit "
            "that photo, but you MUST imagine a room that looks almost identical in layout "
            "and fixtures, only changing the textiles to the Market & Place options.\n\n"
        )
    else:
        upload_context = (
            "The user did NOT upload a photo. Create a realistic room that matches the "
            "description, then apply the textile rules above.\n\n"
        )

    desc_text = room_description or "No extra room description was provided."

    prompt = (
        base_rules
        + room_type_block
        + upload_context
        + "USER ROOM DESCRIPTION:\n"
        + desc_text
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

    except Exception as e:
        st.error(
            "Could not generate a concept image with the current image API.\n\n"
            f"Technical details: {e}"
        )
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

if "last_query" not in st.session_state:
    st.session_state.last_query = ""


# ================== HANDLE TOP SEARCH / CHAT SUBMIT ==================

if submitted and user_input:
    # store last query for room-type inference
    st.session_state.last_query = user_input

    # add user message
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

# ----- LEFT: CHAT + PRODUCT IMAGES + CATALOG PEEK -----
with col_chat:
    st.subheader("Chat with the AI stylist")

    products = st.session_state.last_products
    msgs = st.session_state.messages

    # full chat history
    for msg in msgs:
        avatar = "🙂" if msg["role"] == "user" else "🧵"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Recommended products always directly under last AI answer
    if products is not None and not products.empty and msgs and msgs[-1]["role"] == "assistant":
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

                # info
                with cols[1]:
                    name = row.get("Product name", "")
                    color = row.get("Color", "")
                    price = row.get("Price", "")

                    st.markdown(f"**{name}**")
                    st.markdown(f"- Color: **{color}**")
                    st.markdown(f"- Price: **{price}**")

                    desc_text = (
                        f"This {str(color).lower() if isinstance(color, str) else ''} "
                        f"{name} brings a Market & Place look into the space and ties "
                        "the palette together."
                    )
                    st.markdown(f"- Description: {desc_text}")

                    url = row.get("raw_amazon", "")
                    if isinstance(url, str) and url.strip():
                        st.markdown(f"[View on Amazon]({url})")

            st.markdown("---")

    # Quick catalog peek UNDER the chat & recommended products
    st.markdown("### 🔎 Quick catalog peek")

    quick_query = st.text_input(
        "Filter products by keyword",
        key="quick_query",
        placeholder="e.g. towel, quilt, grey, queen...",
    )
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


# ----- RIGHT: ROOM + CONCEPT VISUALIZER -----
with col_side:
    st.subheader("🛏️ AI concept visualizer")

    st.write(
        "Generates a **styled version of your room**, using Market & Place "
        "products as inspiration.\n\n"
        "- The AI is instructed to **keep the same type of room** (bathroom stays bathroom, "
        "bedroom stays bedroom) and **only change textiles**.\n"
        "- It is explicitly told **not to move furniture, change walls, or put towels on the bed**.\n"
        "- Because the current image API cannot directly edit your exact photo, it recreates the "
        "room from your description + these nuclear rules."
    )

    uploaded_image = st.file_uploader(
        "Upload a photo of your room (for the AI to imitate layout & for preview)",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded_image is not None:
        img_bytes = uploaded_image.getvalue()
        st.session_state.uploaded_room_image_bytes = img_bytes
        st.image(uploaded_image, caption="Uploaded room", use_column_width=True)

    room_description = st.text_area(
        "Describe your room & what you want:",
        value=st.session_state.room_description,
        placeholder="e.g. Small bathroom with walk-in shower and grey tiles, want soft neutral Market & Place towels...",
    )
    st.session_state.room_description = room_description

    if st.button("Generate concept image"):
        with st.spinner("Asking OpenAI to restyle your room..."):
            products_for_image = st.session_state.last_products
            img = generate_concept_image(
                st.session_state.room_description,
                products_for_image,
                st.session_state.uploaded_room_image_bytes,
                st.session_state.last_query,
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










