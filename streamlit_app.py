import base64
import io
import os
import re
from typing import List, Dict

import pandas as pd
import streamlit as st
from openai import OpenAI

# ---------- CONFIG ----------

st.set_page_config(
    page_title="Market & Place AI Stylist",
    layout="wide",
)

client = OpenAI()  # Uses OPENAI_API_KEY env var


# ---------- DATA LOADING ----------

@st.cache_data(show_spinner=False)
def load_catalog() -> pd.DataFrame:
    df = pd.read_excel("market_and_place_products.xlsx")
    # Normalise columns just in case
    df.columns = [c.strip() for c in df.columns]
    for col in ["Product name", "Color", "Price", "raw_amazon", "Image URL:"]:
        if col not in df.columns:
            raise ValueError(f"Missing column in Excel: {col}")
    return df


catalog_df = load_catalog()

# Add some helper columns for quick filtering
def classify_product(row) -> str:
    name = str(row["Product name"]).lower()
    if any(k in name for k in ["towel", "beach towel", "bath towel"]):
        return "towel"
    if any(k in name for k in ["sheet", "quilt", "comforter", "duvet", "coverlet", "bedding"]):
        return "bedding"
    if any(k in name for k in ["rug", "bath rug", "mat"]):
        return "rug"
    return "other"


catalog_df["category"] = catalog_df.apply(classify_product, axis=1)


# ---------- PRODUCT SEARCH HELPERS (NO HALLUCINATIONS) ----------

def parse_price_cap(text: str) -> float | None:
    """
    Look for things like 'under $80', 'below 60', '< 100', '$50 max', etc.
    """
    text = text.lower()
    # under / below / less than
    m = re.search(r"(under|below|less than|<=|<)\s*\$?(\d+)", text)
    if m:
        return float(m.group(2))
    # 'for $80' or 'for 80'
    m = re.search(r"for\s*\$?(\d+)", text)
    if m:
        return float(m.group(1))
    return None


def detect_desired_category(text: str) -> str | None:
    t = text.lower()
    if "towel" in t or "bathroom" in t or "bath" in t:
        return "towel"
    if any(k in t for k in ["bedding", "bed", "sheet", "quilt", "duvet", "comforter"]):
        return "bedding"
    if any(k in t for k in ["rug", "mat"]):
        return "rug"
    return None


def product_keyword_filter(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Soft keyword filter – only ever *removes* rows, never invents anything.
    """
    tokens = [t for t in re.split(r"[^\w]+", query.lower()) if t]
    if not tokens:
        return df

    mask = pd.Series(False, index=df.index)
    for tok in tokens:
        mask = mask | df["Product name"].str.lower().str.contains(tok, na=False)
        mask = mask | df["Color"].astype(str).str.lower().str.contains(tok, na=False)

    # If nothing matched, just return original df for that category
    if not mask.any():
        return df
    return df[mask]


def get_recommendations(user_query: str, top_n: int = 5) -> pd.DataFrame:
    """
    Core logic: pick products from Excel only. No GPT involved.
    """
    category = detect_desired_category(user_query) or "other"
    df = catalog_df

    if category != "other":
        df = df[df["category"] == category]

    # Apply price cap if present
    cap = parse_price_cap(user_query)
    if cap is not None:
        df = df[df["Price"] <= cap]

    # Keyword filter
    df = product_keyword_filter(df, user_query)

    # Fall back if everything was filtered out
    if df.empty:
        df = catalog_df

    return df.head(top_n)


# ---------- AI IMAGE GENERATION ----------

def build_image_prompt(room_text: str, selected_products: pd.DataFrame) -> str:
    """
    Build a *very* strict prompt for gpt-image-1 using only catalog rows.
    """
    product_lines = []
    for _, row in selected_products.iterrows():
        product_lines.append(
            f"- {row['Product name']} (Color: {row['Color']}, "
            f"Price: ${row['Price']:.2f})"
        )

    product_block = "\n".join(product_lines) if product_lines else "No products found."

    prompt = f"""
MARKET & PLACE IMAGE NUCLEAR RULES (READ CAREFULLY AND OBEY EXACTLY):

1. You are designing ONLY textiles for an existing real-world scene.
2. You MUST treat every product listed below as the ONLY allowed textiles.
   - You are NOT allowed to invent or show any other brands or products.
   - You may stylise slightly but the color palette and pattern style must match the product descriptions.
3. Do NOT change the architecture, layout, lighting, or furniture of the room conceptually:
   - If the user uploads a bathroom, you MUST generate a bathroom.
   - Do NOT turn bathrooms into bedrooms or vice versa.
   - Do NOT move or add furniture such as beds, toilets, vanities, showers, windows, mirrors.
4. You may ONLY change:
   - towels, shower curtains, bath rugs, bath mats, bedding, quilts, sheets, decorative pillows,
     and other soft textiles.
5. Absolutely NO hallucinated products:
   - If something textile-like is visible, it MUST plausibly correspond to one of the products below.
   - If you need more variety, you may repeat or re-arrange these products, but never invent new ones.
6. If you are unsure, choose the simplest option that follows these rules.

ROOM CONTEXT / USER REQUEST:
{room_text.strip()}

MARKET & PLACE TEXTILES YOU ARE ALLOWED TO USE (AND NOTHING ELSE):
{product_block}

Now generate a high-quality concept image that:
- Keeps the same type of room the user described (bathroom stays bathroom, bedroom stays bedroom).
- Keeps all hard surfaces, furniture layout, and wall color simple and neutral.
- Shows Market & Place textiles prominently and clearly.
"""
    # Strip excessive whitespace
    return "\n".join(line.rstrip() for line in prompt.splitlines()).strip()


def generate_concept_image(room_text: str, selected_products: pd.DataFrame) -> bytes | None:
    """
    Call gpt-image-1 to generate a concept image.
    NOTE: OpenAI's current image API does NOT truly edit the uploaded photo.
    It generates a new scene inspired by the prompt.
    """
    prompt = build_image_prompt(room_text, selected_products)

    try:
        img_resp = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024",
        )
        b64_data = img_resp.data[0].b64_json
        return base64.b64decode(b64_data)
    except Exception as e:
        st.error(f"Image generation failed: {e}")
        return None


# ---------- UI LAYOUT ----------

# Centered logo
with st.container():
    st.markdown(
        """
        <div style="text-align: center; margin-top: 1rem; margin-bottom: 0.5rem;">
        """,
        unsafe_allow_html=True,
    )

    if os.path.exists("logo.png"):
        st.image("logo.png", width=420, output_format="PNG")
    else:
        st.write("⚠️ `logo.png` not found in app folder.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align: center; margin-bottom: 0.25rem;'>Market &amp; Place AI Stylist</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='text-align: center; margin-bottom: 1.5rem;'>"
    "Chat with an AI stylist, search the Market &amp; Place catalog, and generate concept visualizations using your own product file."
    "</p>",
    unsafe_allow_html=True,
)

# Return link
st.markdown(
    "[← Return to Market & Place website](https://marketandplace.co/)",
)

st.write("")  # small spacer

# Maintain chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[Dict] = []


left_col, right_col = st.columns([1.1, 1])

# ---------- LEFT: CHAT WITH AI STYLIST (CATALOG-STRICT) ----------

with left_col:
    st.subheader("Ask the AI stylist")

    user_query = st.text_input(
        "Describe what you're looking for:",
        placeholder="e.g. neutral queen bedding under $80 for a bright room, or colourful towels for a modern bathroom",
        key="user_query",
    )

    if st.button("Send", key="send_chat"):
        if user_query.strip():
            recs = get_recommendations(user_query, top_n=6)
            st.session_state.chat_history.insert(
                0,
                {
                    "query": user_query.strip(),
                    "results": recs.to_dict(orient="records"),
                },
            )
            # Clear input
            st.session_state.user_query = ""

    # Display chat history (newest at top)
    for item in st.session_state.chat_history:
        q = item["query"]
        results = item["results"]

        st.markdown(
            f"<div style='padding: 0.6rem 0.75rem; background-color: #f7f7f9; "
            f"border-radius: 0.5rem; margin-top: 0.8rem;'>"
            f"<strong>🧵 You:</strong> {q}"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("**Stylist suggestions (catalog-strict):**")
        for i, prod in enumerate(results, start=1):
            st.markdown(
                f"**{i}. {prod['Product name']}**  \n"
                f"- Color: {prod['Color']}  \n"
                f"- Price: ${prod['Price']:.2f}  \n"
                f"- [View on Amazon]({prod['raw_amazon']})"
            )

    st.markdown("---")
    st.markdown("### Quick catalog peek")

    keyword = st.text_input(
        "Filter products by keyword (optional):",
        placeholder="e.g. towel, queen, navy",
        key="peek_filter",
    )

    peek_df = catalog_df
    if keyword.strip():
        peek_df = product_keyword_filter(peek_df, keyword)

    for _, row in peek_df.head(8).iterrows():
        with st.container():
            cols = st.columns([0.5, 1.2])
            with cols[0]:
                img_url = row["Image URL:"]
                if isinstance(img_url, str) and img_url.strip():
                    st.image(img_url, use_column_width=True)
            with cols[1]:
                st.markdown(f"**{row['Product name']}**")
                st.markdown(f"- Color: {row['Color']}")
                st.markdown(f"- Price: ${row['Price']:.2f}")
                st.markdown(f"[View on Amazon]({row['raw_amazon']})")

# ---------- RIGHT: AI CONCEPT VISUALIZER ----------

with right_col:
    st.subheader("🖼️ Your image")

    uploaded_file = st.file_uploader(
        "Upload a photo of your room (bathroom, bedroom, etc.) or leave empty if you just want a generic store-shelf visualization.",
        type=["jpg", "jpeg", "png"],
    )

    st.markdown("### What would you like to visualize?")
    room_description = st.text_area(
        "Describe your room & what you want:",
        placeholder=(
            "e.g. Small bathroom with grey tiles and white walls, want colourful striped towels and "
            "a bath mat using Market & Place products.\n\n"
            "You can also say: 'towel section on store shelves using our cabana stripe towels'."
        ),
        height=140,
    )

    # Choose what kind of scene
    visualize_mode = st.selectbox(
        "Where should the products appear?",
        [
            "In a real home room (bathroom, bedroom, etc.)",
            "On store shelves / retail display",
        ],
    )

    if st.button("Generate concept image", key="generate_image"):
        if not room_description.strip():
            st.warning("Please describe the room or store display you want to visualize.")
        else:
            # Use the same recommendation logic to pick a handful of products
            img_products = get_recommendations(room_description, top_n=6)

            # Build a combined room text including mode + mention of upload
            mode_text = (
                "The scene should look like a real-life home interior."
                if visualize_mode.startswith("In a real home")
                else "The scene should look like realistic retail store shelves with folded or hanging product."
            )
            upload_note = (
                "The user also uploaded a reference photo; imitate its general layout and mood."
                if uploaded_file is not None
                else "The user did not provide a reference photo; design a clean, simple neutral space."
            )

            combined_room_text = f"{room_description}\n\n{mode_text}\n\n{upload_note}"

            image_bytes = generate_concept_image(combined_room_text, img_products)
            if image_bytes:
                st.image(image_bytes, caption="AI-generated style concept", use_column_width=True)

                # Also show which products were used in the prompt
                st.markdown("#### Products fed into the image prompt")
                for _, row in img_products.iterrows():
                    st.markdown(
                        f"- **{row['Product name']}** (Color: {row['Color']}, "
                        f"Price: ${row['Price']:.2f}) – [View on Amazon]({row['raw_amazon']})"
                    )












