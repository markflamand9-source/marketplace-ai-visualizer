import os
import base64
from io import BytesIO

import pandas as pd
import streamlit as st
from openai import OpenAI
from PIL import Image

# ---------- OpenAI client ----------
client = OpenAI()

# ---------- Page config ----------
st.set_page_config(
    page_title="Market & Place AI Stylist",
    layout="wide",
)

# ---------- Helpers ----------

@st.cache_data(show_spinner=False)
def load_catalog(path: str = "market_and_place_products.xlsx") -> pd.DataFrame:
    df = pd.read_excel(path)

    # Normalize column names just in case
    df.columns = [c.strip() for c in df.columns]

    # Lower-case name for quick search
    df["name_lower"] = df["Product name"].str.lower()

    # Basic category tagging
    def categorize(name: str) -> str:
        n = name.lower()
        if "towel" in n:
            if "beach" in n:
                return "beach_towel"
            if "luxury" in n:
                return "luxury_towel"
            return "towel"
        if any(x in n for x in ["sheet", "quilt", "comforter", "duvet", "bedding", "coverlet"]):
            return "bedding"
        return "other"

    df["category"] = df["Product name"].astype(str).apply(categorize)
    return df


def choose_category_from_query(q: str) -> str:
    ql = q.lower()
    if "towel" in ql or "bath" in ql:
        if "beach" in ql:
            return "beach_towel"
        if "luxury" in ql:
            return "luxury_towel"
        return "towel"
    if any(x in ql for x in ["sheet", "bedding", "duvet", "comforter", "quilt", "coverlet"]):
        return "bedding"
    return "other"


def filter_products(df: pd.DataFrame, query: str, max_results: int = 6) -> pd.DataFrame:
    """
    Return a small slice of catalog that best matches the query.
    Enforces towel vs bedding logic.
    """
    ql = query.lower()
    desired_cat = choose_category_from_query(ql)

    if desired_cat != "other":
        subset = df[df["category"] == desired_cat].copy()
        # If they said "luxury" and we have luxury within the towels/bedding, bias to that
        if "luxury" in ql and not subset.empty:
            lux = subset[subset["Product name"].str.contains("luxury", case=False, na=False)]
            if not lux.empty:
                subset = lux
    else:
        subset = df.copy()

    # Simple keyword scoring
    keywords = [w for w in ql.split() if len(w) > 2]
    def score(row):
        name = row["name_lower"]
        return sum(name.count(k) for k in keywords)

    subset["score"] = subset.apply(score, axis=1)
    subset = subset.sort_values("score", ascending=False)

    if subset["score"].max() == 0:
        # If no real keyword hit, just return first few in that category
        subset = subset.drop(columns=["score"]).head(max_results)
    else:
        subset = subset.head(max_results).drop(columns=["score"])

    return subset


def openai_image_from_prompt(prompt: str, size: str = "1024x1024") -> Image.Image:
    """Call OpenAI image API and return a PIL image. Errors are raised to caller."""
    img_resp = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size=size,
        n=1,
    )
    b64 = img_resp.data[0].b64_json
    img_bytes = base64.b64decode(b64)
    return Image.open(BytesIO(img_bytes))


def build_room_prompt(room_type: str, user_desc: str, ref_products: pd.DataFrame) -> str:
    """
    Strong instructions to *try* to keep same layout and only change textiles.
    Realistically the model may still improvise, but we push it hard.
    """
    product_lines = []
    for _, row in ref_products.head(6).iterrows():
        product_lines.append(
            f"- {row['Product name']} (Color: {row['Color']})"
        )
    products_text = "\n".join(product_lines) if product_lines else "Market & Place textiles."

    prompt = f"""
Photo of the SAME {room_type} layout as the reference image, from the same camera angle.

HARD RULES:
- Keep walls, floor, windows, doors, fixtures and furniture identical.
- Do NOT add bathtubs, showers, vanities, shelves, beds, chairs, plants or any other new object.
- Do NOT remove any existing furniture or architecture.
- Only change removable TEXTILES (towels, shower curtain, bath mat, bedding, pillows, throws, rugs).
- All visible textiles must look like real Market & Place products:
{products_text}

Style request from user: "{user_desc}".
The room should look realistically photographed, not a CGI scene.
"""
    return prompt.strip()


def build_store_prompt(user_desc: str, ref_products: pd.DataFrame) -> str:
    product_lines = []
    for _, row in ref_products.head(10).iterrows():
        product_lines.append(
            f"- {row['Product name']} (Color: {row['Color']})"
        )
    products_text = "\n".join(product_lines) if product_lines else "Market & Place striped towels."

    prompt = f"""
Realistic retail store shelves fully stocked with Market & Place textiles only.

Layout rules:
- Regular straight store SHELVES only (no bathtubs, no beds, no extra furniture).
- Focus on neatly folded stacks and hanging textiles.
- Background is a simple store aisle; nothing distracting.

All products on the shelves should look like these Market & Place items:
{products_text}

User description: "{user_desc}".
"""
    return prompt.strip()


# ---------- Load data ----------
catalog = load_catalog()

# ---------- Layout: Title + logo ----------
col_left, col_center, col_right = st.columns([1, 2, 1])
with col_center:
    st.image("logo.png", use_column_width=False)
    st.markdown(
        "<h1 style='text-align: center;'>Market &amp; Place AI Stylist</h1>",
        unsafe_allow_html=True,
    )

st.write(
    "Chat with an AI stylist, search the Market & Place catalog, "
    "and generate concept visualizations using your own product file."
)

st.markdown(
    "<a href='https://marketandplace.co/' target='_blank'>"
    "← Return to Market &amp; Place website</a>",
    unsafe_allow_html=True,
)

st.markdown("---")

# ======================================================================
#   1. ASK THE AI STYLIST  (text product suggestions)
# ======================================================================

st.subheader("Ask the AI stylist")

with st.form("stylist_form", clear_on_submit=False):
    user_query = st.text_input(
        "Describe what you're looking for:",
        placeholder="e.g. neutral queen bedding under $80 for a bright room",
    )
    submit_stylist = st.form_submit_button("Send")

# reset old answer on every submit – only show the most recent
if submit_stylist and not user_query.strip():
    st.warning("Tell me what you’re looking for first.")
    st.session_state["stylist_result"] = None
elif submit_stylist:
    try:
        results = filter_products(catalog, user_query, max_results=6)
        st.session_state["stylist_result"] = {"query": user_query, "results": results}
    except Exception as e:
        st.session_state["stylist_result"] = None
        st.error(f"Could not search the catalog: {e}")

if "stylist_result" in st.session_state and st.session_state["stylist_result"]:
    data = st.session_state["stylist_result"]
    q = data["query"]
    df_res = data["results"]

    # Emoji + heading
    st.markdown(
        f"### 🧵 {q.strip().lower() if q.strip() else 'Your ideas'}",
    )
    st.write(
        "Here are some Market & Place products that match your request "
        "and could work well in your space:"
    )

    for idx, row in df_res.iterrows():
        st.markdown(
            f"**{idx + 1}. {row['Product name']}**"
        )
        st.write(f"- Color: {row['Color']}")
        st.write(f"- Price: {row['Price']}")
        st.markdown(f"- [View on Amazon]({row['raw_amazon']})")

        # Optional image thumbnail
        if "Image URL:" in row and isinstance(row["Image URL:"], str):
            st.image(row["Image URL:"], width=220)

        st.markdown("---")

# ======================================================================
#   2. QUICK CATALOG PEEK
# ======================================================================

st.subheader("Quick catalog peek")

peek_query = st.text_input(
    "Filter products by keyword:",
    placeholder="e.g. towel, queen sheet, flannel, beach…",
    key="peek_query",
)

if peek_query.strip():
    pq = peek_query.lower()
    peek_df = catalog[catalog["name_lower"].str.contains(pq)].head(10)
else:
    peek_df = catalog.head(10)

for _, row in peek_df.iterrows():
    cols = st.columns([1, 2])
    with cols[0]:
        if "Image URL:" in row and isinstance(row["Image URL:"], str):
            st.image(row["Image URL:"], width=120)
    with cols[1]:
        st.markdown(f"**{row['Product name']}**")
        st.write(f"Color: {row['Color']}")
        st.write(f"Price: {row['Price']}")
        st.markdown(f"[View on Amazon]({row['raw_amazon']})")
    st.markdown("---")

# ======================================================================
#   3. AI CONCEPT VISUALIZER
# ======================================================================

st.subheader("AI concept visualizer")

st.write(
    "Generate a styled version of your uploaded room using Market & Place products, "
    "or visualize how products would look on store shelves."
)

mode = st.radio(
    "What do you want to visualize?",
    ["In my room photo", "Store shelf / showroom"],
    horizontal=False,
)

if mode == "In my room photo":
    room_type = st.selectbox(
        "Room type:",
        ["bathroom", "bedroom", "living room", "kitchen", "entryway", "other"],
        index=0,
    )

    room_file = st.file_uploader(
        "Upload a reference photo of your room:",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        key="room_upload",
    )

    desc_room = st.text_input(
        "What would you like to visualize?",
        placeholder="e.g. cozy neutral towels and bath mat, no shower curtain",
        key="room_desc",
    )

    if room_file:
        st.image(room_file, caption="Uploaded room (reference)", use_column_width=True)

    if st.button("Generate concept image", key="room_button"):
        if not room_file:
            st.error("Please upload a room image first.")
        elif not desc_room.strip():
            st.error("Describe what textiles you want to change.")
        else:
            with st.spinner("Generating concept image…"):
                try:
                    # Pick products based on description
                    room_products = filter_products(catalog, desc_room, max_results=8)
                    prompt = build_room_prompt(room_type, desc_room, room_products)
                    img = openai_image_from_prompt(prompt)
                    st.image(img, caption="AI-generated style concept", use_column_width=True)
                except Exception as e:
                    st.error(f"Image generation failed: {e}")

else:  # Store shelf / showroom
    st.info(
        "Tip: leave the image empty to generate a clean Market & Place shelf, "
        "or upload a photo of your store aisle as reference."
    )

    store_file = st.file_uploader(
        "Optional: upload a reference photo of your store shelves:",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        key="store_upload",
    )

    desc_store = st.text_input(
        "Describe what you want to see on the shelves:",
        placeholder="e.g. rainbow stacks of beach towels with matching bath mats",
        key="store_desc",
    )

    if store_file:
        st.image(store_file, caption="Uploaded shelves (reference)", use_column_width=True)

    if st.button("Generate shelf concept image", key="store_button"):
        if not desc_store.strip():
            st.error("Describe how you want the shelves to look.")
        else:
            with st.spinner("Generating store shelf concept…"):
                try:
                    store_products = filter_products(catalog, desc_store, max_results=10)
                    prompt = build_store_prompt(desc_store, store_products)
                    img = openai_image_from_prompt(prompt)
                    st.image(img, caption="AI-generated store shelf concept", use_column_width=True)
                except Exception as e:
                    st.error(f"Image generation failed: {e}")












