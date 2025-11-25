import base64
import io
from typing import List, Tuple

import pandas as pd
import streamlit as st
from openai import OpenAI

# -----------------------------
#  Setup
# -----------------------------
st.set_page_config(
    page_title="Market & Place AI Stylist",
    layout="wide",
)

client = OpenAI()

@st.cache_data
def load_catalog() -> pd.DataFrame:
    df = pd.read_excel("market_and_place_products.xlsx")
    df.columns = [c.strip() for c in df.columns]

    # Normalised helper columns
    name_lower = df["Product name"].str.lower()

    def classify(row_name: str) -> str:
        n = row_name.lower()
        if "towel" in n:
            if "beach" in n or "cabana" in n:
                return "beach_towel"
            if "luxury" in n:
                return "luxury_towel"
            return "towel"
        if any(k in n for k in ["sheet", "bedding", "duvet", "quilt", "comforter", "coverlet"]):
            return "sheet"
        return "other"

    df["category"] = name_lower.map(classify)
    df["name_lower"] = name_lower
    return df


CATALOG = load_catalog()

# -----------------------------
#  Helper functions
# -----------------------------
def pick_products_for_query(query: str, max_items: int = 4) -> pd.DataFrame:
    """Return a small subset of catalog rows based strictly on the Excel file."""
    q = query.lower()

    df = CATALOG

    # Decide category
    if "beach" in q and "towel" in q:
        subset = df[df["category"] == "beach_towel"]
    elif "luxury" in q and "towel" in q:
        subset = df[df["category"] == "luxury_towel"]
    elif "towel" in q:
        subset = df[df["category"].isin(["towel", "luxury_towel", "beach_towel"])]
        # avoid beach towels unless user mentioned beach
        subset = subset[~subset["category"].eq("beach_towel")]
    elif any(k in q for k in ["sheet", "bedding", "quilt", "duvet", "comforter"]):
        subset = df[df["category"] == "sheet"]
    else:
        # Fallback – any product containing any keyword
        words = [w for w in q.split() if len(w) > 3]
        mask = False
        for w in words:
            mask = mask | df["name_lower"].str.contains(w)
        subset = df[mask] if isinstance(mask, pd.Series) else df

    if subset.empty:
        return subset

    # If user mentions a color, bias toward that
    color_words = ["blue", "navy", "aqua", "white", "grey", "gray", "beige", "pink", "green", "yellow", "black"]
    for c in color_words:
        if c in q:
            subset = subset[subset["Color"].str.lower().str.contains(c, na=False)] or subset
            break

    return subset.head(max_items)


def render_product_list(df: pd.DataFrame):
    """Nice textual layout for the chat answers."""
    if df.empty:
        st.write("I couldn’t find matching products in the catalog for that request.")
        return

    for i, row in df.iterrows():
        st.markdown(
            f"**{len(st.session_state.get('last_products', [])) + (list(df.index).index(i) + 1)}. "
            f"{row['Product name']}**"
        )
        st.markdown(f"- Color: {row['Color']}")
        st.markdown(f"- Price: {row['Price']}")
        if "raw_amazon" in row and isinstance(row["raw_amazon"], str):
            url = row["raw_amazon"]
            st.markdown(f"- [View on Amazon]({url})")
        if "Image URL:" in row and isinstance(row["Image URL:"], str):
            st.image(row["Image URL:"], width=220)
        st.markdown("---")


def generate_concept_image(
    mode: str,
    room_type: str,
    user_description: str,
    uploaded_file,
) -> Tuple[bytes, str]:
    """
    Call OpenAI Images API.

    mode: "room" (uses uploaded image) or "store"
    Returns (image_bytes, error_message)
    """
    try:
        if mode == "room":
            if uploaded_file is None:
                return b"", "Please upload a room image first."

            base_prompt = (
                "You are an interior stylist AI working for Market & Place.\n"
                "Use the uploaded photo as the base image. "
                "Keep the **same room layout, walls, furniture, windows, floors, lighting, camera angle, "
                "and architecture**.\n"
                "You are ONLY allowed to change or add **textiles** such as towels, bath mats, shower curtains, "
                "bedding, quilts, throws, blankets, and curtains.\n"
                "Do **not** add new furniture, bathtubs, windows, doors, mirrors, art, plants, or objects. "
                "Do not move or remove any existing objects.\n"
                "Style the room using Market & Place products as inspiration for stripes, colors, and patterns, "
                "but do not invent products that are not in the catalog.\n"
                f"Room type: {room_type}.\n"
                f"User request: {user_description}\n"
            )

            image_file = io.BytesIO(uploaded_file.getvalue())

            result = client.images.generate(
                model="gpt-image-1",
                prompt=base_prompt,
                size="1024x1024",
                image=image_file,
            )

        else:  # store / showroom
            base_prompt = (
                "Create a realistic store-shelf visualization for Market & Place products. "
                "Show several tidy shelves with neatly folded towels or stacked bedding, "
                "similar to a big-box retail aisle. "
                "Use clean shelving and a neutral store background. "
                "Use colors and striping inspired by Market & Place textiles. "
                "Do not add unrelated products or signage.\n"
                f"User request: {user_description}"
            )

            result = client.images.generate(
                model="gpt-image-1",
                prompt=base_prompt,
                size="1024x1024",
            )

        b64 = result.data[0].b64_json
        img_bytes = base64.b64decode(b64)
        return img_bytes, ""
    except Exception as e:
        return b"", f"Image generation failed: {e}"


# -----------------------------
#  Session state
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_products" not in st.session_state:
    st.session_state.last_products = []


# -----------------------------
#  Layout – header
# -----------------------------
st.markdown(
    "<div style='text-align:center'>"
    "<img src='logo.png' alt='Market & Place logo' style='max-width:480px; width:60%;'>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='text-align:center;'>Market & Place AI Stylist</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;'>"
    "Chat with an AI stylist, search the Market & Place catalog, "
    "and generate concept visualizations using your own product file."
    "</p>",
    unsafe_allow_html=True,
)

st.markdown("[← Return to Market & Place website](https://marketandplace.co/)")

st.write("---")

# -----------------------------
#  Two-column main layout
# -----------------------------
left_col, right_col = st.columns([1.2, 1])

# ===== LEFT: chat + catalog =====
with left_col:
    st.subheader("Ask the AI stylist")

    with st.form("stylist_form", clear_on_submit=True):
        user_query = st.text_input(
            "Describe what you're looking for:",
            placeholder="e.g. luxury white towels for a modern bathroom",
            key="stylist_query",
        )
        submit_btn = st.form_submit_button("Send")

    if submit_btn and user_query.strip():
        # Clear previous chat – user said they want only the latest answer
        st.session_state.chat_history = []

        # Determine products
        recs = pick_products_for_query(user_query, max_items=4)
        st.session_state.last_products = list(recs.index)

        st.session_state.chat_history.append(
            {"role": "user", "content": user_query.strip()}
        )

        st.session_state.chat_history.append(
            {"role": "assistant", "content": recs}
        )

    # Render latest exchange only
    if st.session_state.chat_history:
        # last user message
        user_msg = st.session_state.chat_history[-2]["content"]
        st.markdown(f"🧵 **{user_msg}**")

        # last assistant "content" is actually the DataFrame
        df_assistant: pd.DataFrame = st.session_state.chat_history[-1]["content"]
        st.write(
            "Here are some Market & Place products that match your request and could work well in your space:"
        )
        render_product_list(df_assistant)

    st.write("---")
    st.subheader("Quick catalog peek")

    catalog_filter = st.text_input(
        "Filter products by keyword:",
        placeholder="e.g. towel, flannel, queen sheet, blue",
        key="catalog_filter",
    )

    df_view = CATALOG
    if catalog_filter.strip():
        q = catalog_filter.lower()
        df_view = df_view[df_view["name_lower"].str.contains(q)]

    # show a few items
    for _, row in df_view.head(6).iterrows():
        cols = st.columns([1, 3])
        with cols[0]:
            if isinstance(row.get("Image URL:"), str):
                st.image(row["Image URL:"], use_column_width=True)
        with cols[1]:
            st.markdown(f"**{row['Product name']}**")
            st.markdown(f"- Color: {row['Color']}")
            st.markdown(f"- Price: {row['Price']}")
            if isinstance(row.get("raw_amazon"), str):
                st.markdown(f"[View on Amazon]({row['raw_amazon']})")
        st.markdown("---")

# ===== RIGHT: image generator =====
with right_col:
    st.subheader("AI concept visualizer")

    st.markdown(
        "Generates a styled version of your uploaded room, using Market & Place products "
        "as inspiration for the textiles. The AI is instructed to keep the layout the same "
        "and only change towels / bedding / curtains / rugs."
    )

    mode = st.radio(
        "What do you want to visualize?",
        ["In my room photo", "Store shelf / showroom"],
        index=0,
        key="visualize_mode",
    )

    if mode == "In my room photo":
        room_type = st.selectbox(
            "Room type:",
            ["bathroom", "bedroom", "living room", "kitchen", "other"],
            index=0,
        )
        uploaded_room = st.file_uploader(
            "Upload a reference photo of your room:",
            type=["jpg", "jpeg", "png"],
        )
        img_desc = st.text_input(
            "What would you like to visualize?",
            placeholder="e.g. add navy-and-white striped towels on the rack",
            key="room_prompt",
        )

        if st.button("Generate concept image", key="room_generate"):
            if not uploaded_room:
                st.error("Please upload a room image first.")
            elif not img_desc.strip():
                st.error("Please describe what you’d like to visualize.")
            else:
                img_bytes, err = generate_concept_image(
                    mode="room",
                    room_type=room_type,
                    user_description=img_desc,
                    uploaded_file=uploaded_room,
                )
                if err:
                    st.error(err)
                else:
                    st.image(img_bytes, caption="AI-generated style concept", use_column_width=True)

    else:  # Store shelf / showroom
        shelf_desc = st.text_input(
            "What would you like to visualize on shelves?",
            placeholder="e.g. assortment of striped beach towels in aqua and navy",
            key="shelf_prompt",
        )

        if st.button("Generate concept image", key="shelf_generate"):
            if not shelf_desc.strip():
                st.error("Please describe what you’d like to visualize on the shelves.")
            else:
                img_bytes, err = generate_concept_image(
                    mode="store",
                    room_type="store shelf",
                    user_description=shelf_desc,
                    uploaded_file=None,
                )
                if err:
                    st.error(err)
                else:
                    st.image(img_bytes, caption="Store shelf concept", use_column_width=True)












