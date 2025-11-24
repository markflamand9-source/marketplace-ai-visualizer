import os
import streamlit as st
import pandas as pd
from openai import OpenAI

# ---------- CONFIG ----------
st.set_page_config(
    page_title="Market & Place AI Stylist",
    layout="wide",
)

st.title("🧠🛋️ Market & Place AI Stylist")
st.write(
    "Chat with an AI stylist, search the Market & Place catalog, and get suggestions "
    "for your room using your own product file."
)

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    return pd.read_excel("market_and_place_products.xlsx")

try:
    df = load_data()
except Exception as e:
    st.error(
        "❌ Could not load `market_and_place_products.xlsx`.\n\n"
        "Make sure the file is in the repo root and has columns: "
        "`Product name`, `Color`, `Price`, `raw_amazon`, `Image URL:`"
    )
    st.stop()

# ---------- OPENAI CLIENT ----------
# You must set OPENAI_API_KEY in Streamlit secrets or environment
api_key = os.getenv("OPENAI_API_KEY")
client = None
if api_key:
    client = OpenAI(api_key=api_key)

# ---------- SIMPLE PRODUCT SEARCH ----------
def find_relevant_products(user_query: str, max_results: int = 8):
    """
    Very simple keyword-based search over your Excel file.
    This keeps things cheap & fast and avoids sending the whole catalog to the model.
    """
    if not user_query:
        return df.head(max_results)

    q = user_query.lower()

    mask = (
        df["Product name"].fillna("").str.lower().str.contains(q)
        | df["Color"].fillna("").str.lower().str.contains(q)
    )

    results = df[mask].copy()

    # If nothing matches, just return a few items so the AI still has options
    if results.empty:
        results = df.sample(min(max_results, len(df)))

    return results.head(max_results)


def format_products_for_prompt(products: pd.DataFrame) -> str:
    """Convert product rows into compact text the model can reason over."""
    lines = []
    for _, row in products.iterrows():
        line = (
            f"- Name: {row.get('Product name', '')}\n"
            f"  Color: {row.get('Color', '')}\n"
            f"  Price: {row.get('Price', '')}\n"
            f"  Amazon URL: {row.get('raw_amazon', '')}\n"
            f"  Image URL: {row.get('Image URL:', '')}"
        )
        lines.append(line)
    return "\n\n".join(lines)


def call_ai_stylist(user_message: str, room_context: str, products: pd.DataFrame) -> str:
    """
    Call OpenAI to act as an AI stylist.
    It only recommends from the passed-in products.
    """
    if client is None:
        return (
            "⚠️ AI is not configured.\n\n"
            "Set your OpenAI API key as the environment variable "
            "`OPENAI_API_KEY` (in Streamlit Cloud: Settings → Secrets)."
        )

    product_text = format_products_for_prompt(products)

    system_prompt = """
You are an interior stylist for Market & Place.
You ONLY recommend products from the list given to you.
When you answer:

- Explain your reasoning in friendly language.
- Mention specific product names and colors.
- Include the exact Amazon URL string you are given, without changing it at all.
- If the user uploaded a room photo, take their description of the room into account
  (size, colors, vibe) when recommending.

Keep answers short but helpful, like a stylist chatting with a customer.
"""

    room_part = room_context or "The user did not describe the room."

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": (
                f"User message:\n{user_message}\n\n"
                f"Room description (from user):\n{room_part}\n\n"
                f"Here is the list of products you MAY choose from:\n\n{product_text}"
            ),
        },
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error calling AI: {e}"


# ---------- LAYOUT ----------
col_left, col_right = st.columns([2, 1])

# ---- LEFT: CHAT WITH AI STYLIST ----
with col_left:
    st.subheader("💬 Chat with the AI stylist")

    if "chat_history" not in st.session_state:
        st.session_state.chat_histor

