import streamlit as st
import ollama
from utils import remove_think_tags

def init_deepseek_interaction():
    """Initialize a conversation history for deepseek-r1 in st.session_state if not already present."""
    if "deepseek_interaction" not in st.session_state:
        st.session_state["deepseek_interaction"] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful stock-analysis assistant. "
                    "Only produce a concise, final answer. Do not show your chain-of-thought. "
                    "Provide historical context about this specific company to help analyze stock trends. "
                    "ONLY 2 bullet points max."
                )
            }
        ]

def append_user_message(content: str):
    st.session_state["deepseek_interaction"].append({"role": "user", "content": content})

def append_assistant_message(content: str):
    st.session_state["deepseek_interaction"].append({"role": "assistant", "content": content})

def call_deepseek_r1_via_ollama() -> str:
    """
    Calls the Ollama-based 'deepseek-r1' model with the conversation in st.session_state["deepseek_interaction"].
    Returns the assistant's final message content.
    """
    try:
        response = ollama.chat(
            model="deepseek-r1",  # Adjust model name/tag if needed
            messages=st.session_state["deepseek_interaction"]
        )
        return response.message.content
    except Exception as e:
        return f"Error calling deepseek-r1: {str(e)}"

def get_deepseek_analysis(symbol: str, start_date, end_date) -> str:
    """
    Creates a prompt about the stock performance, calls deepseek-r1, and returns its final message without chain-of-thought.
    """
    init_deepseek_interaction()

    user_prompt = (
        f"Analyze {symbol} stock performance between {start_date:%Y-%m-%d} and {end_date:%Y-%m-%d}. "
        "Explain key reasons for any price movements, referencing ALL RELEVANT corporate announcements, "
        "ALL recent product launches, partnerships, earnings reports, market sentiment, or other external factors. "
        "Focus on specific developments such as new products, services, innovations, or notable trends directly tied to this company "
        "(e.g., product launches like Apple Vision Pro, new processors, or updates to flagship products). "
        "Provide a concise, fact-based summary in a maximum of 4 bullet points. "
        "If necessary, provide historical context about this company to help analyze stock trends."
    )
    append_user_message(user_prompt)

    raw_reply = call_deepseek_r1_via_ollama()
    append_assistant_message(raw_reply)

    # Remove any <think> tags before displaying
    cleaned_reply = remove_think_tags(raw_reply)
    return cleaned_reply