import streamlit as st
from typing import Generator, Optional, Dict, Union
from groq import Groq
import os
import random
import time

def _get_system_prompt() -> str:
    """Get system prompt from a file."""
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "system_prompt.txt")
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

system_prompt = _get_system_prompt()
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

st.set_page_config(page_icon="🛸", layout="wide", page_title="DigiDopps™ Cosmic Chat")

# Custom CSS with theme support
def get_custom_css():
    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
        
        .stApp {{
            background-image: linear-gradient(to right, 
                {("rgba(15, 12, 41, 0.9)" if st.session_state.theme == "dark" else "rgba(255, 255, 255, 0.9)")}, 
                {("rgba(48, 43, 99, 0.9)" if st.session_state.theme == "dark" else "rgba(240, 240, 255, 0.9)")}, 
                {("rgba(36, 36, 62, 0.9)" if st.session_state.theme == "dark" else "rgba(230, 230, 250, 0.9)")});
            color: {("#ffffff" if st.session_state.theme == "dark" else "#333333")};
            font-family: 'Orbitron', sans-serif;
        }}
        .stSelectbox, .stSlider {{
            background-color: {("rgba(255, 255, 255, 0.1)" if st.session_state.theme == "dark" else "rgba(0, 0, 0, 0.1)")};
            border-radius: 10px;
            padding: 10px;
        }}
        .stChat {{
            border-radius: 15px;
            border: 1px solid {("rgba(255, 255, 255, 0.2)" if st.session_state.theme == "dark" else "rgba(0, 0, 0, 0.2)")};
        }}
        .stChatMessage {{
            background-color: {("rgba(255, 255, 255, 0.05)" if st.session_state.theme == "dark" else "rgba(0, 0, 0, 0.05)")};
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
            animation: fadeIn 0.5s ease-out;
        }}
        .stChatInputContainer {{
            border-top: 1px solid {("rgba(255, 255, 255, 0.2)" if st.session_state.theme == "dark" else "rgba(0, 0, 0, 0.2)")};
            padding-top: 20px;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .cosmic-particle {{
            position: fixed;
            width: 2px;
            height: 2px;
            background-color: #ffffff;
            pointer-events: none;
            opacity: 0;
            animation: twinkle 5s infinite;
        }}
        @keyframes twinkle {{
            0% {{ opacity: 0; }}
            50% {{ opacity: 1; }}
            100% {{ opacity: 0; }}
        }}
    </style>
    """

st.markdown(get_custom_css(), unsafe_allow_html=True)

# Add cosmic particles
for _ in range(50):
    left = random.randint(0, 100)
    top = random.randint(0, 100)
    delay = random.uniform(0, 5)
    st.markdown(f"""
    <div class="cosmic-particle" style="left: {left}vw; top: {top}vh; animation-delay: {delay}s;"></div>
    """, unsafe_allow_html=True)

def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(f'<span style="font-size: 78px; line-height: 1">{emoji}</span>', unsafe_allow_html=True)

icon("🌌")
st.markdown('<h1 style="text-align: center; color: #ADD8E6; text-shadow: 0 0 10px #ADD8E6;">Cosmic Chat with Bashar</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center; color: #FFD700; text-shadow: 0 0 5px #FFD700;">Powered by Groq 🛸</h3>', unsafe_allow_html=True)

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# Define model details
models = {
    "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
    "llama2-70b-4096": {"name": "LLaMA2-70b-chat", "tokens": 4096, "developer": "Meta"},
    "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
    "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
    "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
}

# Layout for model selection, max_tokens slider, and theme toggle
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    model_option = st.selectbox(
        "Choose your cosmic guide 👽",
        options=list(models.keys()),
        format_func=lambda x: f"{models[x]['name']} ({models[x]['developer']})",
        index=2  # Default to LLaMA
    )

# Detect model change and clear chat history if model has changed
if st.session_state.selected_model != model_option:
    st.session_state.messages = [
        {"role": "system", "content": system_prompt}
    ]
    st.session_state.selected_model = model_option

max_tokens_range = models[model_option]["tokens"]

with col2:
    max_tokens = st.slider(
        "Cosmic Energy (Max Tokens) 🚀:",
        min_value=512,
        max_value=max_tokens_range,
        value=min(32768, max_tokens_range),
        step=512,
        help=f"Adjust the maximum cosmic energy for the response. Max for selected guide: {max_tokens_range}"
    )

with col3:
    if st.button("Toggle Theme 🌓"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()

# Display chat messages from history on app rerun
st.markdown("### Cosmic Transmissions")
for message in st.session_state.messages:
    if message["role"] != "system":  # Do not display the system message
        with st.chat_message(message["role"], avatar='👽' if message["role"] == "assistant" else '🧑🏾‍💻'):
            st.markdown(message["content"])

def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
            time.sleep(0.05)  # Add a small delay for a typing effect

st.markdown("### Initiate Cosmic Transmission")
if prompt := st.chat_input("Transmit your cosmic query...", key="user_input"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar='🧑🏾‍💻'):
        st.markdown(prompt)

    try:
        with st.spinner("🌀 Aligning with cosmic frequencies..."):
            chat_completion = client.chat.completions.create(
                model=model_option,
                messages=st.session_state.messages,
                max_tokens=max_tokens,
                stream=True
            )

            # Use the generator function with st.write_stream
            with st.chat_message("assistant", avatar="👽"):
                chat_responses_generator = generate_chat_responses(chat_completion)
                full_response = st.write_stream(chat_responses_generator)
    except Exception as e:
        st.error(f"Cosmic disturbance detected: {e}", icon="🚨")

    # Append the full response to session_state.messages
    if isinstance(full_response, str):
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        combined_response = "\n".join(str(item) for item in full_response)
        st.session_state.messages.append({"role": "assistant", "content": combined_response})

# Visualization of conversation statistics
if len(st.session_state.messages) > 1:
    st.markdown("### Cosmic Conversation Insights")
    user_messages = [msg for msg in st.session_state.messages if msg["role"] == "user"]
    ai_messages = [msg for msg in st.session_state.messages if msg["role"] == "assistant"]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transmissions", len(st.session_state.messages) - 1)  # Subtract 1 to exclude system message
    col2.metric("Your Queries", len(user_messages))
    col3.metric("Cosmic Responses", len(ai_messages))

st.markdown("---")
st.markdown("Channeling cosmic wisdom through Groq 🚀 | Crafted with ❤️ by Vers3Dynamics")
