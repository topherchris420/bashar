import streamlit as st
from typing import Generator
from groq import Groq
import base64
import os
from typing import Optional, Dict, Union


def display_sidebar(self) -> None:
        ""

def _get_system_prompt():
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "system_prompt.txt")
    with open(file_path, "r", encoding="utf-8") as file:
        prompt = file.read()
        return prompt

system_prompt = _get_system_prompt()

display_sidebar()


st.set_page_config(page_icon="coast_chris.png", layout="wide",
                   page_title="Vers3Dynamics")


def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )


icon("🐶")
st.markdown(f'<a href="https://visualverse.streamlit.app/" style="text-decoration:none; color: #0e76a8;"><h2>Vers3Dynamics</h2></a>', unsafe_allow_html=True)
st.subheader("Meet Your Virtual Assistants, Powered by Groq 🚀", divider="rainbow", anchor=False)

# Add a picture with a caption
st.image("images/WelcomeHometitle.png", caption="What can I help you with?", width=200)

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"],
)

# Initialize chat history and selected model
if "messages" not in st.session_state:
    st.session_state.messages = []

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


# Assistant Prompt Management (assuming these functions exist)
def setup_prompts(self) -> None:
  """
  Sets up the prompts by reading them from the specified directory and initializing
  the session state with default and custom assistant prompts.

  If the "assistants" key is not present in the session state, it creates it with
  a default assistant prompt and any additional prompts read from the directory.

  Returns:
      None
  """
  prompts: Dict[str, str] = self.read_prompts_from_directory(self.prompts_directory)
  
  if "assistants" not in st.session_state:
    st.session_state.assistants = {
        "Leonardo da Vinci": (
            "Hello! I'm your helpful assistant, ready to assist you with any questions or tasks you have. "
            "Whether you need information, advice, or just someone to chat with, I'm here to help. "
            "Just let me know how I can assist you today!"
        ),
        **prompts
    }

def sidebar_assistant_management(self) -> None:
  """
  Manages the assistant selection functionality in the sidebar.

  This method allows the user to select an assistant from a dropdown menu.
  The selected assistant is stored in the session state.

  Returns:
      None
  """
  selected_assistant: str = st.sidebar.selectbox(
      "Select an Assistant:",  # Add a title for the dropdown menu
      options=list(st.session_state.assistants.keys())  # Use existing assistant names
  )
  
  st.session_state['selected_assistant'] = selected_assistant




# Layout for model selection and max_tokens slider
col1, col2 = st.columns(2)

with col1:
    model_option = st.selectbox(
        "Connect with the perfect AI:",
        options=list(models.keys()),
        format_func=lambda x: models[x]["name"],
        index=2  # Default to LLaMA
    )

# Detect model change and clear chat history if model has changed
if st.session_state.selected_model != model_option:
    st.session_state.messages = []
    st.session_state.selected_model = model_option

max_tokens_range = models[model_option]["tokens"]

with col2:
    max_tokens = st.slider(
        "Max Tokens 🪙:",
        min_value=512,
        max_value=max_tokens_range,
        value=min(32768, max_tokens_range),
        step=512,
        help=f"Adjust the maximum number of tokens for the model's response. Max for selected model: {max_tokens_range}"
    )

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = '🐶' if message["role"] == "assistant" else '🧑🏾‍💻'
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

system_prompt = _get_system_prompt()
prompt = system_prompt
st.session_state.messages.append({"role": "user", "content": prompt})
st.session_state.selected_model = None


if prompt := st.chat_input("Hi! I'm James, your FurBuddy. How may I help you?", key="user_input"):
    st.session_state.messages.append({"role": "user", "content": prompt})
   
    # Process the user's input and respond accordingly
    # Use the system_prompt variable to guide the chatbot's responses

    with st.chat_message("user", avatar='🧑🏾‍💻'):
        st.markdown(prompt)

    # Fetch response from Groq API
    try:
        chat_completion = client.chat.completions.create(
            model=model_option,
            messages=[
                {
                    "role": m["role"],
                    "content": m["content"]
                }
                for m in st.session_state.messages
            ],
            max_tokens=max_tokens,
            stream=True
        )

        # Use the generator function with st.write_stream
        with st.chat_message("assistant", avatar="🐶"):
            chat_responses_generator = generate_chat_responses(chat_completion)
            full_response = st.write_stream(chat_responses_generator)
    except Exception as e:
        st.error(f"Oops! Something went wrong: {e}", icon="🐢🚨")

    # Append the full response to session_state.messages
    if isinstance(full_response, str):
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})
    else:
        # Handle the case where full_response is not a string
        combined_response = "\n".join(str(item) for item in full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": combined_response})
        st.session_state.messages.append(
            {"role": "assistant", "content": combined_response})

