import streamlit as st
import time
import random

st.set_page_config(
    page_title="ChatGPT-like UI",
    page_icon="💬",
    layout="centered"
)

st.markdown("""
<style>
    .main {
        background-color: #f7f7f8;
    }
    .chat-message {
        padding: 1.5rem; 
        border-radius: 0.8rem; 
        margin-bottom: 1rem; 
        display: flex;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #f0f2f6;
    }
    .chat-message.bot {
        background-color: white;
    }
    .chat-message .avatar {
        width: 35px;
        height: 35px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }
    .chat-message .message {
        flex-grow: 1;
    }
    .sidebar .sidebar-content {
        background-color: #f7f7f8;
        align-items: center;
        justify-content: center;
    }
    .stTextInput>div>div>input {
        border-radius: 0.8rem;
        padding: 1rem;
        font-size: 1.1rem;
    }
    .stButton>button {
        background-color: #10a37f;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-size: 1.1rem;
        border: none;
        justify-content: center;
    }
    .stButton>button:hover {
        background-color: #0d8a6c;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        text-align: center;
        font-size: 0.8rem;
        color: #888;
        padding: 0.5rem;
        background: transparent;
        z-index: 9999;
    }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "bot", "content": "Hello! I'm your AI assistant. How can I help you today?"}
    ]

if "thinking" not in st.session_state:
    st.session_state.thinking = False

def generate_response(prompt):
    
    responses = [
        f"Thank you for asking about '{prompt}'. Here's what I know about this topic...",
        f"That's an interesting question about '{prompt}'. Let me share some thoughts...",
        f"I'd be happy to help with '{prompt}'. Based on my knowledge...",
        f"Regarding '{prompt}', there are several important points to consider..."
    ]
    
    time.sleep(2)
    
    return random.choice(responses)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.thinking:
    with st.chat_message("bot"):
        st.write("Thinking...")

prompt = st.chat_input("Ask something...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.thinking = True
    st.rerun()

if st.session_state.thinking:
    response = generate_response(st.session_state.messages[-1]["content"])
    
    st.session_state.messages.append({"role": "bot", "content": response})
    
    # Display assistant response
    with st.chat_message("bot"):
        st.markdown(response)
    
    st.session_state.thinking = False
with st.sidebar:
    
    if st.button("New Chat"):
        st.session_state.messages = [
            {"role": "bot", "content": "Hello! I'm your AI assistant. How can I help you today?"}
        ]
        st.rerun()
    
    st.divider()
    
    st.caption("This is a demo ChatGPT-like UI built with Streamlit.")

st.markdown("""
<div class="footer">
    © 2025 Demo App
</div>
""", unsafe_allow_html=True)
