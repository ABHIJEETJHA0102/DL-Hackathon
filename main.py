import streamlit as st
import time
import random
import uuid
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="ChatGPT-like UI",
    page_icon="💬",
    layout="centered"
)

# Custom CSS for styling
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
        align-items: center;
            
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
        width: 100%;
        text-align: left;
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
    .session-id {
        background-color: #e9f7f2;
        border-radius: 4px;
        margin-bottom: 10px;
        font-size: 0.9rem;
        border-left: 3px solid #10a37f;
    }
</style>
""", unsafe_allow_html=True)

# Track all sessions and messages
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}

if "all_messages" not in st.session_state:
    st.session_state.all_messages = {}

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "thinking" not in st.session_state:
    st.session_state.thinking = False


# Function to simulate the AI thinking and generating response
def generate_response(prompt):
    responses = [
        f"Thank you for asking about '{prompt}'. Here's what I know about this topic...",
        f"That's an interesting question about '{prompt}'. Let me share some thoughts...",
        f"I'd be happy to help with '{prompt}'. Based on my knowledge...",
        f"Regarding '{prompt}', there are several important points to consider..."
    ]
    
    time.sleep(2)
    
    return random.choice(responses)

if st.session_state.session_id is None:
    # Show the welcome message when no session is active
    with st.chat_message("bot"):
        st.markdown("Hello! I'm your AI assistant. How can I help you today?")
else:
    if st.session_state.session_id and st.session_state.session_id in st.session_state.all_messages:
        messages = st.session_state.all_messages[st.session_state.session_id]
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

prompt = st.chat_input("Ask something...")

if prompt:
    if st.session_state.session_id is None:
        # Create a new session
        new_session_id = str(uuid.uuid4())
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save new session info
        st.session_state.chat_sessions[new_session_id] = {
            "last_updated": current_time
        }
        
        # Set current session
        st.session_state.session_id = new_session_id

        # Initialize messages for new session
        st.session_state.all_messages[new_session_id] = []
    
    session_id = st.session_state.session_id
    st.session_state.all_messages[session_id].append({"role": "user", "content": prompt})
    
    # Update last modified
    st.session_state.chat_sessions[session_id]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    st.session_state.thinking = True
    st.rerun()

if st.session_state.thinking:
    response = generate_response(
        st.session_state.all_messages[st.session_state.session_id][-1]["content"]
    )

    st.session_state.all_messages[st.session_state.session_id].append({"role": "bot", "content": response})
    st.session_state.chat_sessions[st.session_state.session_id]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with st.chat_message("bot"):
        st.markdown(response)

    st.session_state.thinking = False

    
# Sidebar for new chat and settings
with st.sidebar:
    # New chat button
    if st.button("New Chat"):
        st.session_state.session_id = None
        st.rerun()
    
    # Previous sessions section (optional - shows 5 most recent)
    # Show recent sessions
    if len(st.session_state.chat_sessions) > 0:
        st.divider()
        st.subheader("Recent Chats")

        sorted_sessions = sorted(
            st.session_state.chat_sessions.items(),
            key=lambda x: x[1]["last_updated"],
            reverse=True
        )

        count = 0
        for session_id, _ in sorted_sessions:
            label = f"Session {session_id[:8]}..."
            if st.button(label, key=f"session_{session_id}"):
                print(f"[DEBUG] Selected session: {session_id}")
                st.session_state.session_id = session_id
                st.rerun()
            count += 1
            if count >= 5:
                break
    
    st.divider()

st.markdown("""
<div class="footer">
    © 2025 Demo App
</div>
""", unsafe_allow_html=True)
