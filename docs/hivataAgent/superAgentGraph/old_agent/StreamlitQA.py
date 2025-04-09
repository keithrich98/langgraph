# StreamlitQA.py

import streamlit as st
import requests

# Define the API base URL (adjust if necessary).
API_URL = "http://localhost:8000"

# Initialize session state variables.
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "current_question" not in st.session_state:
    st.session_state.current_question = ""
if "finished" not in st.session_state:
    st.session_state.finished = False

st.title("LangGraph Chatbot")

# Log the initial session state.
st.write("DEBUG: Initial session state", dict(st.session_state))
print("DEBUG: Initial session state", dict(st.session_state))

# Function to start a new conversation session via the API.
def start_session():
    st.write("DEBUG: Starting new session...")
    print("DEBUG: Starting new session...")
    response = requests.post(f"{API_URL}/start")
    
    # Log response details
    st.write("DEBUG: Response status code:", response.status_code)
    st.write("DEBUG: API /start response text:", response.text)
    print("DEBUG: Response status code:", response.status_code)
    print("DEBUG: API /start response text:", response.text)
    
    if response.status_code == 200:
        data = response.json()
        st.session_state.session_id = data.get("session_id")
        st.session_state.conversation_history = data.get("conversation_history", [])
        st.session_state.current_question = data.get("current_question", "")
        st.session_state.finished = False
        st.write("DEBUG: Updated session state after /start", dict(st.session_state))
        print("DEBUG: Updated session state after /start", dict(st.session_state))
        # Temporarily comment out rerun to see if state persists:
        st.rerun()
    else:
        st.error("Error starting session.")


# Function to send the user's answer to the API and update the conversation.
def send_answer(answer: str):
    st.write("DEBUG: Sending answer:", answer)
    print("DEBUG: Sending answer:", answer)
    payload = {"session_id": st.session_state.session_id, "answer": answer}
    response = requests.post(f"{API_URL}/next", json=payload)
    st.write("DEBUG: API /next response:", response.text)
    print("DEBUG: API /next response:", response.text)
    if response.status_code == 200:
        data = response.json()
        st.session_state.conversation_history = data["conversation_history"]
        st.session_state.current_question = data.get("current_question", "")
        st.session_state.finished = data.get("finished", False)
        st.write("DEBUG: Updated session state after sending answer", dict(st.session_state))
        print("DEBUG: Updated session state after sending answer", dict(st.session_state))
        st.rerun()
    else:
        st.error("Error processing answer.")

# Sidebar: Option to restart the conversation.
if st.sidebar.button("Restart Conversation"):
    st.session_state.session_id = None
    st.session_state.conversation_history = []
    st.session_state.current_question = ""
    st.session_state.finished = False
    st.write("DEBUG: Session restarted")
    print("DEBUG: Session restarted")
    st.rerun()

# If no session exists, show the "Start Conversation" button.
if st.session_state.session_id is None:
    if st.button("Start Conversation"):
        start_session()

# --- Chat Layout ---
# Display the current question and an input box.
if st.session_state.session_id and not st.session_state.finished:
    st.subheader("Current Question")
    st.markdown(f"**AI:** {st.session_state.current_question}")
    
    with st.form("answer_form", clear_on_submit=True):
        user_input = st.text_input("Your answer:")
        submitted = st.form_submit_button("Send Answer")
        if submitted:
            if user_input.strip() != "":
                send_answer(user_input)
            else:
                st.error("Please enter a valid answer.")

# Display the conversation history below the input box.
if st.session_state.conversation_history:
    st.subheader("Conversation History")
    for msg in st.session_state.conversation_history:
        if msg["role"] == "ai":
            st.markdown(f"**AI:** {msg['content']}")
        else:
            st.markdown(f"**You:** {msg['content']}")

# When the conversation finishes, display a finished message.
if st.session_state.finished:
    st.success("Conversation finished!")