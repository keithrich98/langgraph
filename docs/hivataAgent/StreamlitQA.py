# StreamlitQA.py - Updated with streaming visualization

import streamlit as st
import requests
import json
import time

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
if "streaming_log" not in st.session_state:
    st.session_state.streaming_log = []

st.title("LangGraph Chatbot")

# Function to start a new conversation session via the API.
def start_session():
    st.write("DEBUG: Starting new session...")
    print("DEBUG: Starting new session...")
    response = requests.post(f"{API_URL}/start")
    
    # Log response details
    st.write("DEBUG: Response status code:", response.status_code)
    print("DEBUG: Response status code:", response.status_code)
    
    if response.status_code == 200:
        data = response.json()
        st.session_state.session_id = data.get("session_id")
        st.session_state.conversation_history = data.get("conversation_history", [])
        st.session_state.current_question = data.get("current_question", "")
        st.session_state.finished = False
        st.session_state.streaming_log = []
        print("DEBUG: Updated session state after /start", dict(st.session_state))
        st.rerun()
    else:
        st.error("Error starting session.")

# Function to send the user's answer to the API and update the conversation.
def send_answer(answer: str):
    if not st.session_state.session_id:
        st.error("No active session. Please start a new conversation.")
        return
        
    print("DEBUG: Sending answer:", answer)
    payload = {"session_id": st.session_state.session_id, "answer": answer}
    
    try:
        # First, add the user message to the conversation history immediately
        # so it appears right away in the UI
        st.session_state.conversation_history.append({"role": "human", "content": answer})
        
        # Create a placeholder for streaming content and a progress bar
        response_container = st.empty()
        streaming_indicator = st.empty()
        streaming_text = ""
        
        # Reset streaming log
        st.session_state.streaming_log = []
        
        # Try streaming first
        try:
            with requests.post(f"{API_URL}/stream-next", json=payload, stream=True, timeout=60) as response:
                if response.status_code != 200:
                    raise Exception(f"Stream request failed with status {response.status_code}")
                
                # Show streaming indicator
                streaming_indicator.info("⏳ Streaming in progress...")
                
                # Create columns for different elements
                col1, col2 = st.columns([3, 1])
                token_counter = col2.empty()
                token_count = 0
                
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            try:
                                data = json.loads(line[6:])
                                
                                if 'text' in data:
                                    # Append streaming token to the text
                                    streaming_text += data['text']
                                    response_container.markdown(f"**AI:** {streaming_text}")
                                    
                                    # Update token counter
                                    token_count = data.get('token_num', token_count + 1)
                                    token_counter.metric("Tokens", token_count)
                                    
                                    # Add to streaming log for debug view
                                    token_data = {
                                        "token": data['text'],
                                        "number": data.get('token_num', token_count)
                                    }
                                    st.session_state.streaming_log.append(token_data)
                                
                                if data.get('final', False):
                                    if 'error' in data:
                                        st.error(f"Error: {data['error']}")
                                        # Fall back to non-streaming endpoint
                                        raise Exception("Streaming failed with error")
                                    
                                    # Show completion message
                                    streaming_indicator.success(f"✅ Streaming complete! Received {data.get('total_tokens', token_count)} tokens")
                                    
                                    # Update session state with final data
                                    st.session_state.conversation_history = data["conversation_history"]
                                    st.session_state.current_question = data.get("current_question", "")
                                    st.session_state.finished = data.get("finished", False)
                                    print("DEBUG: Updated session state after streaming")
                                    st.rerun()
                                    return
                            except json.JSONDecodeError:
                                print(f"Failed to decode JSON: {line[6:]}")
                                continue
                
                # If we get here, we didn't get a final message
                streaming_indicator.warning("⚠️ Stream did not complete properly")
                raise Exception("Stream did not complete properly")
                
        except Exception as e:
            print(f"Streaming failed: {str(e)}. Falling back to regular endpoint.")
            # Clear the streaming response
            response_container.empty()
            streaming_indicator.error(f"❌ Streaming failed: {str(e)}. Falling back to regular endpoint.")
            
        # Fallback to non-streaming endpoint
        response = requests.post(f"{API_URL}/next", json=payload)
        if response.status_code == 200:
            data = response.json()
            st.session_state.conversation_history = data["conversation_history"]
            st.session_state.current_question = data.get("current_question", "")
            st.session_state.finished = data.get("finished", False)
            print("DEBUG: Updated session state after falling back to /next")
            st.rerun()
        else:
            st.error(f"Error processing answer. Status code: {response.status_code}")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Sidebar: Option to restart the conversation.
if st.sidebar.button("Restart Conversation"):
    st.session_state.session_id = None
    st.session_state.conversation_history = []
    st.session_state.current_question = ""
    st.session_state.finished = False
    st.session_state.streaming_log = []
    print("DEBUG: Session restarted")
    st.rerun()

# Show debug info in sidebar
with st.sidebar:
    st.subheader("Debug Info")
    if st.checkbox("Show session state"):
        st.json(dict((k, str(v)[:100] if isinstance(v, list) and len(str(v)) > 100 else str(v)) 
                     for k, v in st.session_state.items()))
    
    if st.session_state.session_id:
        if st.button("Get Debug Info"):
            debug_response = requests.get(f"{API_URL}/debug/{st.session_state.session_id}")
            if debug_response.status_code == 200:
                st.json(debug_response.json())
            else:
                st.error("Failed to get debug info")
    
    # Show streaming debug logs
    if st.session_state.streaming_log:
        st.subheader("Streaming Log")
        st.write(f"Total tokens received: {len(st.session_state.streaming_log)}")
        
        if st.checkbox("Show token-by-token log"):
            log_text = ""
            for item in st.session_state.streaming_log:
                log_text += f"Token #{item['number']}: '{item['token']}'\n"
            st.code(log_text, language="text")

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