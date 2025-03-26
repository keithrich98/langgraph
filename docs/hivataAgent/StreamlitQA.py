# StreamlitQA.py - Updated with streaming visualization and term extraction support

import streamlit as st
import requests
import json
import time
import pandas as pd

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
if "extracted_terms" not in st.session_state:
    st.session_state.extracted_terms = {}
if "last_terms_check" not in st.session_state:
    st.session_state.last_terms_check = 0

st.title("Medical Questionnaire with Term Extraction")

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
        st.session_state.extracted_terms = {}
        st.session_state.last_terms_check = 0
        print("DEBUG: Updated session state after /start", dict(st.session_state))
        st.rerun()
    else:
        st.error("Error starting session.")

# Function to fetch extracted terms from the API
def fetch_extracted_terms():
    if not st.session_state.session_id:
        return None
    
    try:
        response = requests.get(f"{API_URL}/extracted-terms/{st.session_state.session_id}")
        if response.status_code == 200:
            terms_data = response.json()
            st.session_state.extracted_terms = terms_data.get("extracted_terms", {})
            st.session_state.last_terms_check = time.time()
            return terms_data
        else:
            print(f"Failed to fetch terms. Status code: {response.status_code}")
            return {"extracted_terms": {}, "error": f"Status code: {response.status_code}"}
    except Exception as e:
        print(f"Error fetching terms: {str(e)}")
        return {"extracted_terms": {}, "error": str(e)}


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
                                    
                                    # Set a timer to check for extracted terms
                                    st.session_state.last_terms_check = 0  # Force check on next refresh
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
            
            # Set a timer to check for extracted terms
            st.session_state.last_terms_check = 0  # Force check on next refresh
            st.rerun()
        else:
            st.error(f"Error processing answer. Status code: {response.status_code}")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Function to display extracted terms
def display_extracted_terms(terms_data):
    if not terms_data or not terms_data.get("extracted_terms"):
        st.info("No medical terms have been extracted yet.")
        return
    
    extracted_terms = terms_data.get("extracted_terms", {})
    if not extracted_terms:
        st.info("No medical terms have been extracted yet.")
        return
    
    st.subheader("📋 Extracted Medical Terms")
    
    # Create a nice table display of terms organized by question
    for q_idx, terms in extracted_terms.items():
        # Find the original question and response from conversation history
        question_text = f"Question {int(q_idx) + 1}"
        answer_text = ""
        
        # Try to find the actual question text
        for i, msg in enumerate(st.session_state.conversation_history):
            if msg["role"] == "ai" and i//2 == int(q_idx):
                question_text = msg["content"].split("\n")[0]
                break
        
        # Try to find the user answer
        for i, msg in enumerate(st.session_state.conversation_history):
            if msg["role"] == "human" and i//2 == int(q_idx):
                answer_text = msg["content"]
                break
        
        with st.expander(f"Q{int(q_idx) + 1}: {question_text[:100]}...", expanded=True):
            if answer_text:
                st.markdown(f"**Answer**: {answer_text}")
            
            # Display terms in a nice grid layout
            if terms:
                # Create columns to display terms in a grid
                cols = st.columns(3)
                for i, term in enumerate(terms):
                    cols[i % 3].markdown(f"• **{term}**")
            else:
                st.info("Processing... No terms extracted yet.")
    
    # Show extraction queue details
    queue_length = terms_data.get("extraction_queue_length", 0)
    if queue_length > 0:
        st.info(f"⏳ {queue_length} answer(s) still in extraction queue...")
    else:
        st.success("✅ All answers processed!")

# Create a layout with two columns
left_col, right_col = st.columns([2, 1])

# Create tabs on the right column for better organization
with right_col:
    tabs = st.tabs(["Medical Terms", "Debug Info", "Memory"])
    
    # Medical Terms Tab
    with tabs[0]:
        st.subheader("Medical Term Extraction")
        
        if st.session_state.session_id:
            # Check if it's time to refresh terms (every 5 seconds)
            current_time = time.time()
            if current_time - st.session_state.last_terms_check > 5:
                terms_data = fetch_extracted_terms()
                st.session_state.last_terms_check = current_time
            else:
                terms_data = {"extracted_terms": st.session_state.extracted_terms}
            
            # Display the extracted terms
            display_extracted_terms(terms_data)
            
            # Add manual refresh button
            if st.button("Refresh Terms", use_container_width=True):
                terms_data = fetch_extracted_terms()
                st.rerun()
            
            # Auto-refresh checkbox
            if st.checkbox("Auto-refresh every 5 seconds", value=False):
                st.empty()
                time.sleep(5)
                fetch_extracted_terms()
                st.rerun()
        else:
            st.info("Start a conversation to see extracted medical terms.")
    
    # Debug Tab
    with tabs[1]:
        st.subheader("Debug Information")
        if st.checkbox("Show session state"):
            st.json(dict((k, str(v)[:100] if isinstance(v, list) and len(str(v)) > 100 else str(v)) 
                        for k, v in st.session_state.items()))
        
        if st.session_state.session_id:
            if st.button("Debug Extraction Queue"):
                debug_response = requests.post(f"{API_URL}/debug-extraction-queue/{st.session_state.session_id}")
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
    
    # Memory Tab
    with tabs[2]:
        st.subheader("Model Context Inspector")
        st.write("Inspect what the verification model sees when processing your answers.")
        
        if st.session_state.session_id:
            question_index = st.number_input("Question Index (leave at -1 for current):", 
                                           min_value=-1, 
                                           max_value=10, 
                                           value=-1)
            
            if st.button("Inspect Model Context", key="inspect_context_btn"):
                # Convert -1 to None for API to use current question index
                idx_param = None if question_index == -1 else question_index
                context_url = f"{API_URL}/inspect-model-context/{st.session_state.session_id}"
                if idx_param is not None:
                    context_url += f"?question_index={idx_param}"
                    
                with st.spinner("Retrieving model context..."):
                    context_response = requests.get(context_url)
                    if context_response.status_code == 200:
                        context_data = context_response.json()
                        
                        # Create cols for token info display
                        col1, col2, col3 = st.columns(3)
                        token_info = context_data.get("token_estimates", {})
                        
                        col1.metric("Total Tokens", token_info.get('total_tokens', 'Unknown'))
                        col2.metric("System Prompt Tokens", token_info.get('system_tokens', 'N/A'))
                        col3.metric("User Prompt Tokens", token_info.get('user_tokens', 'N/A'))
                        
                        # Display system prompt
                        with st.expander("System Prompt", expanded=False):
                            st.code(context_data.get("system_prompt", ""), language="text")
                        
                        # Display user prompt (with conversation history)
                        with st.expander("User Prompt (including conversation history)", expanded=True):
                            st.code(context_data.get("user_prompt", ""), language="text")
                        
                        # Show question information
                        st.subheader("Question Information")
                        st.markdown(f"**Current Question:** {context_data.get('question', 'Unknown')}")
                        st.markdown(f"**Question Index:** {context_data.get('current_question_index', 'Unknown')}")
                        
                        # Additional state information
                        with st.expander("Additional State Information", expanded=False):
                            st.json(context_data.get("state_info", {}))
                    else:
                        st.error(f"Failed to get model context. Status code: {context_response.status_code}")
                        if context_response.text:
                            st.code(context_response.text, language="text")
        else:
            st.info("Start a conversation to inspect the model context.")

# Main interaction area in the left column
with left_col:
    # Sidebar options
    if st.button("Restart Conversation"):
        st.session_state.session_id = None
        st.session_state.conversation_history = []
        st.session_state.current_question = ""
        st.session_state.finished = False
        st.session_state.streaming_log = []
        st.session_state.extracted_terms = {}
        print("DEBUG: Session restarted")
        st.rerun()
    
    # If no session exists, show the "Start Conversation" button.
    if st.session_state.session_id is None:
        st.info("This application demonstrates a medical questionnaire with asynchronous medical term extraction. Terms extracted from your answers will appear in the right panel.")
        st.markdown("""
        ### How it works:
        1. Answer the questions about polymicrogyria
        2. Your answers are verified by an LLM
        3. After verification, medical terms are extracted asynchronously
        4. View the extracted terms in the right panel
        """)
        
        if st.button("Start Questionnaire", use_container_width=True):
            start_session()

    # --- Chat Layout ---
    # Display the current question and an input box.
    if st.session_state.session_id and not st.session_state.finished:
        st.subheader("Current Question")
        st.markdown(f"**AI:** {st.session_state.current_question}")
        
        with st.form("answer_form", clear_on_submit=True):
            user_input = st.text_area("Your answer:", height=100)
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
        st.success("Questionnaire completed! Check the Medical Terms tab to see all extracted terms.")