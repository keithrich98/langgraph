import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables
load_dotenv()

# Initialize OpenAI model
model = ChatOpenAI(model="gpt-4o-mini")

# Set up Streamlit message history
history = StreamlitChatMessageHistory(key="chat_messages")

# Define the chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# Use a **fixed session ID** to maintain history
session_id = "persistent_session"

# Create a chain with memory (now properly using history)
chain_with_history = RunnableWithMessageHistory(
    prompt | model,
    lambda session_id: history,  # Ensure the same history object is used
    input_messages_key="question",
    history_messages_key="history",
)

# Streamlit app layout
st.title("Chatbot with Memory 🤖")
st.write("Ask me anything!")

# Display previous messages **before taking new input**
for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)

# Input for user question
if user_input := st.chat_input("Type your question here..."):
    st.chat_message("user").write(user_input)

    # Invoke the chain with the user's question
    response = chain_with_history.invoke(
        {"question": user_input}, {"configurable": {"session_id": session_id}}
    )

    # Display the AI's response
    st.chat_message("assistant").write(response.content)

    # Add the new messages to the history
    history.add_user_message(user_input)
    history.add_ai_message(response.content)
