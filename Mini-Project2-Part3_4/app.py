import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
from agents import Head_Agent
import time

def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

load_dotenv()
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

st.title("Mini Project 2: Streamlit Chatbot")

#initialize openai model
if "head_agent" not in st.session_state:
    st.session_state.head_agent = Head_Agent(
        openai_key = os.getenv("OPENAI_API_KEY"),
        pinecone_key = os.getenv("PINECONE_API_KEY"),
        pinecone_index_name = "machine-learning-textbook"
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    
    # add user input into session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # create result
    with st.chat_message("assistant"):
        with st.status("Thinking...", expanded=True) as status:
            response = st.session_state.head_agent.run(prompt)
            status.update(label="Finish", state="complete", expanded=False)
            
        st.write_stream(stream_data(response))
    
    st.session_state.messages.append({"role": "assistant", "content": response})
