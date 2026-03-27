# Import the necessary libraries
import os
import sqlite3
from dotenv import load_dotenv  
import streamlit as st
from openai import OpenAI  
from agents import run_single_agent, run_multi_agent

st.title("Mini Project 3: Streamlit FinAgent")
MODEL_SMALL  = "gpt-4o-mini"
MODEL_LARGE  = "gpt-4o"
ACTIVE_MODEL = MODEL_SMALL

# TODO: Replace with your actual OpenAI API key
load_dotenv()
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
ALPHAVANTAGE_API_KEY = os.getenv('ALPHAAVANTAGE_API_KEY')
DB_PATH = "stocks.db"
CSV_PATH = "sp500_companies.csv"

def create_local_database(csv_path=CSV_PATH):
    if not os.path.exists(csv_path):
        st.error(f"'{csv_path}' not found. Please place the Kaggle CSV in the same folder.")
        return
    if os.path.exists(DB_PATH): return # Only create if it doesn't exist
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={
        "symbol":"ticker", "shortname":"company",
        "sector":"sector",  "industry":"industry",
        "exchange":"exchange", "marketcap":"market_cap_raw"
    })
    def cap_bucket(v):
        try:
            v = float(v)
            return "Large" if v >= 10_000_000_000 else "Mid" if v >= 2_000_000_000 else "Small"
        except: return "Unknown"
    df["market_cap"] = df["market_cap_raw"].apply(cap_bucket)
    df = (df.dropna(subset=["ticker","company"])
            .drop_duplicates(subset=["ticker"])
            [["ticker","company","sector","industry","market_cap","exchange"]])
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("stocks", conn, if_exists="replace", index=False)
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_ticker ON stocks(ticker)")
    conn.commit()
    conn.close()

# Define a function to get the conversation history (Not required for Part-2, will be useful in Part-3)
def get_conversation() -> str:
    history = ""
    for message in st.session_state.messages:
        role = message["role"].capitalize()
        content = message["content"]
        history += f"{role}: {content}\n"
    return history

create_local_database()
# STREAMLIT UI
st.set_page_config(page_title="FinTech AI", layout="wide")
with st.sidebar:
    st.title("Config")
    agent_type = st.selectbox("Architecture", ["Single Agent", "Multi-Agent"])
    model_choice = st.selectbox("Model", [MODEL_SMALL, MODEL_LARGE])
    if st.button("Clear Chat"): st.session_state.messages = []; st.rerun()

if "messages" not in st.session_state: st.session_state.messages = []
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "meta" in m: st.caption(f"Mode: {m['meta']}")

if prompt := st.chat_input("Query..."):
    st.chat_message("user").markdown(prompt)
    history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:]])
    full_query = f"History:\n{history}\n\nQuestion: {prompt}"
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if agent_type == "Single Agent":
                res = run_single_agent(client, full_query, model_choice)
                st.markdown(res.answer)
                st.session_state.messages.append({"role": "assistant", "content": res.answer, "meta": f"{agent_type} | {model_choice}"})
            else:
                multi_agent_output = run_multi_agent(client, full_query, model_choice)
                st.markdown(multi_agent_output["final_answer"])
                st.session_state.messages.append({"role": "assistant", "content": multi_agent_output["final_answer"], "meta": f"{agent_type} | {model_choice}"})

# # Wait for user input
# prompt = st.chat_input("What would you like to chat about?")
# if prompt:
#     # ... (append user message to messages)
#     st.session_state.messages.append({"role": "user", "content": prompt})

#     # ... (display user message)
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # Generate AI response
#     with st.chat_message("assistant"):
#         with st.status("Agent is thinking...", expanded=True) as status:
#             st.write("Checking query safety...")
            
#             # Update the agent's internal state
#             st.session_state.head_agent.latest_user_query = prompt
#             # Pass the current history so the Rewriter/Answerer can see it
#             st.session_state.head_agent.history = st.session_state.messages[:-1]
            
#             # Run the main loop
#             full_response = st.session_state.head_agent.main_loop()
            
#             status.update(label="Response generated!", state="complete", expanded=False)
        
#         # Display the final answer
#         st.markdown(full_response)

#     # ... (append AI response to messages)
#     st.session_state.messages.append({"role": "assistant", "content": full_response})
