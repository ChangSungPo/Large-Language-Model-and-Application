# Python

from dataclasses import dataclass, field
import json
import os
from tabnanny import verbose
import textwrap
import time

from openai import OpenAI
from schemas import get_market_status, get_top_gainers_losers, get_price_performance, get_news_sentiment, get_company_overview, get_tickers_by_sector, query_local_db, create_local_database

ALL_TOOL_FUNCTIONS = {
    "get_tickers_by_sector" : get_tickers_by_sector,
    "get_price_performance"  : get_price_performance,
    "get_company_overview"   : get_company_overview,
    "get_market_status"      : get_market_status,
    "get_top_gainers_losers" : get_top_gainers_losers,
    "get_news_sentiment"     : get_news_sentiment,
    "query_local_db"         : query_local_db,
}

# 4. TOOL SCHEMAS
def _s(name, desc, props, req):
    return {"type":"function","function":{
        "name":name,"description":desc,
        "parameters":{"type":"object","properties":props,"required":req}}}

SCHEMA_TICKERS  = _s("get_tickers_by_sector",
    "Return all stocks in a sector or industry from the local database. "
    "Use broad sector names ('Information Technology', 'Energy') or sub-sectors ('semiconductor', 'insurance').",
    {"sector":{"type":"string","description":"Sector or industry name"}}, ["sector"])

SCHEMA_PRICE    = _s("get_price_performance",
    "Get % price change for a list of tickers over a time period. "
    "Periods: '1mo','3mo','6mo','ytd','1y'.",
    {"tickers":{"type":"array","items":{"type":"string"}},
     "period":{"type":"string","default":"1y"}}, ["tickers"])

SCHEMA_OVERVIEW = _s("get_company_overview",
    "Get fundamentals for one stock: P/E ratio, EPS, market cap, 52-week high and low.",
    {"ticker":{"type":"string","description":"Ticker symbol e.g. 'AAPL'"}}, ["ticker"])

SCHEMA_STATUS   = _s("get_market_status",
    "Check whether global stock exchanges are currently open or closed.", {}, [])

SCHEMA_MOVERS   = _s("get_top_gainers_losers",
    "Get today's top gaining, top losing, and most actively traded stocks.", {}, [])

SCHEMA_NEWS     = _s("get_news_sentiment",
    "Get latest news headlines and Bullish/Bearish/Neutral sentiment scores for a stock.",
    {"ticker":{"type":"string"},"limit":{"type":"integer","default":5}}, ["ticker"])

SCHEMA_SQL      = _s("query_local_db",
    "Run a SQL SELECT on stocks.db. "
    "Table 'stocks': ticker, company, sector, industry, market_cap (Large/Mid/Small), exchange.",
    {"sql":{"type":"string","description":"A valid SQL SELECT statement"}}, ["sql"])

ALL_SCHEMAS = [
    SCHEMA_TICKERS, SCHEMA_PRICE, SCHEMA_OVERVIEW,
    SCHEMA_STATUS, SCHEMA_MOVERS, SCHEMA_NEWS, SCHEMA_SQL
]

MARKET_TOOLS      = [SCHEMA_TICKERS, SCHEMA_PRICE, SCHEMA_STATUS, SCHEMA_MOVERS]
FUNDAMENTAL_TOOLS = [SCHEMA_OVERVIEW, SCHEMA_SQL, SCHEMA_TICKERS]
SENTIMENT_TOOLS   = [SCHEMA_NEWS, SCHEMA_TICKERS]

@dataclass
class AgentResult:
    agent_name   : str
    answer       : str
    tools_called : list  = field(default_factory=list)
    raw_data     : dict  = field(default_factory=dict)
    confidence   : float = 0.0
    issues_found : list  = field(default_factory=list)
    reasoning    : str   = ""

def run_specialist_agent(
    client       : OpenAI,
    agent_name   : str,
    system_prompt: str,
    task         : str,
    tool_schemas : list,
    model        : str,
    max_iters    : int  = 8,
    verbose      : bool = False, # Set to False for Streamlit to avoid verbose logging
) -> AgentResult:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]
    tools_called = []
    raw_data = {}

    for i in range(max_iters):
        if verbose: print(f"  [{agent_name}] iteration {i+1}/{max_iters}")
        if tool_schemas:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tool_schemas,
                tool_choice="auto",
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )

        response_message = response.choices[0].message

        if response_message.tool_calls:
            tool_calls = response_message.tool_calls
            messages.append(response_message)

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                if verbose: print(f"  [{agent_name}] calling tool: {function_name} with args {function_args}")

                if function_name not in ALL_TOOL_FUNCTIONS:
                    tool_output = {"error": f"Tool {function_name} not found."}
                else:
                    tool_function = ALL_TOOL_FUNCTIONS[function_name]
                    try:
                        tool_output = tool_function(**function_args)
                    except Exception as e:
                        tool_output = {"error": f"Error executing tool {function_name}: {e}"}

                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(tool_output),
                    }
                )
                tools_called.append(function_name)
                raw_data[function_name] = tool_output
        else:
            final_answer = response_message.content
            if verbose: print(f"  [{agent_name}] returning final answer.")
            return AgentResult(
                agent_name=agent_name,
                answer=final_answer,
                tools_called=tools_called,
                raw_data=raw_data,
            )

    return AgentResult(
        agent_name=agent_name,
        answer="Max iterations reached without providing a final answer.",
        tools_called=tools_called,
        raw_data=raw_data,
        issues_found=["Max iterations reached"]
    )

# Single Agent Implementation
SINGLE_AGENT_PROMPT = """
You are a skilled financial analyst. Your primary goal is to provide accurate and comprehensive answers to financial questions using the available tools.

Here are your rules:
1. Always prioritize using the provided tools to gather factual information. Do not rely on your internal knowledge for current market data, prices, or company fundamentals.
2. If a question requires data that can be obtained from a tool, you must call that tool.
3. For questions involving multiple steps (e.g., 'find tech stocks, then get their performance'), plan your steps carefully and chain tool calls as necessary.
4. When a tool returns an error or no data, clearly state this in your answer. Do not fabricate information or make assumptions.
5. If asked about market status, use `get_market_status`.
6. If asked about top gainers/losers or most active stocks, use `get_top_gainers_losers`.
7. If asked for price performance over a period, use `get_price_performance` with the correct period parameter ('1mo', '3mo', '6mo', 'ytd', '1y').
8. If asked about P/E ratio, EPS, market cap, or 52-week high/low for a single company, use `get_company_overview`.
9. If asked about news sentiment, use `get_news_sentiment`.
10. If asked to find companies by sector, industry, market cap, or exchange, use `get_tickers_by_sector` or `query_local_db` with appropriate SQL. `get_tickers_by_sector` is preferred for simple sector/industry lookups.
11. If a question can be answered by running a SQL query on the `stocks` database (e.g., filtering by market cap, exchange, or specific industry details not covered by `get_tickers_by_sector`), use `query_local_db`.
12. Always synthesize the information from tool calls into a clear, concise, and accurate answer. Cite the data you found. If you cannot answer the question even after using tools, explain why.
"""

def run_single_agent(client: OpenAI, question: str, model: str, verbose: bool = False) -> AgentResult:
    return run_specialist_agent(
        client,
        agent_name="Single Agent",
        system_prompt=SINGLE_AGENT_PROMPT,
        task=question,
        tool_schemas=ALL_SCHEMAS,
        model=model,
        max_iters=10,
        verbose=verbose,
    )

# Multi-Agent Implementation
# ── YOUR MULTI-AGENT IMPLEMENTATION ──────────────────────────
#
# Architecture chosen: [Orchestrator + Specialists + Critic]
# Reason: [explain why you chose this over the alternatives]
# In the stock market, user queries are typically vague and cross-domain. For example: "This stock has plummeted recently—are its fundamentals still solid?"
# Sequential Pipeline: This approach is highly vulnerable to error propagation. A mistake in the initial step creates a "domino effect," where a single hallucination or data error ruins every subsequent process.
# Parallel Specialists: While fast, this model fails to grasp logical dependencies. It struggles to synthesize information when one task's result should fundamentally influence how another task is performed.
# Orchestrator + Critic: This is our chosen architecture. The Orchestrator effectively decomposes a complex query into independent sub-tasks, such as "Price Momentum" and "Fundamental Analysis." 
# Finally, a Critic cross-references the specialist's findings against the Raw Data to ensure that every stock price and financial metric is 100% accurate and hallucination-free.
#
# Specialist breakdown:
#   Agent 1 — [Market Analyst, Market, SCHEMA_PRICE, SCHEMA_MOVERS, SCHEMA_STATUS]
#   Agent 2 — [Fundamental Researcher, Foundamental, SCHEMA_OVERVIEW, SCHEMA_NEWS]
#   Agent 3 — [Database Clerk, Sentiment or Database, SCHEMA_TICKERS, SCHEMA_SQL]
#
# Verification mechanism: [how does your system check answer quality?]
# Our Critic Agent operates without tools, focusing entirely on cross-referencing Specialist answers with Raw Tool Data. 
# If an Agent hallucinates a number (e.g., reporting a P/E of 15 when the data shows 20), the Critic flags the error in issues_found and lowers the overall confidence score.
#
### YOUR CODE HERE
PROMPT_MARKET_ANALYST = """
You are a Market Analyst. 
### STRICT BEHAVIORAL CODE:
1. EVIDENCE ONLY: You are strictly forbidden from stating any stock price or percentage change unless it was returned by a tool in the current session.
2. REFUSAL POLICY: If a tool returns an error or no data, your answer must be: "I cannot provide this data because the tool returned no results." Do NOT estimate or use outdated knowledge.
3. CITATION: Every number you state must be followed by its source tool name in brackets, e.g., "AAPL rose 2.5% [get_price_performance]".
4. NO FLUFF: Do not provide general market commentary unless it is directly supported by the retrieved news or price data.
"""

PROMPT_FUNDAMENTAL_RESEARCHER = """
You are a Fundamental Researcher.
### STRICT BEHAVIORAL CODE:
1. DATA INTEGRITY: If the user asks for a P/E ratio and the tool fails, state "Data unavailable." NEVER say "The P/E ratio is approximately X."
2. NO HALLUCINATION: You do not know anything about 2024, 2025, or 2026 unless the tools (get_news_sentiment or get_company_overview) tell you. 
3. VERIFICATION: Compare the 52-week high/low from the tool against the current price. If there is a contradiction in the tool data, report the contradiction instead of picking a number.
4. SOURCE TAGGING: Label every financial metric with its source tool.
"""

PROMPT_DATABASE_CLERK = """
You are a Data Engineer and Database Specialist.
Your goal is to provide accurate lists of companies from the local S&P 500 database.

TASKS:
1. Find tickers by sector or industry using 'get_tickers_by_sector'.
2. Execute complex SQL queries on 'stocks.db' if the user has specific filtering needs.

RULES:
- The table name is 'stocks'. Columns: ticker, company, sector, industry, market_cap, exchange.
- If a sector search returns nothing, try a broader term or a partial match via SQL LIKE.
- Return structured lists that other agents can easily use.
"""

PROMPT_CRITIC = """
You are a Zero-Tolerance Financial Auditor. 
Your sole mission is to verify if the Specialist's answer is 100% grounded in the provided Raw Tool Data.

### AUDIT RULES:
1. NO RAW DATA = NO NUMBERS: If the Raw Tool Data is empty ({}, []), but the Specialist provides specific numbers (e.g., P/E ratio, Price, %), you MUST mark [Verdict: Fail].
2. DATA CONSISTENCY: Every decimal point matters. If the Specialist says "up 2.5%" but the Raw Data says "2.1%", mark [Verdict: Fail].
3. NO OUTSIDE KNOWLEDGE: Specialists are forbidden from using their internal training data for "current" metrics. If they provide data not found in the JSON, they have failed.

### OUTPUT FORMAT (Strictly follow this structure):
- Confidence Score: (0.0 to 1.0)
- Issues Found: (Detail exactly which numbers or claims are unsupported. If none, say "None")
- Verdict: [Pass] or [Fail]
- Feedback for Specialist: (If Failed, give a 1-sentence instruction on how to fix it, e.g., "Stop hallucinating the P/E ratio; the tool returned no data.")
"""

def run_multi_agent(client: OpenAI, user_question: str, model: str, verbose: bool = False) -> dict:
    start_time = time.time()
    MAX_RETRIES = 2
    print(f"[User Question]: {user_question}\n")
    
    # --- 1. Orchestration Phase
    planning_prompt = f"""
        You are the Lead Investment Strategist. Your primary responsibility is TASK DECOMPOSITION and TOOL ENFORCEMENT.

        ### MANDATORY PLANNING RULES:
        1. NEVER ANSWER DIRECTLY: You are a planner, not a researcher. You must NOT provide any financial data yourself.
        2. FORCE DELEGATION: Every user question must result in at least ONE task for a specialist. If the question involves data, you MUST use a tool.
        3. TOOL-TASK LINKING: Explicitly name the tool in the task (e.g., "Use get_company_overview for AAPL P/E ratio").
        4. NO EMPTY PLANS: Returning an empty list [] is a total failure.

        ### SPECIALIST ROUTING GUIDE:
        
        **Market Analyst** → Price & Market Status
        - Task: Get price performance over time periods (1mo, 3mo, 6mo, ytd, 1y)
        - Task: Get today's top gainers/losers or most active stocks
        - Task: Check if exchanges are open/closed
        - Tools: get_price_performance, get_top_gainers_losers, get_market_status
        
        **Fundamental Researcher** → Company Fundamentals & News (ALWAYS use for P/E, EPS, Market Cap, 52-week measurements)
        - Task: Get P/E ratio, EPS, Market Capitalization, 52-week high/low for a ticker
        - Task: Get news sentiment and headlines for a ticker
        - Tools: get_company_overview, get_news_sentiment
        
        **Database Clerk** → Local Database Searches Only
        - Task: Find all companies in a specific sector or industry
        - Task: Filter companies by market cap category (Large/Mid/Small)
        - Task: Execute custom SQL queries on the S&P 500 database
        - Tools: get_tickers_by_sector, query_local_db

        User Question: "{user_question}"

        ### OUTPUT FORMAT:
        Return a JSON object with a key "tasks" containing a list of task objects.
        Example: {{ "tasks": [ {{"agent": "...", "task": "..."}} ] }}
    """
    
    plan_response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": planning_prompt}],
        response_format={"type": "json_object"} 
    )
    plan = json.loads(plan_response.choices[0].message.content).get("tasks", [])
    print("MA plan: ")
    print(plan)
    
    specialist_results = []
    
    # --- 2. Execution Phase:
    agent_config = {
        "Market Analyst": {"prompt": PROMPT_MARKET_ANALYST, "tools": [SCHEMA_PRICE, SCHEMA_MOVERS, SCHEMA_STATUS]},
        "Fundamental Researcher": {"prompt": PROMPT_FUNDAMENTAL_RESEARCHER, "tools": [SCHEMA_OVERVIEW, SCHEMA_NEWS]},
        "Database Clerk": {"prompt": PROMPT_DATABASE_CLERK, "tools": [SCHEMA_TICKERS, SCHEMA_SQL]}
    }

    for item in plan:
        agent_name = item["agent"]
        original_task = item["task"]
        current_task = original_task
        
        if agent_name in agent_config:
            attempts = 0
            success = False
            final_res = None

            while attempts <= MAX_RETRIES and not success:
                attempts += 1
                print(f"[Orchestrator][{attempts}] Assigning to {agent_name} (Attempt {attempts}): {current_task}")

                res = run_specialist_agent(
                    client=client,
                    agent_name=agent_name,
                    system_prompt=agent_config[agent_name]["prompt"],
                    task=current_task,
                    tool_schemas=agent_config[agent_name]["tools"],
                    model=model,
                    verbose=verbose
                )

                print(f"[Critic][{attempts}] Checking {agent_name}'s work...")

                verification_input = f"""
                User Question: {user_question}
                Specialist Answer: {res.answer}
                Raw Tool Data Used: {json.dumps(res.raw_data)}
                """

                critic_res = run_specialist_agent(
                    client=client,
                    agent_name="Critic",
                    system_prompt=PROMPT_CRITIC,
                    task=verification_input,
                    tool_schemas=[],
                    model=model,
                    verbose=verbose
                )

                res.reasoning = critic_res.answer

                if "Pass" in critic_res.answer:
                    print(f"✅ {agent_name} passed verification.")
                    res.confidence = 0.95
                    success = True
                    final_res = res
                else:
                    print(f"❌ {agent_name} failed verification. Critic feedback: {critic_res.answer}")
                    current_task = f"""
                    Your previous answer was rejected by the auditor.
                    CRITIC FEEDBACK: {critic_res.answer}
                    Please redo the task: {original_task}
                    STRICT REQUIREMENT: Fix the issues mentioned and use ONLY the numbers found in the RAW JSON.
                    """
                    res.confidence = 0.2
                    final_res = res
                
            specialist_results.append(final_res)

    # --- 4. Synthesis Phase:
    print(f"[Synthesizer] Crafting final report...")
    
    # Put all answers from Specialists and combine with Critic give to Synthesizer
    all_context = "\n\n".join([
        f"Specialist: {r.agent_name}\nData: {r.answer}\nVerification: {r.reasoning}" 
        for r in specialist_results
    ])
    
    synthesis_prompt = f"""
    You are a Senior Investment Strategist. 
    Your goal is to synthesize reports from multiple specialists into one final answer.

    ### SPECIALIST REPORTS TO ANALYZE:
    {all_context}

    ### INTEGRITY GUIDELINES:
    1. NO DATA, NO GUESSING: If a specialist was unable to retrieve data (Status: FAILED), do NOT make up an answer. Instead, say: "Currently, I do not have access to the specific metric."
    2. VERIFIED DATA ONLY: Only include information that has been marked as VERIFIED.
    3. SEAMLESS INTEGRATION: Do not mention internal agent names. 

    Original User Question: {user_question}
    """
    
    final_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a professional financial synthesizer."},
            {"role": "user", "content": synthesis_prompt}
        ]
    )
    
    final_answer = final_response.choices[0].message.content
    elapsed_sec = round(time.time() - start_time, 2)
    print(final_answer)
    print(specialist_results)

    return {
        "final_answer"  : final_answer,
        "agent_results" : specialist_results,
        "elapsed_sec"   : elapsed_sec,
        "architecture"  : "orchestrator-critic"
    }