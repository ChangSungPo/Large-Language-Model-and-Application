import os
DB_PATH = "stocks.db"
# AV_BASE = "https://www.alphavantage.co"
AV_BASE = "http://localhost:2345"
import pandas as pd
import sqlite3
import requests
ALPHAVANTAGE_API_KEY = os.getenv('ALPHAAVANTAGE_API_KEY')

def create_local_database(csv_path: str = "sp500_companies.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"'{csv_path}' not found.\n"
            "Download from: https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks"
        )
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
    n = pd.read_sql_query("SELECT COUNT(*) AS n FROM stocks", conn).iloc[0]["n"]
    print(f"✅ {n} companies loaded into stocks.db")
    print("\nDistinct sector values stored in DB:")
    print(pd.read_sql_query("SELECT DISTINCT sector FROM stocks ORDER BY sector", conn).to_string(index=False))
    conn.close()

# ── Tool 1 ── Provided ────────────────────────────────────────
def get_price_performance(tickers: list, period: str = "1y") -> dict:
    """
    % price change for a list of tickers over a period.
    Valid periods: '1mo', '3mo', '6mo', 'ytd', '1y'
    Returns: { TICKER: {start_price, end_price, pct_change, period} }
    """
    results = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if data.empty:
                results[ticker] = {"error": "No data — possibly delisted"}
                continue
            start = float(data["Close"].iloc[0].item())
            end   = float(data["Close"].iloc[-1].item())
            results[ticker] = {
                "start_price": round(start, 2),
                "end_price"  : round(end,   2),
                "pct_change" : round((end - start) / start * 100, 2),
                "period"     : period,
            }
        except Exception as e:
            results[ticker] = {"error": str(e)}
    return results

# ── Tool 2 ── Provided ────────────────────────────────────────
def get_market_status() -> dict:
    """Open / closed status for global stock exchanges."""
    return requests.get(
        f"{AV_BASE}/query?function=MARKET_STATUS"
        f"&apikey={ALPHAVANTAGE_API_KEY}", timeout=10
    ).json()

# ── Tool 3 ── Provided ────────────────────────────────────────
def get_top_gainers_losers() -> dict:
    """Today's top gaining, top losing, and most active tickers."""
    return requests.get(
        f"{AV_BASE}/query?function=TOP_GAINERS_LOSERS"
        f"&apikey={ALPHAVANTAGE_API_KEY}", timeout=10
    ).json()

# ── Tool 4 ── Provided ────────────────────────────────────────
def get_news_sentiment(ticker: str, limit: int = 5) -> dict:
    """
    Latest headlines + Bullish / Bearish / Neutral sentiment for a ticker.
    Returns: { ticker, articles: [{title, source, sentiment, score}] }
    """
    data = requests.get(
        f"{AV_BASE}/query?function=NEWS_SENTIMENT"
        f"&tickers={ticker}&limit={limit}&apikey={ALPHAVANTAGE_API_KEY}", timeout=10
    ).json()
    return {
        "ticker": ticker,
        "articles": [
            {
                "title"    : a.get("title"),
                "source"   : a.get("source"),
                "sentiment": a.get("overall_sentiment_label"),
                "score"    : a.get("overall_sentiment_score"),
            }
            for a in data.get("feed", [])[:limit]
        ],
    }

# ── Tool 5 ── Provided ────────────────────────────────────────
def query_local_db(sql: str) -> dict:
    """
    Run any SQL SELECT on stocks.db.
    Table 'stocks' columns: ticker, company, sector, industry, market_cap, exchange
    market_cap values: 'Large' | 'Mid' | 'Small'
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql_query(sql, conn)
        conn.close()
        return {"columns": list(df.columns), "rows": df.to_dict(orient="records")}
    except Exception as e:
        return {"error": str(e)}

# ── Tool 6 — YOUR IMPLEMENTATION ─────────────────────────────
def get_company_overview(ticker: str) -> dict:
    ### YOUR CODE HERE
    data = requests.get(
        f'{AV_BASE}/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHAVANTAGE_API_KEY}'
    ).json()
    if "Name" not in data:
        return {"error": f"No overview data for {ticker}"}
    return {
        "ticker": ticker,
        "name": data.get("Name"),
        "sector": data.get("Sector"),
        "pe_ratio": data.get("PERatio"),
        "eps": data.get("EPS"),
        "market_cap": data.get("MarketCapitalization"),
        "52w_high": data.get("52WeekHigh"),
        "52w_low": data.get("52WeekLow"),
    }


# ── Tool 7 — YOUR IMPLEMENTATION ─────────────────────────────
def get_tickers_by_sector(sector: str) -> dict:
    ### YOUR CODE HERE
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        f"SELECT * FROM stocks WHERE sector = '{sector}'", conn
    )
    if df.empty:
        df = pd.read_sql_query(
            f"SELECT * FROM stocks WHERE industry LIKE '%{sector}%'", conn
        )
    conn.close()
    return {
        "sector": sector,
        "stocks": df.to_dict(orient="records"),
    }
