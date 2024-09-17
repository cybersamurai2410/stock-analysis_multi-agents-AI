import os
import requests 
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process
from crewai_tools import WebsiteSearchTool, ScrapeWebsiteTool, BaseTool, tool 

import yfinance as yf
import markdown
import pdfkit

load_dotenv()
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o-mini'

search_tool = WebsiteSearchTool()
scrape_tool = ScrapeWebsiteTool()

@tool("Stock_Data")
def fetch_stock_data(ticker: str) -> str:
    """Fetch stock data and historical market data."""
    stock = yf.Ticker(ticker)
    
    # Fetch current stock information and history of prices
    stock_info = stock.info
    hist = stock.history(period="1mo")  

    output = (
        f"Stock Data for {ticker}:\n"
        f"P/E Ratio: {stock_info.get('forwardPE', 'N/A')}\n"
        f"EPS: {stock_info.get('trailingEps', 'N/A')}\n"
        f"Revenue: {stock_info.get('totalRevenue', 'N/A')}\n"
        f"Debt to Equity: {stock_info.get('debtToEquity', 'N/A')}\n"
        f"Market Cap: {stock_info.get('marketCap', 'N/A')}\n"
        f"Dividend Yield: {stock_info.get('dividendYield', 'N/A')}\n"
        f"Open Price: {stock_info.get('open', 'N/A')}\n"
        f"Close Price: {stock_info.get('previousClose', 'N/A')}\n"
        f"Day High: {stock_info.get('dayHigh', 'N/A')}\n"
        f"Day Low: {stock_info.get('dayLow', 'N/A')}\n"
        f"Volume: {stock_info.get('volume', 'N/A')}\n\n"
    )

    output += "Historical Stock Prices (Past Month):\n"
    for date, row in hist.iterrows():
        output += (
            f"Date: {date.date()}, Open: {row['Open']}, High: {row['High']}, "
            f"Low: {row['Low']}, Close: {row['Close']}, Volume: {row['Volume']}\n"
        )

    return output

@tool("Stock_News")
def fetch_stock_news(ticker: str) -> str:
    """Fetch recent news articles related to the company stock of a given ticker."""
    stock = yf.Ticker(ticker)
    news_items = stock.news
    
    # Format the news into a readable summary
    news_summary = []
    for item in news_items[:5]:  # Limit to the top 5 news articles
        title = item.get('title', 'No title available')
        publisher = item.get('publisher', 'Unknown publisher')
        link = item.get('link', 'No link available')
        summary = f"{title} - Published by {publisher}. Read more: {link}"
        news_summary.append(summary)
    
    # Join all summaries into a single string
    return "Recent news:\n" + "\n\n".join(news_summary)

"""
- Agents: https://docs.crewai.com/core-concepts/Agents/
- Tasks: https://docs.crewai.com/core-concepts/Tasks/
"""

# Data collection 
data_collector = Agent(
    role="Stock Data Collector",
    goal="Efficiently gather stock market data for financial analysis.",
    backstory=("A reliable financial data collector who has access to stock data APIs and tools."),
    tools=[fetch_stock_data],
    verbose=True,
    max_iter=5,
)

data_collection_task = Task(
    description="Collect key stock data metrics for {company_stock}.",
    expected_output="A dictionary of the most relevant financial metrics for stock analysis.",
    agent=data_collector
)

# News Researcher 
researcher = Agent(
    role="News Research Analyst",
    goal="Find and summarize the latest news articles about the company.",
    backstory=("A diligent researcher who keeps an eye on the latest financial news and trends that impact stock performance."),
    tools=[fetch_stock_news, search_tool, scrape_tool],
    verbose=True,
    max_iter=5,
)

research_task = Task(
    description="Find the latest financial news for {company_stock} and summarize the key points from recent articles.",
    expected_output="A summary of the most recent and relevant news articles about {company_stock}.",
    agent=researcher,
)

# Financial Analyst (maybe seperate into fanancial research and report writer)
financial_analyst = Agent(
    role="Expert Financial Analyst",
    goal="Analyze financial data and market trends to write a comprehensive report of the stock analysis.",
    backstory=("An experienced financial analyst who delivers detailed insights on stock performance and market factors while providing recommendations to investors based on the data."),
    # tools=[search_tool, scrape_tool],  
    verbose=True,
    max_iter=5,
    allow_delegation=True,
    
)

financial_analysis_task = Task( 
    description=(
        "Analyze the financial data of {company_stock} using metrics."
        "Use recent news and market trends to write the report."
    ),
    expected_output=(
        "A comprehensive report of stock analysis about the company along with recommendations for investors based on current trends."
    ),
    agent=financial_analyst,
    output_file="stock_report.txt",
    async_execution=True,
)

# Crew
inputs = {
    "company_stock": "ibm" # stock_comparison: list converted to string format  
}

crew = Crew(
    agents=[data_collector, researcher, financial_analyst],
    tasks=[data_collection_task, research_task, financial_analysis_task],
    process=Process.sequential,  
    full_output=True,
    verbose=True
)
crew_output = crew.kickoff(inputs=inputs)
print("Report: \n", crew_output) 

# Save stock analysis report 
with open('stock_report.txt', 'r') as file:
    markdown_text = file.read()

html = markdown.markdown(markdown_text)
print("html: \n", html)

with open("stock_report.html", "w") as file:
    file.write(html)

"""
Features:
- manager
- hierarchical 
- custom tool stock api; return selected keys from dictionary 
- multiple custom tools runnning async for different financial data retreival and custom calculations
- compare multiple stocks; add to input dict
- custom function for executing ml model that predicts stock price 

Issues:
- Input company name instead of tickers.
- Input multiple company names for comparison.

Run:
streamlit run main.py
"""
