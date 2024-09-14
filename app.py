import os
import requests 
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process
from crewai_tools import WebsiteSearchTool, ScrapeWebsiteTool, BaseTool, tool 

# Stocks API 
import yfinance as yf
# import finnhub
# from polygon import RESTClient
# from alpha_vantage.timeseries import TimeSeries

"""
Agents:
- financial analyst
- research analyst
- investment advisor

Tasks:
- financial analysis
- research
- filings_analysis
- recommend 
"""

load_dotenv()
# os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'

# finnhub_client = finnhub.Client(api_key="crhn2e9r01qjv9rl6fo0crhn2e9r01qjv9rl6fog") # https://github.com/Finnhub-Stock-API/finnhub-python
# client = RESTClient(api_key="<API_KEY>") # https://github.com/polygon-io/client-python
# alpha vantage: C4GAKKFA1JOD3AAS

##########################################
@tool("Stock_Data")
def fetch_stock_data(ticker: str) -> dict:
    """Fetch relevant stock data for financial analysis of a given ticker."""
    stock = yf.Ticker(ticker)
    stock_info = stock.info

    # Select key metrics for stock analysis
    data = {
        'P/E Ratio': stock_info.get('forwardPE'),
        'EPS': stock_info.get('trailingEps'),
        'Revenue': stock_info.get('totalRevenue'),
        'Debt to Equity': stock_info.get('debtToEquity'),
        'Market Cap': stock_info.get('marketCap'),
        'Dividend Yield': stock_info.get('dividendYield'),
        'Open Price': stock_info.get('open'),
        'Close Price': stock_info.get('previousClose'),
        'Day High': stock_info.get('dayHigh'),
        'Day Low': stock_info.get('dayLow'),
        'Volume': stock_info.get('volume')
    }

    output = (
        f"Stock Data for {ticker}:\n"
        f"P/E Ratio: {data['P/E Ratio']}\n"
        f"EPS: {data['EPS']}\n"
        f"Revenue: {data['Revenue']}\n"
        f"Debt to Equity: {data['Debt to Equity']}\n"
        f"Market Cap: {data['Market Cap']}\n"
        f"Dividend Yield: {data['Dividend Yield']}\n"
        f"Open Price: {data['Open Price']}\n"
        f"Close Price: {data['Close Price']}\n"
        f"Day High: {data['Day High']}\n"
        f"Day Low: {data['Day Low']}\n"
        f"Volume: {data['Volume']}\n"
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

# Data collection 
data_collector = Agent(
    role="Stock Data Collector",
    goal="Efficiently gather stock market data for financial analysis.",
    backstory=("A reliable financial data collector who has access to stock data APIs and tools."),
    tools=[fetch_stock_data],
    verbose=True,
)

data_collection_task = Task(
    description="Collect key stock data metrics for {company_stock}.",
    expected_output="A dictionary of the most relevant financial metrics for stock analysis.",
    agent=data_collector,
)

# News Researcher 
researcher = Agent(
    role="News Research Analyst",
    goal="Find and summarize the latest news articles about the company.",
    backstory=("A diligent researcher who keeps an eye on the latest financial news and trends that impact stock performance."),
    tools=[fetch_stock_news, WebsiteSearchTool, ScrapeWebsiteTool],
    verbose=True
)

research_task = Task(
    description="Find the latest financial news for {company_stock} and summarize the key points from the top 5 articles.",
    expected_output="A summary of the most recent and relevant news articles about {company_stock}.",
    agent=researcher
)

# Financial Analyst 
financial_analyst = Agent(
    role="Expert Financial Analyst",
    goal="Analyze financial data and market trends to provide a comprehensive stock report.",
    backstory=("A highly experienced financial analyst who delivers detailed insights on stock performance and market factors, while providing well-rounded recommendations to clients."),
    tools=[WebsiteSearchTool, ScrapeWebsiteTool],  
    verbose=True
)

financial_analysis_task = Task(
    description=(
        "Analyze the financial data of {company_stock} using key metrics. "
        "Also, consider recent news and market trends provided by the research agent."
    ),
    expected_output=(
        "A comprehensive stock report that includes the company's financial standing, potential growth, and market risks, "
        "along with recommendations for investors based on current trends."
    ),
    agent=financial_analyst
)

# Crew
inputs = {
    "company_stock": "IBM"
}

crew = Crew(
    agents=[data_collector, researcher, financial_analyst],
    tasks=[data_collection_task, research_task, financial_analysis_task],
    process=Process.sequential,  # Sequential process: tasks executed in order
    full_output=True,
    verbose=True
)
crew_output = crew.kickoff(inputs=inputs)
print(crew)

##########################################

"""
Features:
- manager
- hierarchical 
- allow delegation 
- custom tool stock api; return selected keys from dictionary 
- multiple custom tools runnning async for different financial data retreival and custom calculations
- compare multiple stocks 

streamlit run main.py
"""
