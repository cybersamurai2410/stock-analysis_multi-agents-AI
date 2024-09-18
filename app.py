import os
import requests 
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process
from crewai_tools import WebsiteSearchTool, ScrapeWebsiteTool, BaseTool, tool 

import yfinance as yf
import markdown
import pdfkit

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.application import MIMEApplication
from email import encoders

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
    return "Recent News:\n" + "\n\n".join(news_summary)

# Data collection 
data_collector = Agent(
    role="Stock Data Collector",
    goal="Efficiently gather stock market data for financial analysis.",
    backstory=("A reliable financial data collector who has access to stock data APIs and tools."),
    tools=[fetch_stock_data],
    verbose=True,
    max_iter=5,
    allow_delegation=False, 
)

data_collection_task = Task(
    description="Collect key stock data metrics for {company_stock} using its ticker format.",
    expected_output="Data about most relevant financial metrics for stock analysis.",
    agent=data_collector,
    async_execution=False,
)

# News Researcher 
news_reader = Agent(
    role="News Reader",
    goal="Find and summarize the latest news articles about the company.",
    backstory=("A diligent researcher who keeps an eye on the latest financial news and trends that impact stock performance."),
    tools=[fetch_stock_news, search_tool, scrape_tool],
    verbose=True,
    max_iter=5,
    allow_delegation=False, 
)

news_reader_task = Task(
    description="Find the latest financial news for {company_stock} and summarize the key points from recent articles.",
    expected_output="A summary of the most recent and relevant news articles about {company_stock}.",
    agent=news_reader,
    async_execution=False,
)

# Stock Market Researcher 
stock_market_researcher = Agent(
    role="Stock Market Researcher",
    goal="Research stock performance, market trends and industry movements to provide insights.",
    backstory=("An experienced stock market researcher who gathers information from sources to offer insights about the performance of the company."),
    tools=[search_tool, scrape_tool],  
    verbose=True,
    max_iter=5,
    allow_delegation=False,  
)

stock_market_research_task = Task( # Separate into multiple tasks for the same agent 
    description=(
        "Conduct research on {company_stock} focusing on:\n"
        "- General market trends effecting the company.\n"
        "- Industry comparisons between competitors and recent events affecting the company.\n"
        "- Risks and opportunities related to current market conditions.\n"
    ),
    expected_output=(
        "A clear analysis of {company_stock} covering market trends, industry comparisons, risks and opportunities."
    ),
    agent=stock_market_researcher,
    async_execution=False,
)

# Financial Analyst  
financial_analyst = Agent(
    role="Financial Analyst",
    goal="Analyze financial stock data and use information about the company to write a comprehensive stock analysis report.",
    backstory=("A skilled financial analyst who analyzes company data and provides detailed stock reports."),
    verbose=True,
    max_iter=5,
    allow_delegation=False,  
)

financial_analysis_task = Task(
    description=(
        "Analyze the research on {company_stock} and write a comprehensive stock analysis report."
    ),
    expected_output=(
        "A detailed report that includes analysis of the stock data, financial insights, recent news and market information "
        "followed by the conclusion."
    ),
    agent=financial_analyst,
    output_file="stock_report.txt",
    async_execution=False,
    context=[data_collection_task, news_reader_task, stock_market_research_task],
)

# Crew inputs 
inputs = {
    "company_stock": "amazon"  
}

# crew = Crew(
#     agents=[data_collector, news_reader, stock_market_researcher, financial_analyst],
#     tasks=[data_collection_task, news_reader_task, stock_market_research_task, financial_analysis_task],
#     process=Process.sequential,  
#     full_output=True,
#     verbose=True, 
#     memory=True, 
# )

# Manager required for hierarchical process 
manager = Agent(
    role="Project Manager",
    goal="Coordinate the entire stock analysis workflow, ensuring that all agents complete their tasks efficiently and that the final report is comprehensive and accurate.",
    backstory=(
        "An expert project manager with a deep understanding of financial analysis and stock market trends. "
        "You are responsible for managing the flow of tasks, ensuring data collection, research and report writing are done in a coordinated manner."
        "Do not repeat the task that handles stock data collection and use only the data that is provided."
    ),
    allow_delegation=True,  
)

crew = Crew(
    agents=[data_collector, news_reader, stock_market_researcher, financial_analyst],
    tasks=[data_collection_task, news_reader_task, stock_market_research_task, financial_analysis_task],
    process=Process.hierarchical,  
    manager_agent=manager,
    full_output=True,
    verbose=True, 
    memory=True, 
)

crew_output = crew.kickoff(inputs=inputs)
print("Report:\n", crew_output) 

# Crew output logs 
print(f"\nRaw Output:\n {crew_output.raw}")
print(f"\nTasks Output:\n {crew_output.tasks_output}")
print(f"\nToken Usage:\n {crew_output.token_usage}")

# Save stock analysis report 
with open("stock_report.txt", "r") as file:
    markdown_text = file.read()

html = markdown.markdown(markdown_text)
with open("stock_report.html", "w") as file:
    file.write(html)

# Convert HTML to PDF report 
config = pdfkit.configuration(wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe")
pdfkit.from_file("stock_report.html", "stock_report.pdf", configuration=config)

# Send report by email 

"""
Features:
- compare multiple stocks; add to input dict as separate crew
- custom tool for executing ml model that predicts timeseries stock price 
- store chain of thought; crew logs 

Issues:

Run:
streamlit run main.py
"""
