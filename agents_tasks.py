from crewai import Agent, Task, Crew, Process
from crewai.tasks import TaskOutput 
from crewai_tools import WebsiteSearchTool, ScrapeWebsiteTool
from custom_tools import fetch_stock_data, fetch_stock_financials, fetch_stock_news

search_tool = WebsiteSearchTool()
scrape_tool = ScrapeWebsiteTool()

# Callbacks 
# def crew_step_callback(agent, task, step):
#     print(f"Step Callback - Agent: {agent.role}, Task: {task.description}, Step: {step}")

# def crew_task_callback(output: TaskOutput):
#     print(f"Task Callback - Task Completed: {output.description}")

# Data collection 
data_collector = Agent(
    role="Stock Data Collector",
    goal="Efficiently gather stock market data for financial analysis.",
    backstory=("A reliable financial data collector who has access to stock data APIs and tools."),
    tools=[fetch_stock_data, fetch_stock_financials],
    verbose=True,
    max_iter=5,
    allow_delegation=False, 
)

data_collection_task = Task(
    description="Collect key stock data metrics for {company_stock} using its ticker format. Use only the data provided and do not request for more metrics.",
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
    max_iter=10,
    allow_delegation=False,  
    max_execution_time=180,
    max_retry_limit=2,
)

stock_market_research_task = Task( 
    description=(
        "Conduct research on {company_stock} focusing on these topics:\n"
        "- General market trends effecting the company.\n"
        "- Industry comparisons between competitors and recent events affecting the company.\n"
        "- Risks and opportunities related to current market conditions.\n"
        "Rules: Perform up to 3 different search queries per topic. "
        "Do not repeat the same queries if the result is not helpful. "
        "If the information is unavailable after the 3 attempts then skip to the next topic. "
        "IMPORTANT: If the website returns a cookie consent information prevents scraping content after 1 attempt then never search that website again."
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

manager = Agent(
    role="Project Manager",
    goal="Coordinate the entire stock analysis workflow, ensuring that all agents complete their tasks efficiently and that the final report is comprehensive and accurate.",
    backstory=(
        "An expert project manager with a deep understanding of financial analysis and stock market trends. "
        "You are responsible for managing the flow of tasks, ensuring data collection, research and report writing are done in a coordinated manner."
        "You must make sure that no tasks are repeated and only use the information provided."
    ),
    allow_delegation=True,  
)

# Hierarchical Agents
crew = Crew(
    agents=[data_collector, news_reader, stock_market_researcher, financial_analyst],
    tasks=[data_collection_task, news_reader_task, stock_market_research_task, financial_analysis_task],
    process=Process.hierarchical,  
    manager_agent=manager,
    full_output=True,
    verbose=True, 
    memory=True, 
    planning=True,
    # step_callback=crew_step_callback,
    # task_callback=crew_task_callback,
)

# Sequential Agents
# crew = Crew(
#     agents=[data_collector, news_reader, financial_analyst],
#     tasks=[data_collection_task, news_reader_task, financial_analysis_task],
#     process=Process.sequential,  
#     full_output=True,
#     verbose=True, 
#     memory=True, 
# )
