# Stock Analysis Report Generator

[Click to watch the Application Demo](https://drive.google.com/file/d/15O_BUONNgnGq0lsCz9U6X0y-nrmWZKyy/view?usp=sharing)
<br></br>
[AgentOps Evaluation Demo](https://drive.google.com/file/d/15O_BUONNgnGq0lsCz9U6X0y-nrmWZKyy/view?usp=sharing)

## Overview
The Stock Analysis Report Generator allows users to generate comprehensive stock analysis reports based on company names or stock tickers. The application uses multi-agent AI using the [CrewAI framework](https://github.com/crewAIInc/crewAI) to collect stock data, news, research/web scraping and perform financial analysis. The Crew operates as hierarchical process with a manager agent coordinating the tasks of the other agaents. 

**Agents:**
- Stock Data Collector
- News Reader
- Stock Market Researcher
- Financial Analyst
- Project Manager

## Workflow
- Input company name or stock ticker to generate reports.
- Fetches real-time stock data and historical prices.
- Extract financial data from income statements, balance sheets and cash flow.
- Summarizes recent news articles related to the stock.
- Conducts stock market research with web scraping. 
- Generates a detailed stock analysis report in both text/markdown and PDF formats.
- Displays the report and chain of thought reasoning in the user interface. 
- Sends the generated report via email.

**Run Application:**
```bash
streamlit run app.py
```

## Screenshots

