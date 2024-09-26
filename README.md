# Stock Analysis Report Generator

[Click to watch the Application Demo](https://drive.google.com/file/d/15O_BUONNgnGq0lsCz9U6X0y-nrmWZKyy/view?usp=sharing)
<br></br>
[AgentOps Evaluation Demo](https://drive.google.com/file/d/15O_BUONNgnGq0lsCz9U6X0y-nrmWZKyy/view?usp=sharing)

## Overview
The Stock Analysis Report Generator allows users to generate comprehensive stock analysis reports based on company names or stock tickers. The application uses multi-agent AI with OpenAI GPT models using the [CrewAI framework](https://github.com/crewAIInc/crewAI) to collect stock data, news, research/web scraping and perform financial analysis. The Crew operates as hierarchical process with a manager agent coordinating the tasks of the other agaents. The agentic wokrflow has been evaluated using the AgentOps platform to trace the LLM output.

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
<img width="308" alt="img1" src="https://github.com/user-attachments/assets/f087f0ab-4a62-4de4-8235-79d588f9acc8">
<img width="388" alt="img2" src="https://github.com/user-attachments/assets/592d5a9b-7118-407e-bb55-39610414ffb6">
<img width="505" alt="img3" src="https://github.com/user-attachments/assets/fef3d724-0c81-49f9-a949-80ccd7e63938"><br></br>
<img width="411" alt="img4" src="https://github.com/user-attachments/assets/5892b045-ab56-4e52-a1eb-92cf7ad8fa15">
