from crewai_tools import BaseTool, tool 
import yfinance as yf
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.application import MIMEApplication
from email import encoders

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

@tool("Stock_Financials")
def fetch_stock_financials(ticker: str) -> str:
    """Fetch financial statements for the stock."""
    stock = yf.Ticker(ticker)
    
    income_stmt = stock.income_stmt
    balance_sheet = stock.balance_sheet
    cash_flow = stock.cashflow

    output = f"Financial Statements for {ticker}:\n\n"
    
    output += "Income Statement (Annual):\n"
    output += income_stmt.to_string() + "\n\n"
    
    output += "Balance Sheet (Annual):\n"
    output += balance_sheet.to_string() + "\n\n"
    
    output += "Cash Flow Statement (Annual):\n"
    output += cash_flow.to_string() + "\n"

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


def send_report(sender_email, receiver_email, password, subject, body, file_name):
    # Create a multipart message container. This will contain both text and attachments.
    message = MIMEMultipart()
    message['From'] = sender_email  
    message['To'] = receiver_email  
    message['Subject'] = subject  

    # Attach the text body of the email. 'plain' indicates it is plain text.
    message.attach(MIMEText(body, 'plain'))

    # Open the HTML report file in binary mode to attach it to the email.
    with open(file_name, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")  # Create a MIMEBase instance, which can represent any attachment type.
        part.set_payload(attachment.read())  # Read the attachment file content into the MIMEBase instance.
        encoders.encode_base64(part)  # Encode the file content in base64 to ensure safe email transport.
        
        # Add a header to indicate the file name that will appear in the email; this is for the attachment
        part.add_header(
            "Content-Disposition",
            f"attachment; filename={file_name}",
        )
        message.attach(part)  # Attach the MIMEBase instance (the file) to the message

    # Send the email
    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)  # Connect securely to Gmail's SMTP server on the SSL port
        server.login(sender_email, password)  # Log in to the server using the sender's credentials
        server.sendmail(sender_email, receiver_email, message.as_string())  # Convert the message to a string and send the email
        server.quit()  # Quit the server connection
        print("Email sent successfully!")  
    except Exception as e:
        print(f"Error: {e}")  
