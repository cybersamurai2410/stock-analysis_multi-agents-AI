import streamlit as st
import os
from dotenv import load_dotenv
import agentops
import markdown
import pdfkit

from agents_tasks import crew
from custom_tools import send_report

load_dotenv()
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o-mini'
agentops.init()

st.title("Stock Analysis Report Generator")

# Ensure session state tracks whether the report has been generated and store the output logs
if 'report_generated' not in st.session_state:
    
    st.session_state['report_generated'] = False

if 'crew_output' not in st.session_state:
    st.session_state['crew_output'] = None

# Input field to enter the company name
company_name = st.text_input("Enter Company Name or Stock Ticker", "")

# Button to generate the stock analysis report
if st.button("Generate Report"):
    if company_name != "":
        with st.spinner("Generating stock analysis report..."):
            crew_output = crew.kickoff(inputs={"company_stock": company_name})
            st.session_state['crew_output'] = crew_output
            # print("Report:\n", crew_output) # crew_output.raw

            # Crew output logs 
            print(f"\nRaw Output:\n {crew_output.raw}")
            print(f"\nTasks Output:\n {crew_output.tasks_output}")
            print(f"\nToken Usage:\n {crew_output.token_usage}")
            # print(f"\nUsage Metrics:\n {crew.usage_metrics}") # crew_output.token_usage

        st.success(f"Report for {company_name} generated successfully!")
        st.session_state['report_generated'] = True  # Set state to indicate report has been generated
    else:
        st.error("Please enter a valid company name or stock ticker.")
        st.session_state['report_generated'] = False  

# Check if the report has been generated 
if st.session_state['report_generated']:
    # st.markdown(example_report)

    # Display generated report 
    crew_output = st.session_state['crew_output']
    if crew_output:
        st.markdown(crew_output.raw)

    # Save stock analysis report 
    with open("stock_report.txt", "r") as file:
        markdown_text = file.read()

    html = markdown.markdown(markdown_text)
    with open("stock_report.html", "w") as file:
        file.write(html)

    # Convert HTML to PDF report 
    config = pdfkit.configuration(wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe")
    pdfkit.from_file("stock_report.html", "stock_report.pdf", configuration=config)

    # Display chain of thought reasoning and API call metrics 
    with st.expander("Show Chain of Thought"):
        # st.markdown(chain_of_thought)
        st.markdown(crew_output.tasks_output)
        st.markdown(crew_output.token_usage) 

    # Send report by email
    email_address = st.text_input("Enter your email", "")
    if st.button("Send Email"):
        sender_email = os.getenv('SENDER_EMAIL')  
        receiver_email = email_address
        password = os.getenv('EMAIL_PASSWORD') 
        subject = f"Stock Analysis Report: {company_name}"  
        body = "Please find the attached stock analysis report." 
        file_name="stock_report.pdf"

        send_report(sender_email, receiver_email, password, subject, body, file_name)
        st.success(f"Email sent successfully to {email_address}!")
