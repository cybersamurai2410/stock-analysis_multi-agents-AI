# config.py
import streamlit as st
from dotenv import load_dotenv
import os
import agentops

def initialize_app():
    # Load environment variables only once
    if 'env_loaded' not in st.session_state:
        load_dotenv()
        st.session_state.env_loaded = True

    # Initialize AgentOps only once
    if 'agentops_initialized' not in st.session_state:
        agentops.init()
        st.session_state.agentops_initialized = True

    # Set and store the model name in session state
    if 'model_name' not in st.session_state:
        st.session_state.model_name = 'gpt-4o-mini'

        os.environ["OPENAI_MODEL_NAME"] = st.session_state.model_name
        print(f"Model being used: {os.getenv('OPENAI_MODEL_NAME')}")
