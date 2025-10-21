import streamlit as st
import os
from src.query_engine import QueryEngine
from dotenv import load_dotenv


# st.set_page_config(page_title="RAG Chatbot", layout="centered")
# st.title(" RAG Chatbot —EV Vehicles")


# query = st.text_input("Ask a question about your data")

# if st.button("Submit"):
#     if query.strip():
#         with st.spinner("Analyzing"):
#             engine = QueryEngine(use_gemini=True)
#             response = engine.process_query(query)
#         st.write(response.get("answer", "No response"))
#     else:
#         st.warning("Please enter a question first!")


# Load .env for API keys
load_dotenv()

st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("RAG Chatbot — EV Vehicles")

query = st.text_input("Ask a question about your data")

if st.button("Submit"):
    if query.strip():
        with st.spinner("Analyzing"):
            engine = QueryEngine(use_gemini=True)
            response = engine.process_query(query)

        answer = response.get("answer", "No response")
        sources = response.get("sources", [])

        # Display main answer
        st.markdown(f"""
        <div style="font-family: 'Arial', sans-serif; font-size: 16px; line-height:1.5;">
            <b>Answer:</b><br>{answer}
        </div>
        """, unsafe_allow_html=True)

        # Display sources if any
        if sources:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("<b>Sources:</b>", unsafe_allow_html=True)
            for src in sources:
                st.markdown(f"""
                <div style="font-family: 'Arial', sans-serif; font-size: 14px; margin-bottom:10px;">
                    <b>{src.get('filename','')}</b>:<br>{src.get('content','')}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("Please enter a question first!")