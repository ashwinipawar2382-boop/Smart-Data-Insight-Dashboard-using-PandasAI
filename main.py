pip install streamlit pandas pandasai pandasai-litellm matplotlib openai
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Data Insight Dashboard", layout="wide")

st.title("📊 Smart Data Insight Dashboard with PandasAI")

# ——————————————————————————————————————————————————————
# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.write("### 📋 Your Dataset Preview")
st.dataframe(df.head())

# ——————————————————————————————————————————————————————
# Ask a natural language question
st.write("---")
query = st.text_input("Ask a question about your data (e.g., 'Show me total sales by year')")

if not query:
    st.info("Enter a question to get insights from the data.")
    st.stop()

# ——————————————————————————————————————————————————————
# Run PandasAI
openai_key = st.text_input("Enter your OpenAI API Key", type="password")
if not openai_key:
    st.warning("🔑 An OpenAI key is required to use PandasAI.")
    st.stop()

# Initialize PandasAI with OpenAI
llm = OpenAI(api_token=openai_key)
smart_df = SmartDataframe(df, config={"llm": llm})

with st.spinner("Analyzing your data..."):
    try:
        result = smart_df.chat(query)
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

# ——————————————————————————————————————————————————————
# Show result
st.write("## ✅ Result from PandasAI")
if isinstance(result, (pd.DataFrame, list)):
    try:
        st.dataframe(result)
    except:
        st.write(result)
else:
    st.write(result)

# ——————————————————————————————————————————————————————
# If plot was generated
if hasattr(smart_df, "last_plot"):
    st.write("## 📈 Visualization")
    st.pyplot(smart_df.last_plot)

st.write("---")
st.write("Powered by PandasAI & OpenAI.")
