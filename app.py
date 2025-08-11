import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from io import BytesIO
import base64
from groq import Groq
import asyncio

# Custom CSS
def load_css(theme="light"):
    if theme == "dark":
        return """
        <style>
        .main { background: linear-gradient(to right, #1f2937, #111827); color: #f3f4f6; }
        .stButton>button { background-color: #6366f1; color: white; border-radius: 10px; padding: 10px 20px; transition: all 0.3s ease; }
        .stButton>button:hover { background-color: #4f46e5; transform: scale(1.05); box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
        .stSlider { background: #374151; border-radius: 8px; }
        .stTextInput input { border: 2px solid #4f46e5; border-radius: 8px; }
        .anomaly-highlight { animation: fadeIn 1s ease-in-out; color: #ef4444; }
        .insight-highlight { animation: slideIn 0.5s ease; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        @keyframes slideIn { from { transform: translateY(20px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
        .tooltip { position: relative; display: inline-block; }
        .tooltip .tooltiptext { visibility: hidden; width: 160px; background-color: #4b5563; color: #f3f4f6; text-align: center; border-radius: 6px; padding: 8px; position: absolute; z-index: 1; bottom: 125%; left: 50%; margin-left: -80px; opacity: 0; transition: opacity 0.3s; }
        .tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }
        </style>
        """
    else:
        return """
        <style>
        .main { background: linear-gradient(to right, #f0f2f6, #e0e7ff); color: #1f2937; }
        .stButton>button { background-color: #4f46e5; color: white; border-radius: 10px; padding: 10px 20px; transition: all 0.3s ease; }
        .stButton>button:hover { background-color: #4338ca; transform: scale(1.05); box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
        .stSlider { background: #e0e7ff; border-radius: 8px; }
        .stTextInput input { border: 2px solid #4f46e5; border-radius: 8px; }
        .anomaly-highlight { animation: fadeIn 1s ease-in-out; color: #ef4444; }
        .insight-highlight { animation: slideIn 0.5s ease; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        @keyframes slideIn { from { transform: translateY(20px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
        .tooltip { position: relative; display: inline-block; }
        .tooltip .tooltiptext { visibility: hidden; width: 160px; background-color: #6b7280; color: #fff; text-align: center; border-radius: 6px; padding: 8px; position: absolute; z-index: 1; bottom: 125%; left: 50%; margin-left: -80px; opacity: 0; transition: opacity 0.3s; }
        .tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }
        </style>
        """

st.markdown(load_css(st.session_state.get("theme", "light")), unsafe_allow_html=True)

st.title("AnomalyScope AI: Advanced Dataset Anomaly Detector")
st.markdown("<p style='font-weight: bold;'>Unleash AI-driven insights to dominate your hackathon! Detect anomalies with precision.</p>", unsafe_allow_html=True)

# Session State
if 'df' not in st.session_state:
    st.session_state.df = None
if 'anomalies' not in st.session_state:
    st.session_state.anomalies = None
if 'client' not in st.session_state:
    st.session_state.client = None

# Sidebar
with st.sidebar:
    st.header("Advanced Settings")
    groq_key = st.text_input("Groq API Key", type="password", help="Get a free key at console.groq.com")
    if st.button("Validate API Key"):
        with st.spinner("Validating..."):
            try:
                client = Groq(api_key=groq_key)
                test_response = client.chat.completions.create(
                    messages=[{"role": "user", "content": "Test"}],
                    model="llama-3.3-70b-versatile"
                )
                st.session_state.client = client
                st.success("API Key Valid!")
            except Exception as e:
                st.error(f"Invalid Key: {str(e)}")
    model_type = st.selectbox("Anomaly Model", ["Isolation Forest", "DBSCAN", "Local Outlier Factor"])
    sensitivity = st.slider("Sensitivity (%)", 1, 20, 5, help="Higher detects more anomalies")
    chart_type = st.selectbox("Chart Type", ["Scatter", "Line", "Heatmap"])
    preprocess = st.checkbox("Auto-Preprocess (Fill/Normalize)", value=True)
    add_context = st.checkbox("Enrich with Context", value=True)
    context_input = st.text_input("Event/Date for Context (e.g., '2025-08-10 market crash')") if add_context else ""
    st.markdown("<div class='tooltip'>ℹ️<span class='tooltiptext'>Context suggests causes; add NewsAPI for real-time.</span></div>", unsafe_allow_html=True)

# Async Groq Call
async def get_llm_explanation(prompt):
    client = st.session_state.client
    if client:
        try:
            chat_completion = await asyncio.to_thread(client.chat.completions.create,
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile"
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Fallback: Groq error - {str(e)}"
    return "Fallback: No valid Groq key. Check your input."

# File Upload
uploaded_file = st.file_uploader("Upload CSV/JSON", type=["csv", "json"])

if uploaded_file:
    with st.spinner("Analyzing dataset..."):
        progress = st.progress(0)
        for i in range(100):
            asyncio.run(asyncio.sleep(0.01))
            progress.progress(i + 1)

        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_json(uploaded_file)

        if len(df) < 5:
            st.error("Dataset too small (min 5 rows).")
        else:
            if preprocess:
                df = df.fillna(df.mean(numeric_only=True))
                numeric_cols = df.select_dtypes(include=np.number).columns
                df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min() + 1e-8)

            st.session_state.df = df
            st.write("Dataset Preview", df.head())

            # Quick Insights
            if st.button("Quick Insights"):
                prompt = f"Summarize key stats for dataset: {df.describe().to_string()[:200]}"
                insight = asyncio.run(get_llm_explanation(prompt))
                st.markdown(f"<p class='insight-highlight'>📊 {insight}</p>", unsafe_allow_html=True)

            # Anomaly Detection
            numeric_cols = df.select_dtypes(include=np.number).columns
            if len(numeric_cols) == 0:
                st.error("No numeric data found.")
            else:
                data = df[numeric_cols].values
                if model_type == "Isolation Forest":
                    model = IsolationForest(contamination=sensitivity/100, random_state=42)
                    preds = model.fit_predict(data)
                elif model_type == "DBSCAN":
                    model = DBSCAN(eps=0.5, min_samples=5)
                    preds = model.fit_predict(data)
                    preds = np.where(preds == -1, -1, 1)
                else:
                    model = LocalOutlierFactor(n_neighbors=20, contamination=sensitivity/100)
                    preds = model.fit_predict(data)

                df["anomaly"] = preds
                st.session_state.anomalies = df[df["anomaly"] == -1]

if st.session_state.anomalies is not None:
    anomalies = st.session_state.anomalies
    df = st.session_state.df

    st.markdown("<h3 class='anomaly-highlight'>Detected Anomalies</h3>", unsafe_allow_html=True)
    st.dataframe(anomalies.style.highlight_max(color='red', axis=0))

    # Visualization
    fig = None
    numeric_cols = df.select_dtypes(include=np.number).columns
    if chart_type == "Scatter":
        fig = px.scatter(df, x=df.index, y=numeric_cols[0], color="anomaly",
                         color_discrete_map={1: "blue", -1: "red"}, title="Anomalies")
    elif chart_type == "Line":
        fig = px.line(df, x=df.index, y=numeric_cols[0], title="Time-Series")
        fig.add_scatter(x=anomalies.index, y=anomalies[numeric_cols[0]], mode='markers', marker=dict(color='red', size=10))
    else:
        corr = df[numeric_cols].corr()
        fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='Viridis'))
        fig.update_layout(title="Correlation Heatmap")

    if fig:
        fig.update_layout(transition_duration=500, autosize=True)
        st.plotly_chart(fig, use_container_width=True)

    # LLM Explanations
    st.markdown("<h3>Anomaly Explanations (Powered by Groq)</h3>", unsafe_allow_html=True)
    explanations = []
    for idx, row in anomalies.iterrows():
        value = row[numeric_cols[0]]
        mean_dev = abs(value - df[numeric_cols[0]].mean())
        prompt = f"Explain why value {value:.2f} at index {idx} is an anomaly. Deviation: {mean_dev:.2f}. Dataset: {df.describe().to_string()[:200]}."
        if add_context and context_input:
            prompt += f" Context: {context_input}. Suggest real-world causes like X trends or events."
        explanation = asyncio.run(get_llm_explanation(prompt))
        explanations.append([idx, value, explanation])
        st.markdown(f"<p class='tooltip'>Anomaly at {idx}: {explanation}<span class='tooltiptext'>LLM Insight</span></p>", unsafe_allow_html=True)

    # PDF Report
    if st.button("Download Enhanced PDF Report"):
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("AnomalyScope AI Report", styles['Title']))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Detected {len(anomalies)} anomalies with {model_type}.", styles['Normal']))

        data = [["Index", "Value", "Explanation"]] + explanations
        t = Table(data)
        t.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), '#4f46e5'), ('TEXTCOLOR', (0,0), (-1,0), 'white')]))
        elements.append(t)

        img_data = fig.to_image(format="png", width=600, height=400)
        img_buffer = BytesIO(img_data)
        elements.append(RLImage(img_buffer, width=500, height=300))

        doc.build(elements)
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="anomaly_report.pdf">Download PDF</a>'
        st.markdown(href, unsafe_allow_html=True)

# Sample Datasets
sample_type = st.selectbox("Load Sample Dataset", ["None", "Sales Data", "Sensor Logs"])
if sample_type != "None":
    if sample_type == "Sales Data":
        df = pd.DataFrame({"Date": pd.date_range("2025-01-01", periods=100), "Sales": np.random.normal(100, 10, 100)})
        df.loc[50:55, "Sales"] *= 2
    else:
        df = pd.DataFrame({"Time": pd.date_range("2025-01-01", periods=100), "Temp": np.random.normal(25, 2, 100)})
        df.loc[70:75, "Temp"] += 10
    st.session_state.df = df
    st.rerun()
