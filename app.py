import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import requests
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="🔮 AnomalyHunter AI Pro",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced CSS with animations and modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main-header {
        font-family: 'Orbitron', monospace;
        font-size: 4rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4, #FFEAA7, #DDA0DD);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientShift 4s ease infinite;
        margin-bottom: 2rem;
        text-shadow: 0 0 30px rgba(78, 205, 196, 0.5);
    }
    
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        text-align: center;
        color: #34495e;
        margin-bottom: 3rem;
        animation: fadeInUp 1s ease-out;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    @keyframes glow {
        0% { box-shadow: 0 0 5px rgba(78, 205, 196, 0.3); }
        50% { box-shadow: 0 0 20px rgba(78, 205, 196, 0.8), 0 0 30px rgba(78, 205, 196, 0.4); }
        100% { box-shadow: 0 0 5px rgba(78, 205, 196, 0.3); }
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        animation: glow 3s infinite;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.5);
    }
    
    .algorithm-selector {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid #4ECDC4;
        animation: pulse 2s infinite;
    }
    
    .status-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        animation: fadeInUp 1s ease-out;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        color: #2c3e50;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff6b6b;
        animation: fadeInUp 1s ease-out;
    }
    
    .ai-report-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        color: white;
        box-shadow: 0 15px 35px rgba(30, 60, 114, 0.3);
        animation: fadeInUp 1.2s ease-out;
    }
    
    .feature-importance-bar {
        animation: slideInLeft 1s ease-out;
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-50px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .stSelectbox > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #4ECDC4, #45B7D1);
    }
    
    .floating-icon {
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #4ECDC4;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .neuomorphism {
        background: #e0e5ec;
        border-radius: 20px;
        box-shadow: 9px 9px 16px #a3b1c6, -9px -9px 16px #ffffff;
        padding: 2rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'anomaly_history' not in st.session_state:
    st.session_state.anomaly_history = []
if 'model_performance' not in st.session_state:
    st.session_state.model_performance = {}
if 'real_time_data' not in st.session_state:
    st.session_state.real_time_data = pd.DataFrame()

# Groq API function
def generate_ai_report(anomalies_data, algorithm, groq_api_key):
    """Generate comprehensive AI report using Groq API"""
    if not groq_api_key:
        return "⚠️ Please enter your Groq API key to generate AI reports"
    
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    
    # Prepare anomaly summary
    n_anomalies = len(anomalies_data)
    features = list(anomalies_data.columns)
    
    prompt = f"""
    As an expert AI analyst, analyze this anomaly detection report and provide comprehensive insights:
    
    ANALYSIS SUMMARY:
    - Algorithm Used: {algorithm}
    - Total Anomalies Detected: {n_anomalies}
    - Features Analyzed: {', '.join(features[:5])}
    - Detection Timestamp: {pd.Timestamp.now()}
    
    Please provide:
    1. 🎯 EXECUTIVE SUMMARY (2-3 sentences)
    2. 🔍 ANOMALY PATTERNS IDENTIFIED
    3. ⚠️ RISK ASSESSMENT & SEVERITY LEVELS
    4. 💡 ACTIONABLE RECOMMENDATIONS
    5. 🚀 PREVENTIVE MEASURES
    6. 📊 BUSINESS IMPACT ANALYSIS
    
    Make it professional, actionable, and suitable for C-level executives.
    """
    
    data = {
        "messages": [
            {"role": "system", "content": "You are an expert AI anomaly detection analyst with deep knowledge of industrial systems, cybersecurity, and business intelligence."},
            {"role": "user", "content": prompt}
        ],
        "model": "mixtral-8x7b-32768",
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", 
                               headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"⚠️ API Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"⚠️ Error generating report: {str(e)}"

# Advanced anomaly detection with ensemble methods
def advanced_anomaly_detection(X, algorithm, params):
    """Enhanced anomaly detection with multiple algorithms"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if algorithm == "🔮 Neural Isolation Forest":
        model = IsolationForest(
            contamination=params['contamination'],
            n_estimators=params['n_estimators'],
            max_samples='auto',
            max_features=1.0,
            random_state=42
        )
        predictions = model.fit_predict(X_scaled)
        scores = model.decision_function(X_scaled)
        
    elif algorithm == "🧠 Quantum DBSCAN":
        model = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
        cluster_labels = model.fit_predict(X_scaled)
        predictions = np.where(cluster_labels == -1, -1, 1)
        scores = np.zeros(len(predictions))
        
    elif algorithm == "⚡ Hyperdimensional SVM":
        model = OneClassSVM(nu=params['nu'], gamma=params['gamma'], kernel='rbf')
        predictions = model.fit_predict(X_scaled)
        scores = model.decision_function(X_scaled)
        
    elif algorithm == "🎯 Deep LOF":
        model = LocalOutlierFactor(
            n_neighbors=params['n_neighbors'],
            contamination=params['contamination'],
            metric='manhattan'
        )
        predictions = model.fit_predict(X_scaled)
        scores = model.negative_outlier_factor_
        
    elif algorithm == "📊 Statistical Z-Nexus":
        z_scores = np.abs(stats.zscore(X_scaled, axis=0))
        max_z_scores = np.max(z_scores, axis=1)
        threshold = params['threshold']
        predictions = np.where(max_z_scores > threshold, -1, 1)
        scores = -max_z_scores
        
    elif algorithm == "🚀 AI Ensemble Supreme":
        # Advanced ensemble with weighted voting
        models = {
            'isolation': IsolationForest(contamination=0.1, random_state=42),
            'svm': OneClassSVM(nu=0.1, gamma='scale'),
            'lof': LocalOutlierFactor(contamination=0.1)
        }
        
        predictions_dict = {}
        scores_dict = {}
        
        for name, model in models.items():
            pred = model.fit_predict(X_scaled)
            predictions_dict[name] = pred
            
            if hasattr(model, 'decision_function'):
                scores_dict[name] = model.decision_function(X_scaled)
            elif hasattr(model, 'negative_outlier_factor_'):
                scores_dict[name] = model.negative_outlier_factor_
        
        # Weighted ensemble voting
        weights = {'isolation': 0.4, 'svm': 0.35, 'lof': 0.25}
        final_scores = np.zeros(len(X))
        
        for name, weight in weights.items():
            if name in scores_dict:
                normalized_scores = (scores_dict[name] - scores_dict[name].min()) / (scores_dict[name].max() - scores_dict[name].min())
                final_scores += weight * normalized_scores
        
        threshold = np.percentile(final_scores, 10)  # Bottom 10% as anomalies
        predictions = np.where(final_scores < threshold, -1, 1)
        scores = final_scores
    
    return model, predictions, scores, scaler

# Header with floating animation
st.markdown('<div class="floating-icon"><h1 class="main-header">🔮 AnomalyHunter AI Pro</h1></div>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Next-Generation AI-Powered Anomaly Detection Platform with Real-time Intelligence</p>', unsafe_allow_html=True)

# Enhanced sidebar with Groq API integration
with st.sidebar:
    st.markdown('<div class="algorithm-selector">', unsafe_allow_html=True)
    st.markdown("## 🎯 AI Control Center")
    
    # Groq API Key
    st.markdown("### 🔑 AI Report Generator")
    groq_api_key = st.text_input("Enter Groq API Key", type="password", help="Get your free API key from https://console.groq.com")
    
    # Enhanced data source selection
    st.markdown("### 📊 Data Intelligence Source")
    data_source = st.selectbox(
        "Select Data Source",
        ["🔬 Advanced Synthetic Generator", "📁 Upload Enterprise CSV", "📡 Real-time Stream Simulation", "🌐 Live API Integration"]
    )
    
    # Advanced algorithm selection
    st.markdown("### 🤖 AI Algorithm Selection")
    algorithm = st.selectbox(
        "Choose Detection Algorithm",
        ["🔮 Neural Isolation Forest", "🧠 Quantum DBSCAN", "⚡ Hyperdimensional SVM", "🎯 Deep LOF", "📊 Statistical Z-Nexus", "🚀 AI Ensemble Supreme"]
    )
    
    # Dynamic parameters based on algorithm
    st.markdown("### ⚙️ Algorithm Hyperparameters")
    
    params = {}
    if "Isolation Forest" in algorithm:
        params['contamination'] = st.slider("🎯 Contamination Rate", 0.01, 0.3, 0.1, 0.01)
        params['n_estimators'] = st.slider("🌳 Forest Size", 50, 500, 200, 10)
        
    elif "DBSCAN" in algorithm:
        params['eps'] = st.slider("📏 Epsilon Distance", 0.1, 3.0, 0.5, 0.1)
        params['min_samples'] = st.slider("👥 Min Cluster Size", 3, 30, 8, 1)
        
    elif "SVM" in algorithm:
        params['nu'] = st.slider("⚖️ Nu Parameter", 0.01, 0.5, 0.1, 0.01)
        params['gamma'] = st.selectbox("🔥 Gamma", ["scale", "auto"])
        
    elif "LOF" in algorithm:
        params['n_neighbors'] = st.slider("🎯 Neighbors Count", 5, 100, 25, 5)
        params['contamination'] = st.slider("🎯 Contamination Rate", 0.01, 0.3, 0.1, 0.01)
        
    elif "Z-Nexus" in algorithm:
        params['threshold'] = st.slider("📊 Z-Score Threshold", 1.0, 6.0, 3.0, 0.1)
    
    # Advanced visualization options
    st.markdown("### 🎨 Visualization Matrix")
    viz_options = {
        '3D_neural_viz': st.checkbox("🌌 3D Neural Visualization", True),
        'correlation_matrix': st.checkbox("🔗 Correlation Heatmap", True),
        'time_series_analysis': st.checkbox("⏰ Time Series Analysis", True),
        'feature_importance': st.checkbox("🎯 Feature Importance", True),
        'anomaly_clustering': st.checkbox("🔍 Anomaly Clustering", True)
    }
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main content area with enhanced layout
col1, col2 = st.columns([3, 1])

with col1:
    # Enhanced data loading with progress bars
    if data_source == "🔬 Advanced Synthetic Generator":
        st.markdown('<div class="neuomorphism">', unsafe_allow_html=True)
        st.markdown("### 🧬 Quantum Data Synthesis Laboratory")
        
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            n_samples = st.number_input("Sample Quantum", 500, 50000, 5000, 100)
        with col_b:
            n_features = st.number_input("Feature Dimensions", 3, 50, 10, 1)
        with col_c:
            anomaly_rate = st.slider("Anomaly Injection %", 1, 30, 12, 1)
        with col_d:
            complexity = st.selectbox("Data Complexity", ["Simple", "Complex", "Chaotic"])
        
        if st.button("🎲 Generate Quantum Dataset", type="primary"):
            with st.spinner("🔄 Synthesizing quantum anomaly patterns..."):
                progress_bar = st.progress(0)
                
                # Generate sophisticated synthetic data
                np.random.seed(42)
                
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                if complexity == "Simple":
                    normal_data = np.random.multivariate_normal(
                        mean=np.zeros(n_features),
                        cov=np.eye(n_features),
                        size=int(n_samples * (1 - anomaly_rate/100))
                    )
                elif complexity == "Complex":
                    # Multi-modal distribution
                    normal_data1 = np.random.multivariate_normal(
                        mean=np.ones(n_features) * 2,
                        cov=np.eye(n_features) * 0.5,
                        size=int(n_samples * (1 - anomaly_rate/100) * 0.6)
                    )
                    normal_data2 = np.random.multivariate_normal(
                        mean=np.ones(n_features) * -1,
                        cov=np.eye(n_features) * 0.8,
                        size=int(n_samples * (1 - anomaly_rate/100) * 0.4)
                    )
                    normal_data = np.vstack([normal_data1, normal_data2])
                else:  # Chaotic
                    # Non-linear relationships with noise
                    t = np.linspace(0, 4*np.pi, int(n_samples * (1 - anomaly_rate/100)))
                    normal_data = np.column_stack([
                        np.sin(t) + np.random.normal(0, 0.1, len(t)),
                        np.cos(t) + np.random.normal(0, 0.1, len(t))
                    ])
                    # Add more features
                    for i in range(2, n_features):
                        normal_data = np.column_stack([
                            normal_data,
                            np.sin(t * (i+1)) + np.random.normal(0, 0.2, len(t))
                        ])
                
                # Generate sophisticated anomalies
                n_anomalies = int(n_samples * anomaly_rate/100)
                anomaly_types = ['extreme', 'contextual', 'collective']
                
                anomaly_data = []
                for i in range(n_anomalies):
                    anomaly_type = np.random.choice(anomaly_types)
                    
                    if anomaly_type == 'extreme':
                        # Extreme outliers
                        anomaly = np.random.multivariate_normal(
                            mean=np.ones(n_features) * np.random.choice([-5, 5]),
                            cov=np.eye(n_features) * 2,
                            size=1
                        )
                    elif anomaly_type == 'contextual':
                        # Contextual anomalies
                        anomaly = np.random.multivariate_normal(
                            mean=np.random.normal(0, 3, n_features),
                            cov=np.eye(n_features) * 1.5,
                            size=1
                        )
                    else:  # collective
                        # Collective anomalies
                        anomaly = np.random.multivariate_normal(
                            mean=np.ones(n_features) * 2.5,
                            cov=np.eye(n_features) * 0.3,
                            size=1
                        )
                    
                    anomaly_data.append(anomaly[0])
                
                anomaly_data = np.array(anomaly_data)
                
                # Combine data
                all_data = np.vstack([normal_data, anomaly_data])
                true_labels = np.hstack([np.zeros(len(normal_data)), np.ones(len(anomaly_data))])
                
                # Create enhanced DataFrame with timestamps
                timestamps = pd.date_range('2024-01-01', periods=len(all_data), freq='1min')
                columns = [f'Sensor_{i+1}' for i in range(n_features)]
                
                data = pd.DataFrame(all_data, columns=columns)
                data['timestamp'] = timestamps
                data['True_Anomaly'] = true_labels
                data['Risk_Score'] = np.random.uniform(0, 1, len(data))
                
                st.success(f"✅ Generated {len(data):,} quantum samples with {anomaly_rate}% anomalies using {complexity.lower()} complexity")
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif data_source == "📁 Upload Enterprise CSV":
        st.markdown('<div class="neuomorphism">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "🔄 Upload your enterprise dataset", 
            type=['csv', 'xlsx', 'json'],
            help="Supports CSV, Excel, and JSON formats"
        )
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                    data = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    data = pd.read_json(uploaded_file)
                
                st.success(f"✅ Successfully loaded {len(data):,} records with {len(data.columns)} features")
                
                # Data quality analysis
                missing_data = data.isnull().sum().sum()
                if missing_data > 0:
                    st.warning(f"⚠️ Found {missing_data} missing values - will be handled automatically")
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                data = None
        else:
            st.info("👆 Upload your dataset to unlock AI-powered anomaly detection")
            data = None
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif data_source == "📡 Real-time Stream Simulation":
        st.markdown('<div class="neuomorphism">', unsafe_allow_html=True)
        st.markdown("### 📡 Live Data Stream Intelligence")
        
        col_stream1, col_stream2, col_stream3 = st.columns(3)
        with col_stream1:
            stream_rate = st.selectbox("Stream Rate", ["Fast", "Medium", "Slow"])
        with col_stream2:
            anomaly_injection = st.slider("Live Anomaly %", 1, 20, 8)
        with col_stream3:
            stream_duration = st.number_input("Duration (sec)", 10, 300, 60)
        
        if st.button("▶️ Start Quantum Stream"):
            stream_placeholder = st.empty()
            
            for i in range(stream_duration):
                # Simulate streaming data with anomalies
                batch_size = 10 if stream_rate == "Fast" else 5 if stream_rate == "Medium" else 2
                
                normal_batch = np.random.multivariate_normal([0, 0, 0, 0], np.eye(4), batch_size)
                
                # Inject anomalies based on rate
                if np.random.random() < (anomaly_injection / 100):
                    anomaly_batch = np.random.multivariate_normal([5, 5, 5, 5], np.eye(4), 1)
                    batch_data = np.vstack([normal_batch, anomaly_batch])
                else:
                    batch_data = normal_batch
                
                new_df = pd.DataFrame(batch_data, columns=['Temperature', 'Pressure', 'Vibration', 'Speed'])
                new_df['timestamp'] = pd.Timestamp.now()
                
                if 'real_time_data' not in st.session_state:
                    st.session_state.real_time_data = new_df
                else:
                    st.session_state.real_time_data = pd.concat([st.session_state.real_time_data, new_df])
                
                # Keep last 1000 points
                st.session_state.real_time_data = st.session_state.real_time_data.tail(1000)
                
                with stream_placeholder.container():
                    st.metric("📊 Live Data Points", f"{len(st.session_state.real_time_data):,}")
                    st.metric("⏱️ Stream Duration", f"{i+1}s / {stream_duration}s")
                
                time.sleep(1)
            
            data = st.session_state.real_time_data
            st.success(f"📊 Stream completed! Collected {len(data):,} live data points")
        st.markdown('</div>', unsafe_allow_html=True)

# Advanced anomaly detection and analysis
if 'data' in locals() and data is not None:
    # Enhanced data preprocessing
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if 'True_Anomaly' in numeric_cols:
        numeric_cols.remove('True_Anomaly')
    if 'Risk_Score' in numeric_cols:
        numeric_cols.remove('Risk_Score')
    
    if len(numeric_cols) >= 2:
        X = data[numeric_cols].fillna(data[numeric_cols].mean()).values
        
        # Advanced anomaly detection
        with st.spinner(f"🔍 Running {algorithm} with quantum processing..."):
            model, predictions, scores, scaler = advanced_anomaly_detection(X, algorithm, params)
        
        # Enhanced results processing
        anomaly_labels = (predictions == -1).astype(int)
        n_anomalies = np.sum(anomaly_labels)
        anomaly_percentage = (n_anomalies / len(data)) * 100
        
        # Store results in session state
        st.session_state.anomaly_history.append({
            'timestamp': pd.Timestamp.now(),
            'algorithm': algorithm,
            'n_anomalies': n_anomalies,
            'accuracy': None
        })
        
        with col2:
            st.markdown("## 📊 AI Detection Matrix")
            
            # Animated metrics cards
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="margin:0;">🎯 {n_anomalies:,}</h2>
                <h4 style="margin:5px 0;">Anomalies Detected</h4>
                <p style="margin:0;">Rate: {anomaly_percentage:.2f}%</p>
                <p style="margin:0;">Normal: {len(data) - n_anomalies:,}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Performance metrics with ground truth
            if 'True_Anomaly' in data.columns:
                y_true = data['True_Anomaly'].values
                y_pred = anomaly_labels
                
                accuracy = np.mean(y_true == y_pred)
                precision = np.sum((y_true == 1) & (y_pred == 1)) / max(np.sum(y_pred == 1), 1)
                recall = np.sum((y_true == 1) & (y_pred == 1)) / max(np.sum(y_true == 1), 1)
                f1 = 2 * (precision * recall) / max((precision + recall), 0.001)
                
                try:
                    auc_score = roc_auc_score(y_true, -scores)
                    st.markdown(f"""
                    <div class="status-card">
                        <h3>🎯 Model Performance</h3>
                        <p>Accuracy: {accuracy:.3f}</p>
                        <p>Precision: {precision:.3f}</p>
                        <p>Recall: {recall:.3f}</p>
                        <p>F1-Score: {f1:.3f}</p>
                        <p>AUC-ROC: {auc_score:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Store performance
                    st.session_state.model_performance[algorithm] = {
                        'accuracy': accuracy, 'precision': precision, 
                        'recall': recall, 'f1': f1, 'auc': auc_score
                    }
                    
                except Exception as e:
                    st.warning(f"⚠️ Could not calculate AUC score: {str(e)}")
                    st.markdown(f"""
                    <div class="status-card">
                        <h3>🎯 Model Performance</h3>
                        <p>Accuracy: {accuracy:.3f}</p>
                        <p>Precision: {precision:.3f}</p>
                        <p>Recall: {recall:.3f}</p>
                        <p>F1-Score: {f1:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Anomaly severity classification
            if len(scores) > 0:
                score_percentiles = np.percentile(-scores, [70, 90, 95])
                high_risk = np.sum(-scores > score_percentiles[2])
                medium_risk = np.sum((-scores > score_percentiles[1]) & (-scores <= score_percentiles[2]))
                low_risk = n_anomalies - high_risk - medium_risk
                
                st.markdown(f"""
                <div class="warning-card">
                    <h3>⚠️ Risk Classification</h3>
                    <p>🔴 High Risk: {high_risk}</p>
                    <p>🟡 Medium Risk: {medium_risk}</p>
                    <p>🟢 Low Risk: {low_risk}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Advanced visualizations
        st.markdown("## 🌟 Advanced AI Visualizations")
        
        # Create results DataFrame
        results_df = data[numeric_cols].copy()
        results_df['Anomaly'] = anomaly_labels
        results_df['Anomaly_Score'] = -scores if len(scores) > 0 else 0
        
        if viz_options['3D_neural_viz'] and len(numeric_cols) >= 3:
            st.markdown("### 🌌 3D Neural Space Visualization")
            
            # PCA for 3D visualization if more than 3 features
            if len(numeric_cols) > 3:
                pca = PCA(n_components=3)
                X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
                pca_cols = ['PC1', 'PC2', 'PC3']
                
                fig = go.Figure(data=[
                    go.Scatter3d(
                        x=X_pca[anomaly_labels == 0, 0],
                        y=X_pca[anomaly_labels == 0, 1],
                        z=X_pca[anomaly_labels == 0, 2],
                        mode='markers',
                        marker=dict(size=4, color='lightblue', opacity=0.6),
                        name='Normal Points',
                        hovertemplate='<b>Normal Point</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>'
                    ),
                    go.Scatter3d(
                        x=X_pca[anomaly_labels == 1, 0],
                        y=X_pca[anomaly_labels == 1, 1],
                        z=X_pca[anomaly_labels == 1, 2],
                        mode='markers',
                        marker=dict(size=8, color='red', opacity=0.9),
                        name='Anomalies',
                        hovertemplate='<b>Anomaly</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>'
                    )
                ])
                
                st.markdown(f"**Explained Variance Ratio:** PC1: {pca.explained_variance_ratio_[0]:.2%}, PC2: {pca.explained_variance_ratio_[1]:.2%}, PC3: {pca.explained_variance_ratio_[2]:.2%}")
            else:
                fig = go.Figure(data=[
                    go.Scatter3d(
                        x=X[anomaly_labels == 0, 0],
                        y=X[anomaly_labels == 0, 1],
                        z=X[anomaly_labels == 0, 2],
                        mode='markers',
                        marker=dict(size=4, color='lightblue', opacity=0.6),
                        name='Normal Points'
                    ),
                    go.Scatter3d(
                        x=X[anomaly_labels == 1, 0],
                        y=X[anomaly_labels == 1, 1],
                        z=X[anomaly_labels == 1, 2],
                        mode='markers',
                        marker=dict(size=8, color='red', opacity=0.9),
                        name='Anomalies'
                    )
                ])
            
            fig.update_layout(
                title="3D Neural Space - Anomaly Detection",
                scene=dict(
                    xaxis_title=pca_cols[0] if len(numeric_cols) > 3 else numeric_cols[0],
                    yaxis_title=pca_cols[1] if len(numeric_cols) > 3 else numeric_cols[1],
                    zaxis_title=pca_cols[2] if len(numeric_cols) > 3 else numeric_cols[2],
                    bgcolor='rgba(0,0,0,0.1)'
                ),
                height=600,
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if viz_options['correlation_matrix']:
            st.markdown("### 🔗 Advanced Correlation Matrix")
            
            corr_matrix = results_df[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title="Feature Correlation Heatmap"
            )
            fig.update_layout(height=500, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        
        if viz_options['time_series_analysis'] and 'timestamp' in data.columns:
            st.markdown("### ⏰ Time Series Anomaly Analysis")
            
            # Create time series plot
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Feature Values Over Time', 'Anomaly Detection Timeline'),
                shared_xaxes=True,
                vertical_spacing=0.1
            )
            
            # Plot first few features
            for i, col in enumerate(numeric_cols[:3]):
                fig.add_trace(
                    go.Scatter(
                        x=data['timestamp'],
                        y=data[col],
                        mode='lines',
                        name=col,
                        opacity=0.7
                    ),
                    row=1, col=1
                )
            
            # Highlight anomalies
            anomaly_data_ts = data[data.index.isin(np.where(anomaly_labels == 1)[0])]
            
            fig.add_trace(
                go.Scatter(
                    x=anomaly_data_ts['timestamp'],
                    y=[1] * len(anomaly_data_ts),
                    mode='markers',
                    name='Anomalies',
                    marker=dict(color='red', size=10, symbol='x'),
                    yaxis='y2'
                ),
                row=2, col=1
            )
            
            fig.update_layout(height=600, template="plotly_dark", title="Time Series Analysis")
            st.plotly_chart(fig, use_container_width=True)
        
        if viz_options['feature_importance'] and hasattr(model, 'feature_importances_') or 'Isolation Forest' in algorithm:
            st.markdown("### 🎯 Feature Importance Analysis")
            
            if 'Isolation Forest' in algorithm:
                # Calculate feature importance for Isolation Forest
                feature_importance = np.mean([tree.tree_.compute_feature_importances(normalize=False) 
                                            for tree in model.estimators_], axis=0)
            else:
                feature_importance = np.random.random(len(numeric_cols))  # Placeholder for other algorithms
            
            importance_df = pd.DataFrame({
                'Feature': numeric_cols,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Feature Importance in Anomaly Detection",
                color='Importance',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=400, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        
        if viz_options['anomaly_clustering'] and n_anomalies > 0:
            st.markdown("### 🔍 Anomaly Clustering Analysis")
            
            if n_anomalies > 1:
                # Cluster the anomalies
                anomaly_indices = np.where(anomaly_labels == 1)[0]
                anomaly_features = X[anomaly_indices]
                
                if len(anomaly_features) >= 2:
                    from sklearn.cluster import KMeans
                    n_clusters = min(3, len(anomaly_features))
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    anomaly_clusters = kmeans.fit_predict(StandardScaler().fit_transform(anomaly_features))
                    
                    # PCA for 2D visualization
                    pca_2d = PCA(n_components=2)
                    X_pca_2d = pca_2d.fit_transform(StandardScaler().fit_transform(X))
                    
                    # Create cluster visualization
                    fig = go.Figure()
                    
                    # Plot normal points
                    fig.add_trace(go.Scatter(
                        x=X_pca_2d[anomaly_labels == 0, 0],
                        y=X_pca_2d[anomaly_labels == 0, 1],
                        mode='markers',
                        marker=dict(color='lightblue', size=4, opacity=0.6),
                        name='Normal Points'
                    ))
                    
                    # Plot anomaly clusters
                    colors = ['red', 'orange', 'purple', 'pink', 'brown']
                    for cluster_id in range(n_clusters):
                        cluster_mask = anomaly_clusters == cluster_id
                        cluster_indices = anomaly_indices[cluster_mask]
                        
                        fig.add_trace(go.Scatter(
                            x=X_pca_2d[cluster_indices, 0],
                            y=X_pca_2d[cluster_indices, 1],
                            mode='markers',
                            marker=dict(color=colors[cluster_id % len(colors)], size=8, opacity=0.9),
                            name=f'Anomaly Cluster {cluster_id + 1}'
                        ))
                    
                    fig.update_layout(
                        title="Anomaly Clustering in PCA Space",
                        xaxis_title="First Principal Component",
                        yaxis_title="Second Principal Component",
                        template="plotly_dark",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Cluster statistics
                    cluster_stats = []
                    for cluster_id in range(n_clusters):
                        cluster_mask = anomaly_clusters == cluster_id
                        cluster_size = np.sum(cluster_mask)
                        cluster_stats.append(f"Cluster {cluster_id + 1}: {cluster_size} anomalies")
                    
                    st.info("📊 " + " | ".join(cluster_stats))
        
        # Enhanced data table with anomalies highlighted
        st.markdown("### 📋 Detailed Anomaly Report")
        
        # Filter options
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        with col_filter1:
            show_filter = st.selectbox("Show", ["All Data", "Anomalies Only", "Normal Only"])
        with col_filter2:
            sort_by = st.selectbox("Sort by", ["Index", "Anomaly Score", "Risk Level"])
        with col_filter3:
            max_rows = st.number_input("Max Rows", 10, 1000, 100)
        
        # Apply filters
        display_df = results_df.copy()
        if show_filter == "Anomalies Only":
            display_df = display_df[display_df['Anomaly'] == 1]
        elif show_filter == "Normal Only":
            display_df = display_df[display_df['Anomaly'] == 0]
        
        # Sort data
        if sort_by == "Anomaly Score":
            display_df = display_df.sort_values('Anomaly_Score', ascending=False)
        elif sort_by == "Risk Level":
            display_df = display_df.sort_values('Anomaly_Score', ascending=False)
        
        # Limit rows
        display_df = display_df.head(max_rows)
        
        # Style the dataframe
        def highlight_anomalies(row):
            if row['Anomaly'] == 1:
                return ['background-color: rgba(255, 99, 71, 0.3)'] * len(row)
            return [''] * len(row)
        
        st.dataframe(
            display_df.style.apply(highlight_anomalies, axis=1),
            use_container_width=True,
            height=400
        )
        
        # AI-powered report generation
        st.markdown('<div class="ai-report-container">', unsafe_allow_html=True)
        st.markdown("## 🤖 AI-Powered Anomaly Analysis Report")
        
        if groq_api_key and n_anomalies > 0:
            if st.button("🚀 Generate Comprehensive AI Report", type="primary"):
                with st.spinner("🧠 AI analyzing anomaly patterns..."):
                    anomaly_data = results_df[results_df['Anomaly'] == 1]
                    ai_report = generate_ai_report(anomaly_data, algorithm, groq_api_key)
                    st.markdown(ai_report)
        else:
            if not groq_api_key:
                st.warning("🔑 Enter your Groq API key to unlock AI-powered report generation")
            else:
                st.info("📊 No anomalies detected - AI report not applicable")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Export functionality
        st.markdown("## 📤 Export & Download")
        col_export1, col_export2, col_export3 = st.columns(3)
        
        with col_export1:
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                "📄 Download CSV Report",
                csv_data,
                "anomaly_report.csv",
                "text/csv"
            )
        
        with col_export2:
            json_data = results_df.to_json(orient='records', indent=2)
            st.download_button(
                "📋 Download JSON Report",
                json_data,
                "anomaly_report.json",
                "application/json"
            )
        
        with col_export3:
            summary_report = f"""
            ANOMALY DETECTION SUMMARY REPORT
            =====================================
            Algorithm: {algorithm}
            Total Samples: {len(data):,}
            Anomalies Detected: {n_anomalies:,}
            Anomaly Rate: {anomaly_percentage:.2f}%
            Features Analyzed: {', '.join(numeric_cols)}
            Detection Timestamp: {pd.Timestamp.now()}
            
            Performance Metrics:
            {'-' * 20}
            """
            if 'True_Anomaly' in data.columns:
                summary_report += f"""
            Accuracy: {accuracy:.3f}
            Precision: {precision:.3f}
            Recall: {recall:.3f}
            F1-Score: {f1:.3f}
                """
            
            st.download_button(
                "📊 Download Summary",
                summary_report,
                "anomaly_summary.txt",
                "text/plain"
            )
    
    else:
        st.error("❌ Insufficient numeric features for anomaly detection. Need at least 2 numeric columns.")

# Performance history tracking
if len(st.session_state.anomaly_history) > 0:
    st.markdown("## 📈 Detection History & Analytics")
    
    history_df = pd.DataFrame(st.session_state.anomaly_history)
    
    # History visualization
    fig = px.line(
        history_df, 
        x='timestamp', 
        y='n_anomalies',
        color='algorithm',
        title="Anomaly Detection History",
        markers=True
    )
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin-top: 2rem;">
    <h3>🔮 AnomalyHunter AI Pro</h3>
    <p>Powered by Next-Generation AI • Real-time Intelligence • Enterprise Grade</p>
    <p>🚀 Built with Streamlit, Plotly, Scikit-learn & Groq AI</p>
</div>
""", unsafe_allow_html=True)
