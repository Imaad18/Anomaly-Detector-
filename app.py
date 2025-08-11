import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="🚀 AnomalyHunter Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .algorithm-box {
        border: 2px solid #4ECDC4;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background: rgba(78, 205, 196, 0.1);
    }
    
    .stSelectbox > div > div {
        background-color: #2E86AB;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🚀 AnomalyHunter Pro</h1>', unsafe_allow_html=True)
st.markdown("### *Next-Generation Anomaly Detection Platform*")

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 Configuration Panel")
    
    # Data source selection
    data_source = st.selectbox(
        "📊 Select Data Source",
        ["Upload CSV", "Generate Synthetic Data", "Live Data Stream"]
    )
    
    # Algorithm selection
    algorithm = st.selectbox(
        "🤖 Choose Algorithm",
        ["Isolation Forest", "DBSCAN", "One-Class SVM", "Local Outlier Factor", "Statistical Z-Score", "Ensemble Method"]
    )
    
    # Parameters
    st.markdown("### ⚙️ Algorithm Parameters")
    
    if algorithm == "Isolation Forest":
        contamination = st.slider("Contamination Rate", 0.01, 0.5, 0.1)
        n_estimators = st.slider("Number of Estimators", 10, 200, 100)
    elif algorithm == "DBSCAN":
        eps = st.slider("Epsilon", 0.1, 2.0, 0.5)
        min_samples = st.slider("Min Samples", 2, 20, 5)
    elif algorithm == "One-Class SVM":
        nu = st.slider("Nu (Outlier Fraction)", 0.01, 0.5, 0.1)
        gamma = st.selectbox("Gamma", ["scale", "auto"])
    elif algorithm == "Local Outlier Factor":
        n_neighbors = st.slider("Number of Neighbors", 5, 50, 20)
        contamination = st.slider("Contamination Rate", 0.01, 0.5, 0.1)
    
    # Visualization options
    st.markdown("### 🎨 Visualization")
    show_3d = st.checkbox("3D Visualization", value=True)
    show_heatmap = st.checkbox("Correlation Heatmap", value=True)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Data loading/generation
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(data)} records with {len(data.columns)} features")
        else:
            st.info("👆 Please upload a CSV file to get started")
            data = None
    
    elif data_source == "Generate Synthetic Data":
        st.markdown("### 🧬 Synthetic Data Generator")
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            n_samples = st.number_input("Number of Samples", 100, 10000, 1000)
        with col_b:
            n_features = st.number_input("Number of Features", 2, 20, 5)
        with col_c:
            anomaly_rate = st.slider("Anomaly Rate (%)", 1, 20, 10)
        
        if st.button("🎲 Generate Data", type="primary"):
            # Generate normal data
            np.random.seed(42)
            normal_data = np.random.multivariate_normal(
                mean=np.zeros(n_features),
                cov=np.eye(n_features),
                size=int(n_samples * (1 - anomaly_rate/100))
            )
            
            # Generate anomalies
            n_anomalies = int(n_samples * anomaly_rate/100)
            anomaly_data = np.random.multivariate_normal(
                mean=np.ones(n_features) * 3,
                cov=np.eye(n_features) * 2,
                size=n_anomalies
            )
            
            # Combine data
            all_data = np.vstack([normal_data, anomaly_data])
            true_labels = np.hstack([np.zeros(len(normal_data)), np.ones(len(anomaly_data))])
            
            # Create DataFrame
            columns = [f'Feature_{i+1}' for i in range(n_features)]
            data = pd.DataFrame(all_data, columns=columns)
            data['True_Anomaly'] = true_labels
            
            st.success(f"✅ Generated {len(data)} samples with {anomaly_rate}% anomalies")
    
    elif data_source == "Live Data Stream":
        st.markdown("### 📡 Live Data Stream Simulation")
        
        if 'stream_data' not in st.session_state:
            st.session_state.stream_data = pd.DataFrame()
        
        if st.button("▶️ Start Stream"):
            # Simulate streaming data
            new_batch = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 50)
            # Add some anomalies
            anomalies = np.random.multivariate_normal([4, 4], [[1, 0], [0, 1]], 5)
            batch_data = np.vstack([new_batch, anomalies])
            
            new_df = pd.DataFrame(batch_data, columns=['X', 'Y'])
            st.session_state.stream_data = pd.concat([st.session_state.stream_data, new_df])
            
            data = st.session_state.stream_data.tail(1000)  # Keep last 1000 points
            st.success(f"📊 Streaming... {len(data)} points collected")

# Anomaly detection logic
if 'data' in locals() and data is not None:
    # Prepare data for analysis
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if 'True_Anomaly' in numeric_cols:
        numeric_cols = numeric_cols.drop('True_Anomaly')
    
    if len(numeric_cols) >= 2:
        X = data[numeric_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply selected algorithm
        with st.spinner(f"🔍 Running {algorithm}..."):
            if algorithm == "Isolation Forest":
                model = IsolationForest(contamination=contamination, n_estimators=n_estimators, random_state=42)
                predictions = model.fit_predict(X_scaled)
                scores = model.decision_function(X_scaled)
                
            elif algorithm == "DBSCAN":
                model = DBSCAN(eps=eps, min_samples=min_samples)
                cluster_labels = model.fit_predict(X_scaled)
                predictions = np.where(cluster_labels == -1, -1, 1)
                scores = np.zeros(len(predictions))  # DBSCAN doesn't provide scores
                
            elif algorithm == "One-Class SVM":
                model = OneClassSVM(nu=nu, gamma=gamma)
                predictions = model.fit_predict(X_scaled)
                scores = model.decision_function(X_scaled)
                
            elif algorithm == "Local Outlier Factor":
                model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
                predictions = model.fit_predict(X_scaled)
                scores = model.negative_outlier_factor_
                
            elif algorithm == "Statistical Z-Score":
                z_scores = np.abs(stats.zscore(X_scaled, axis=0))
                max_z_scores = np.max(z_scores, axis=1)
                threshold = st.sidebar.slider("Z-Score Threshold", 1.0, 5.0, 3.0)
                predictions = np.where(max_z_scores > threshold, -1, 1)
                scores = -max_z_scores
                
            elif algorithm == "Ensemble Method":
                # Combine multiple algorithms
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                svm = OneClassSVM(nu=0.1)
                lof = LocalOutlierFactor(contamination=0.1)
                
                pred1 = iso_forest.fit_predict(X_scaled)
                pred2 = svm.fit_predict(X_scaled)
                pred3 = lof.fit_predict(X_scaled)
                
                # Majority voting
                ensemble_pred = np.array([pred1, pred2, pred3])
                predictions = np.where(np.sum(ensemble_pred == -1, axis=0) >= 2, -1, 1)
                scores = iso_forest.decision_function(X_scaled)
        
        # Convert predictions to binary (1 for normal, 0 for anomaly)
        anomaly_labels = (predictions == -1).astype(int)
        
        # Results summary
        n_anomalies = np.sum(anomaly_labels)
        anomaly_percentage = (n_anomalies / len(data)) * 100
        
        with col2:
            st.markdown("## 📊 Detection Results")
            
            st.markdown(f"""
            <div class="metric-container">
                <h3>🎯 {n_anomalies} Anomalies Detected</h3>
                <p>Anomaly Rate: {anomaly_percentage:.2f}%</p>
                <p>Normal Points: {len(data) - n_anomalies}</p>
                <p>Algorithm: {algorithm}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Performance metrics (if ground truth available)
            if 'True_Anomaly' in data.columns:
                from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
                
                y_true = data['True_Anomaly'].values
                y_pred = anomaly_labels
                
                accuracy = np.mean(y_true == y_pred)
                if len(np.unique(y_true)) > 1 and len(np.unique(scores)) > 1:
                    try:
                        auc_score = roc_auc_score(y_true, -scores)
                        st.metric("🎯 AUC Score", f"{auc_score:.3f}")
                    except:
                        st.metric("🎯 Accuracy", f"{accuracy:.3f}")
                
                st.metric("🎯 Accuracy", f"{accuracy:.3f}")
        
        # Visualizations
        st.markdown("## 📈 Interactive Visualizations")
        
        # Main scatter plot
        if len(numeric_cols) >= 2:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Anomaly Detection Results', 'Anomaly Scores Distribution', 
                              'Feature Correlation', '3D Visualization' if show_3d and len(numeric_cols) >= 3 else 'PCA Projection'),
                specs=[[{"type": "scatter"}, {"type": "histogram"}],
                       [{"type": "heatmap"}, {"type": "scatter3d" if show_3d and len(numeric_cols) >= 3 else "scatter"}]]
            )
            
            # Main scatter plot
            colors = ['red' if x == 1 else 'blue' for x in anomaly_labels]
            fig.add_trace(
                go.Scatter(
                    x=X[:, 0], y=X[:, 1],
                    mode='markers',
                    marker=dict(color=colors, size=8, opacity=0.7),
                    name='Data Points',
                    text=[f'Point {i}<br>Anomaly: {"Yes" if anomaly_labels[i] else "No"}' for i in range(len(X))],
                    hovertemplate='%{text}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Anomaly scores histogram
            fig.add_trace(
                go.Histogram(x=scores, name='Anomaly Scores', opacity=0.7),
                row=1, col=2
            )
            
            # Correlation heatmap
            if show_heatmap:
                corr_matrix = data[numeric_cols].corr()
                fig.add_trace(
                    go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='RdBu',
                        name='Correlation'
                    ),
                    row=2, col=1
                )
            
            # 3D or PCA plot
            if show_3d and len(numeric_cols) >= 3:
                fig.add_trace(
                    go.Scatter3d(
                        x=X[:, 0], y=X[:, 1], z=X[:, 2],
                        mode='markers',
                        marker=dict(color=colors, size=5),
                        name='3D View'
                    ),
                    row=2, col=2
                )
            else:
                # PCA projection
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                
                fig.add_trace(
                    go.Scatter(
                        x=X_pca[:, 0], y=X_pca[:, 1],
                        mode='markers',
                        marker=dict(color=colors, size=8, opacity=0.7),
                        name='PCA Projection'
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(height=800, showlegend=True, title_text="🔍 Comprehensive Anomaly Analysis")
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance (for tree-based methods)
        if hasattr(model, 'feature_importances_') or algorithm == "Isolation Forest":
            st.markdown("## 🎯 Feature Importance")
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': numeric_cols,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                               title="Feature Importance for Anomaly Detection")
                st.plotly_chart(fig_imp, use_container_width=True)
        
        # Detailed results table
        st.markdown("## 📋 Detailed Results")
        
        result_df = data.copy()
        result_df['Anomaly_Detected'] = anomaly_labels
        result_df['Anomaly_Score'] = scores
        result_df['Risk_Level'] = pd.cut(scores, bins=3, labels=['Low', 'Medium', 'High'])
        
        # Filter options
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            show_anomalies_only = st.checkbox("Show Anomalies Only")
        with col_filter2:
            risk_filter = st.multiselect("Risk Level Filter", ['Low', 'Medium', 'High'], 
                                       default=['Low', 'Medium', 'High'])
        
        # Apply filters
        filtered_df = result_df.copy()
        if show_anomalies_only:
            filtered_df = filtered_df[filtered_df['Anomaly_Detected'] == 1]
        
        filtered_df = filtered_df[filtered_df['Risk_Level'].isin(risk_filter)]
        
        st.dataframe(filtered_df, use_container_width=True)
        
        # Download results
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f'anomaly_detection_results_{algorithm.lower().replace(" ", "_")}.csv',
            mime='text/csv'
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🚀 <strong>AnomalyHunter Pro</strong> - Built for Hackathon Excellence</p>
    <p>Powered by Advanced ML Algorithms | Real-time Detection | Interactive Visualizations</p>
</div>
""", unsafe_allow_html=True)
