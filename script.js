import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.graph_objects as go
import plotly.figure_factory as ff
import math
import random
import time
from io import StringIO
from scipy import stats

class QuantumTunnelingPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
    
    def parse_experimental_data(self, text_data):
        """Parse the specific format of quantum tunneling data"""
        lines = text_data.split('\n')
        data = []
        barrier_data = []
        
        for i, line in enumerate(lines):
            if 'LDR Value:' in line and '|' in line:
                try:
                    parts = line.split('|')
                    ldr_value = float(parts[0].split(':')[1].strip())
                    threshold = float(parts[1].split(':')[1].strip())
                    random_val = float(parts[2].split(':')[1].strip())
                    probability = float(parts[3].split(':')[1].strip())
                    tunneling = 'YES' in parts[4] if len(parts) > 4 else 'NO'
                    
                    data.append({
                        'measurement_id': i,
                        'ldr_value': ldr_value,
                        'threshold': threshold,
                        'random': random_val,
                        'probability': probability,
                        'tunneling': tunneling,
                        'timestamp': pd.Timestamp.now() - pd.Timedelta(seconds=(len(data)))
                    })
                except:
                    continue
            
            elif 'Barrier:' in line:
                barrier = line.split(':')[1].strip()
                barrier_count = barrier.count('#')
                dot_count = barrier.count('.')
                barrier_data.append({
                    'measurement_id': len(barrier_data),
                    'barrier_strength': barrier_count,
                    'barrier_gaps': dot_count
                })
                
        return pd.DataFrame(data), pd.DataFrame(barrier_data)
    
    def calculate_statistics(self, df):
        """Calculate advanced statistics for the tunneling data"""
        stats_dict = {
            'Total Measurements': len(df),
            'Tunneling Events': df['tunneling'].value_counts().get('YES', 0),
            'Tunneling Rate': (df['tunneling'].value_counts().get('YES', 0) / len(df)) * 100,
            'Average LDR': df['ldr_value'].mean(),
            'LDR Std Dev': df['ldr_value'].std(),
            'Average Probability': df['probability'].mean(),
            'Probability Std Dev': df['probability'].std(),
            'Correlation (LDR vs Prob)': df['ldr_value'].corr(df['probability'])
        }
        return stats_dict
    
    def simulate_tunneling(self, ldr_value):
        """Simulate tunneling probability with enhanced physics modeling"""
        base_probability = (math.sin(ldr_value/100) + 1) / 2
        noise = random.uniform(-0.1, 0.1)
        potential_barrier = math.exp(-ldr_value / 500)
        quantum_factor = 1 / (1 + math.exp(-(ldr_value - 500)/100))
        probability = max(0, min(1, (base_probability + noise) * potential_barrier * quantum_factor))
        return probability

def create_heatmap(df):
    """Create a 2D heatmap of tunneling events"""
    heatmap_data = df.pivot_table(
        values='probability',
        index=pd.qcut(df['ldr_value'], 10),
        columns=pd.qcut(df['random'], 10),
        aggfunc='mean'
    )
    return go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=np.arange(10),
        y=np.arange(10),
        colorscale='Viridis',
        colorbar=dict(title='Probability')
    ))

def main():
    st.title("🌌 Enhanced Quantum Tunneling Analyzer")
    
    st.sidebar.title("Navigation")
    mode = st.sidebar.radio("Select Mode", ["Upload Data", "Single Prediction", "Real-time Simulation", "Advanced Analytics"])
    
    predictor = QuantumTunnelingPredictor()
    
    if mode == "Upload Data":
        st.write("""
        ### 📤 Upload Experimental Data
        Upload your quantum tunneling experiment data for advanced visualization and analysis.
        """)
        
        uploaded_file = st.file_uploader("Upload your experiment data (TXT)", type=['txt'])
        
        if uploaded_file is not None:
            raw_data = uploaded_file.read().decode()
            df, barrier_df = predictor.parse_experimental_data(raw_data)
            
            if not df.empty:
                # Calculate statistics
                stats = predictor.calculate_statistics(df)
                
                # Enhanced Data Overview with more metrics
                st.header("📊 Advanced Data Overview")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Measurements", stats['Total Measurements'])
                    st.metric("Average LDR", f"{stats['Average LDR']:.1f}")
                with col2:
                    st.metric("Tunneling Events", f"{stats['Tunneling Events']}")
                    st.metric("LDR Std Dev", f"{stats['LDR Std Dev']:.1f}")
                with col3:
                    st.metric("Tunneling Rate", f"{stats['Tunneling Rate']:.1f}%")
                    st.metric("Avg Probability", f"{stats['Average Probability']:.3f}")
                with col4:
                    st.metric("Correlation", f"{stats['Correlation (LDR vs Prob)']:.3f}")
                    st.metric("Prob Std Dev", f"{stats['Probability Std Dev']:.3f}")

                # Main Visualization
                st.header("📈 Quantum Tunneling Visualization")
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df['measurement_id'],
                    y=df['ldr_value'],
                    name='LDR Value',
                    mode='lines+markers',
                    line=dict(color='blue'),
                    marker=dict(
                        size=8,
                        color=df['tunneling'].map({'YES': 'red', 'NO': 'blue'})
                    )
                ))
                
                fig.add_trace(go.Scatter(
                    x=df['measurement_id'],
                    y=df['threshold'],
                    name='Threshold',
                    mode='lines',
                    line=dict(color='red', dash='dash')
                ))
                
                if not barrier_df.empty:
                    fig.add_trace(go.Bar(
                        x=barrier_df['measurement_id'],
                        y=barrier_df['barrier_strength'],
                        name='Barrier Strength',
                        opacity=0.3,
                        marker_color='gray'
                    ))
                
                fig.update_layout(
                    title="Quantum Tunneling Time Series Analysis",
                    xaxis_title="Measurement Index",
                    yaxis_title="Value",
                    height=600,
                    showlegend=True,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig)

                # Distribution Analysis
                st.subheader("📊 Probability Distribution Analysis")
                fig_dist = go.Figure()
                
                # Add histogram for tunneling events
                tunneling_data = df[df['tunneling'] == 'YES']['probability']
                no_tunneling_data = df[df['tunneling'] == 'NO']['probability']
                
                if not tunneling_data.empty:
                    fig_dist.add_trace(go.Histogram(
                        x=tunneling_data,
                        name='Tunneling Events',
                        opacity=0.75
                    ))
                
                if not no_tunneling_data.empty:
                    fig_dist.add_trace(go.Histogram(
                        x=no_tunneling_data,
                        name='No Tunneling',
                        opacity=0.75
                    ))
                
                fig_dist.update_layout(
                    title="Probability Distribution by Outcome",
                    xaxis_title="Probability",
                    yaxis_title="Count",
                    barmode='overlay',
                    height=500
                )
                
                st.plotly_chart(fig_dist)

                # Phase Space Plot
                st.subheader("🌌 Phase Space Analysis")
                fig_phase = px.scatter(
                    df,
                    x='ldr_value',
                    y='probability',
                    color='tunneling',
                    size='random',
                    hover_data=['measurement_id'],
                    title='Quantum Tunneling Phase Space'
                )
                
                fig_phase.update_layout(height=500)
                st.plotly_chart(fig_phase)

                # Tunneling rate over time
                st.subheader("📈 Tunneling Rate Analysis")
                window_size = st.slider("Moving Average Window Size", 5, 50, 20)
                df['tunneling_binary'] = (df['tunneling'] == 'YES').astype(int)
                df['tunneling_rate_ma'] = df['tunneling_binary'].rolling(window=window_size).mean()
                
                fig_rate = go.Figure()
                fig_rate.add_trace(go.Scatter(
                    x=df['measurement_id'],
                    y=df['tunneling_rate_ma'],
                    mode='lines',
                    name=f'Tunneling Rate ({window_size}-point MA)',
                    line=dict(color='purple')
                ))
                
                fig_rate.update_layout(
                    title=f"Tunneling Rate Over Time ({window_size}-point Moving Average)",
                    xaxis_title="Measurement Index",
                    yaxis_title="Tunneling Rate",
                    height=400
                )
                
                st.plotly_chart(fig_rate)

    elif mode == "Single Prediction":
        st.write("### 🔍 Single Value Prediction")
        ldr_value = st.slider("Enter LDR value", 0, 1023, 450)
        
        if st.button("Calculate Probability"):
            probability = predictor.simulate_tunneling(ldr_value)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Tunneling Probability"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgray"},
                        {'range': [30, 70], 'color': "gray"},
                        {'range': [70, 100], 'color': "darkgray"}
                    ]
                }
            ))
            st.plotly_chart(fig)
            
            # Barrier Visualization
            barrier_length = 50
            barrier_strength = int((1 - probability) * barrier_length)
            barrier = '#' * barrier_strength + '.' * (barrier_length - barrier_strength)
            st.code(f"Barrier: {barrier}", language=None)

    elif mode == "Real-time Simulation":
        st.write("### ⚡ Real-time Simulation")
        chart_placeholder = st.empty()
        data = []
        
        if st.button("Start Simulation"):
            for _ in range(50):
                ldr_value = random.uniform(0, 1023)
                probability = predictor.simulate_tunneling(ldr_value)
                data.append({
                    'LDR Value': ldr_value,
                    'Probability': probability,
                    'Timestamp': len(data)
                })
                
                df = pd.DataFrame(data)
                fig = px.line(
                    df, 
                    x='Timestamp',
                    y=['LDR Value', 'Probability'],
                    title='Real-time Tunneling Simulation'
                )
                chart_placeholder.plotly_chart(fig)
                
                time.sleep(0.1)

    else:  # Advanced Analytics mode
# [Previous code remains the same until the Advanced Analytics mode]

    else:  # Advanced Analytics mode
        st.write("### 🔬 Advanced Analytics Dashboard")
        
        # Initialize these variables at the start of the Advanced Analytics section
        tunneling_prob = pd.Series()
        no_tunneling_prob = pd.Series()
        
        # Simulation parameters
        st.sidebar.subheader("Simulation Parameters")
        num_samples = st.sidebar.slider("Number of Samples", 100, 1000, 500)
        noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.1)
        
        if st.button("Run Advanced Analysis"):
            # Generate synthetic data
            ldr_values = np.random.uniform(0, 1023, num_samples)
            probabilities = np.array([predictor.simulate_tunneling(ldr) for ldr in ldr_values])
            tunneling = np.random.random(num_samples) < probabilities
            
            # Create DataFrame
            analysis_df = pd.DataFrame({
                'ldr_value': ldr_values,
                'probability': probabilities,
                'tunneling': tunneling
            })
            
            # Update these variables after creating the DataFrame
            tunneling_prob = analysis_df[analysis_df['tunneling']]['probability']
            no_tunneling_prob = analysis_df[~analysis_df['tunneling']]['probability']
            
            # Advanced visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Probability density estimation
                if len(tunneling_prob) > 0 and len(no_tunneling_prob) > 0:
                    try:
                        fig_kde = ff.create_distplot(
                            [tunneling_prob.tolist(), no_tunneling_prob.tolist()],
                            ['Tunneling', 'No Tunneling'],
                            bin_size=0.02
                        )
                        fig_kde.update_layout(title="Probability Density Estimation")
                        st.plotly_chart(fig_kde)
                    except Exception as e:
                        st.write("Error creating density plot:", str(e))
                else:
                    st.write("Insufficient data for density estimation")
            
            with col2:
                # Quantum regime analysis
                fig_quantum = px.scatter(
                    analysis_df,
                    x='ldr_value',
                    y='probability',
                    color=analysis_df['tunneling'].astype(str),
                    title="Quantum Regime Analysis"
                )
                st.plotly_chart(fig_quantum)
            
            # Statistical tests
            st.subheader("📊 Statistical Tests")
            
            if len(tunneling_prob) > 0 and len(no_tunneling_prob) > 0:
                try:
                    ks_stat, ks_pval = stats.ks_2samp(tunneling_prob, no_tunneling_prob)
                    st.write(f"Kolmogorov-Smirnov Test:")
                    st.write(f"- Statistic: {ks_stat:.4f}")
                    st.write(f"- p-value: {ks_pval:.4f}")
                except Exception as e:
                    st.write("Error performing KS test:", str(e))
            else:
                st.write("Insufficient data for Kolmogorov-Smirnov Test")
            
            if len(analysis_df['probability']) > 1:
                try:
                    ci = stats.t.interval(
                        0.95,
                        len(analysis_df['probability'])-1,
                        loc=analysis_df['probability'].mean(),
                        scale=stats.sem(analysis_df['probability'])
                    )
                    st.write(f"95% Confidence Interval for Probability: [{ci[0]:.4f}, {ci[1]:.4f}]")
                except Exception as e:
                    st.write("Error calculating confidence interval:", str(e))
            else:
                st.write("Insufficient data for confidence interval calculation")

if __name__ == "__main__":
    main()
