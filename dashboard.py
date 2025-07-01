#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Dashboard for Time Series Forecasting Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Time Series Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import project modules with error handling
try:
    from config import DATA_DIR, RESULTS_DIR
except ImportError:
    DATA_DIR = "data/"
    RESULTS_DIR = "results/"
    logger.warning("Could not import config, using default directories")

# Custom CSS styling
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def load_data():
    """Load and cache the latest financial data"""
    try:
        # Try to load processed data files in order of preference
        data_files = [
            os.path.join(DATA_DIR, "data_with_regimes.csv"),
            os.path.join(DATA_DIR, "enhanced_features.csv"),
            os.path.join(DATA_DIR, "raw_financial_data.csv")
        ]
        
        for file_path in data_files:
            if os.path.exists(file_path):
                try:
                    data = pd.read_csv(file_path)
                    if 'date' in data.columns:
                        data['date'] = pd.to_datetime(data['date'])
                    logger.info(f"Successfully loaded data from {file_path}")
                    return data
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
                    continue
        
        # If no processed data exists, show a message
        st.warning("No processed data found. Please run the data pipeline first.")
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        logger.error(f"Data loading error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def load_model_results():
    """Load model performance results"""
    try:
        results_files = [
            os.path.join(RESULTS_DIR, "model_performance.csv"),
            os.path.join(RESULTS_DIR, "comprehensive_model_performance.csv")
        ]
        
        for file_path in results_files:
            if os.path.exists(file_path):
                data = pd.read_csv(file_path)
                logger.info(f"Loaded model results from {file_path}")
                return data
        
        return pd.DataFrame()
        
    except Exception as e:
        logger.warning(f"Could not load model results: {e}")
        return pd.DataFrame()

def create_price_chart(data, symbol):
    """Create interactive price chart"""
    if data.empty or symbol not in data['Symbol'].values:
        return go.Figure().add_annotation(
            text="No data available for selected symbol",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font_size=16
        )
    
    symbol_data = data[data['Symbol'] == symbol].copy()
    symbol_data = symbol_data.sort_values('date')
    
    if symbol_data.empty:
        return go.Figure()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Price Action', 'Volume'],
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Main price chart
    if all(col in symbol_data.columns for col in ['open', 'high', 'low', 'close']):
        fig.add_trace(
            go.Candlestick(
                x=symbol_data['date'],
                open=symbol_data['open'],
                high=symbol_data['high'],
                low=symbol_data['low'],
                close=symbol_data['close'],
                name='Price',
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )
    else:
        # Fallback to line chart if OHLC data not available
        fig.add_trace(
            go.Scatter(
                x=symbol_data['date'],
                y=symbol_data['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
    
    # Volume chart
    if 'volume' in symbol_data.columns:
        fig.add_trace(
            go.Bar(
                x=symbol_data['date'],
                y=symbol_data['volume'],
                name='Volume',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        height=600,
        title=f"{symbol} - Technical Analysis Dashboard",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def create_correlation_heatmap(data):
    """Create cross-market correlation heatmap"""
    if data.empty or 'Symbol' not in data.columns:
        return go.Figure()
    
    try:
        # Pivot data to get symbols as columns
        pivot_data = data.pivot_table(
            index='date',
            columns='Symbol',
            values='close',
            aggfunc='first'
        )
        
        if pivot_data.empty:
            return go.Figure()
        
        # Calculate correlation matrix
        corr_matrix = pivot_data.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=corr_matrix.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Cross-Market Correlation Matrix",
            width=700,
            height=600
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating correlation heatmap: {e}")
        return go.Figure()

def create_model_performance_chart(results_df):
    """Create model performance comparison chart"""
    if results_df.empty:
        return go.Figure()
    
    try:
        # Group by model if multiple entries exist
        if 'model' in results_df.columns:
            model_metrics = results_df.groupby('model').agg({
                col: 'mean' for col in results_df.columns 
                if col != 'model' and pd.api.types.is_numeric_dtype(results_df[col])
            }).round(4)
        else:
            model_metrics = results_df
        
        if model_metrics.empty:
            return go.Figure()
        
        # Create chart for different metrics
        metrics = [col for col in model_metrics.columns if col in ['mae', 'rmse', 'r2', 'mape']]
        
        if not metrics:
            return go.Figure()
        
        fig = go.Figure()
        
        # Add bars for each metric
        for i, metric in enumerate(metrics):
            fig.add_trace(go.Bar(
                name=metric.upper(),
                x=model_metrics.index,
                y=model_metrics[metric],
                offsetgroup=i
            ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Metric Values",
            barmode='group',
            height=500,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating model performance chart: {e}")
        return go.Figure()

def create_attention_heatmap(attention_weights, feature_names=None):
    """Create attention heatmap visualization"""
    if attention_weights is None:
        return go.Figure().add_annotation(
            text="No attention weights available",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font_size=16
        )
    
    try:
        # Convert to numpy array if needed
        if hasattr(attention_weights, 'numpy'):
            attention_weights = attention_weights.numpy()
        
        # If 3D, take mean across first dimension (batch)
        if attention_weights.ndim == 3:
            attention_weights = np.mean(attention_weights, axis=0)
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(attention_weights.shape[-1])]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=attention_weights,
            x=feature_names[:attention_weights.shape[1]] if len(feature_names) >= attention_weights.shape[1] else feature_names,
            y=[f"Time_{i}" for i in range(attention_weights.shape[0])],
            colorscale='Viridis',
            colorbar=dict(title="Attention Weight")
        ))
        
        fig.update_layout(
            title="Cross-Market Attention Heatmap",
            xaxis_title="Features",
            yaxis_title="Time Steps",
            height=500,
            xaxis_tickangle=-45
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating attention heatmap: {e}")
        return go.Figure().add_annotation(
            text=f"Error creating heatmap: {str(e)}",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font_size=14
        )

def create_cross_market_influence_chart(influence_scores):
    """Create cross-market influence scores visualization"""
    if not influence_scores:
        return go.Figure().add_annotation(
            text="No cross-market influence data available",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font_size=16
        )
    
    try:
        # Convert to lists for plotting
        features = list(influence_scores.keys())[:15]  # Top 15 features
        scores = [influence_scores[f] for f in features]
        
        # Create horizontal bar chart
        fig = go.Figure(data=go.Bar(
            y=features,
            x=scores,
            orientation='h',
            marker_color='rgba(55, 83, 109, 0.6)',
            marker_line_color='rgba(55, 83, 109, 1.0)',
            marker_line_width=1
        ))
        
        fig.update_layout(
            title="Cross-Market Influence Scores",
            xaxis_title="Influence Score",
            yaxis_title="Features",
            height=600,
            margin=dict(l=200)  # More space for feature names
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating influence chart: {e}")
        return go.Figure()

def main():
    """Main dashboard application"""
    
    # Header
    st.title("🚀 Time Series Forecasting Dashboard")
    st.markdown("### Cross-Market Predictive Analytics with Regime Detection")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading data..."):
        data = load_data()
        model_results = load_model_results()
    
    # Sidebar controls
    st.sidebar.header("🔧 Dashboard Controls")
    
    # Check if data is available
    if data.empty:
        st.error("❌ No data available. Please run the data pipeline first.")
        st.info("💡 To get started, run: `python main_pipeline.py`")
        return
    
    # Symbol selection
    available_symbols = sorted(data['Symbol'].unique()) if 'Symbol' in data.columns else []
    if available_symbols:
        selected_symbol = st.sidebar.selectbox(
            "📊 Select Symbol",
            available_symbols,
            index=0
        )
    else:
        st.error("No symbols found in data")
        return
    
    # Date range selection
    if 'date' in data.columns:
        min_date = data['date'].min().date()
        max_date = data['date'].max().date()
        
        date_range = st.sidebar.date_input(
            "📅 Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Filter data by date range
        if len(date_range) == 2:
            start_date, end_date = date_range
            data = data[
                (data['date'].dt.date >= start_date) & 
                (data['date'].dt.date <= end_date)
            ]
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Market Overview", 
        "🎯 Analysis", 
        "🤖 Model Performance", 
        "🌐 Cross-Market",
        "🔮 What-If Scenarios"
    ])
    
    # Tab 1: Market Overview
    with tab1:
        st.header("📈 Market Overview")
        
        if selected_symbol and not data.empty:
            symbol_data = data[data['Symbol'] == selected_symbol]
            
            if not symbol_data.empty:
                latest = symbol_data.iloc[-1]
                
                # Key metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    price = latest.get('close', 0)
                    st.metric(
                        "💰 Latest Price", 
                        f"${price:.2f}",
                        delta=f"{latest.get('returns', 0):.2%}" if 'returns' in latest else None
                    )
                
                with col2:
                    volume = latest.get('volume', 0)
                    st.metric("📊 Volume", f"{volume:,.0f}")
                
                with col3:
                    # Try different volatility column names
                    volatility = latest.get('volatility_20', latest.get('volatility_10', latest.get('volatility_5', latest.get('volatility', 0))))
                    st.metric("📉 Volatility", f"{volatility:.4f}")
                
                with col4:
                    # Try different regime column names
                    regime = latest.get('composite_regime', latest.get('volatility_regime', latest.get('regime', 'Unknown')))
                    regime_colors = {
                        'High_Vol': '🔴',
                        'Medium_Vol': '🟡', 
                        'Low_Vol': '🟢',
                        'high_volatility': '🔴',
                        'medium_volatility': '🟡', 
                        'low_volatility': '🟢',
                        'Uptrend': '📈',
                        'Downtrend': '📉',
                        'Sideways': '↔️'
                    }
                    regime_indicator = regime_colors.get(regime, '⚪')
                    st.metric("🎯 Current Regime", f"{regime_indicator} {regime}")
                
                # Price chart
                st.subheader(f"📊 {selected_symbol} Technical Analysis")
                price_chart = create_price_chart(data, selected_symbol)
                st.plotly_chart(price_chart, use_container_width=True)
                
                # Recent data table
                st.subheader("📋 Recent Data Points")
                display_cols = ['date', 'close', 'volume', 'returns']
                available_cols = [col for col in display_cols if col in symbol_data.columns]
                if available_cols:
                    recent_data = symbol_data[available_cols].tail(10)
                    st.dataframe(recent_data, use_container_width=True)
        
        else:
            st.info("Please select a symbol to view market data.")
    
    # Tab 2: Analysis
    with tab2:
        st.header("🎯 Market Analysis")
        
        regime_cols = [col for col in data.columns if 'regime' in col.lower()]
        
        if regime_cols:
            regime_col = regime_cols[0]  # Use first available regime column
            
            # Regime distribution
            col1, col2 = st.columns(2)
            
            with col1:
                regime_counts = data[regime_col].value_counts()
                fig = px.pie(
                    values=regime_counts.values,
                    names=regime_counts.index,
                    title="Market Regime Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Regime statistics
                st.subheader("📊 Regime Statistics")
                
                # Dynamically find available columns for analysis
                analysis_cols = ['returns', 'volume']
                available_analysis_cols = [col for col in analysis_cols if col in data.columns]
                
                # Find volatility column
                volatility_cols = [col for col in ['volatility_20', 'volatility_10', 'volatility_5', 'volatility'] if col in data.columns]
                if volatility_cols:
                    available_analysis_cols.extend(volatility_cols[:1])  # Add first available volatility column
                
                if available_analysis_cols:
                    regime_stats = data.groupby(regime_col)[available_analysis_cols].agg('mean').round(4)
                else:
                    regime_stats = pd.DataFrame()
                
                if not regime_stats.empty:
                    st.dataframe(regime_stats)
                else:
                    st.info("No regime statistics available")
        
        else:
            st.info("No regime analysis available. Run regime detection first.")
    
    # Tab 3: Model Performance
    with tab3:
        st.header("🤖 Model Performance Analysis")
        
        if not model_results.empty:
            # Performance chart
            st.subheader("📊 Model Comparison")
            perf_chart = create_model_performance_chart(model_results)
            st.plotly_chart(perf_chart, use_container_width=True)
            
            # Detailed results table
            st.subheader("📋 Detailed Model Results")
            st.dataframe(model_results, use_container_width=True)
            
            # Best model highlight
            if 'r2' in model_results.columns:
                best_idx = model_results['r2'].idxmax()
                best_model = model_results.iloc[best_idx]
                st.success(f"🏆 Best Model: {best_model.get('model', 'Unknown')} (R² = {best_model.get('r2', 0):.4f})")
            
        else:
            st.info("🔄 No model results available. Train models first.")
    
    # Tab 4: Cross-Market Analysis
    with tab4:
        st.header("🌐 Cross-Market Analysis")
        
        if not data.empty and 'Symbol' in data.columns:
            # Correlation heatmap
            st.subheader("🔥 Cross-Market Correlations")
            corr_chart = create_correlation_heatmap(data)
            st.plotly_chart(corr_chart, use_container_width=True)
            
            # Market summary
            st.subheader("📊 Market Summary")
            
            # Dynamically determine available columns
            base_cols = ['close', 'returns', 'volume']
            available_summary_cols = [col for col in base_cols if col in data.columns]
            
            # Find volatility column
            volatility_cols = [col for col in ['volatility_20', 'volatility_10', 'volatility_5', 'volatility'] if col in data.columns]
            if volatility_cols:
                available_summary_cols.extend(volatility_cols[:1])  # Add first available volatility column
            
            if available_summary_cols:
                # Create dynamic aggregation dict
                agg_dict = {}
                for col in available_summary_cols:
                    if col == 'close':
                        agg_dict[col] = 'last'
                    else:
                        agg_dict[col] = 'mean'
                
                market_summary = data.groupby('Symbol')[available_summary_cols].agg(agg_dict).round(4)
                
                st.dataframe(market_summary, use_container_width=True)
            
        else:
            st.info("Cross-market analysis requires multiple symbols in the dataset.")
    
    # Tab 5: What-If Scenarios
    with tab5:
        st.header("🔮 What-If Scenario Analysis")
        st.markdown("Simulate different market conditions and see how predictions change")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🎛️ Scenario Parameters")
            
            # Market regime selection
            regime_options = ['High_Vol', 'Medium_Vol', 'Low_Vol', 'Uptrend', 'Downtrend', 'Sideways']
            selected_regime = st.selectbox("📊 Market Regime", regime_options, index=1)
            
            # Volatility adjustment
            volatility_multiplier = st.slider(
                "📈 Volatility Multiplier", 
                min_value=0.1, max_value=3.0, value=1.0, step=0.1,
                help="Adjust market volatility (1.0 = normal)"
            )
            
            # Return adjustment
            return_adjustment = st.slider(
                "💰 Return Shift (%)", 
                min_value=-10.0, max_value=10.0, value=0.0, step=0.5,
                help="Shift expected returns up or down"
            )
            
            # Cross-market correlation
            correlation_strength = st.slider(
                "🔗 Cross-Market Correlation", 
                min_value=0.0, max_value=1.0, value=0.5, step=0.1,
                help="Strength of cross-market relationships"
            )
            
            # Market stress scenario
            stress_scenario = st.checkbox("⚠️ Apply Market Stress Scenario")
            
            if st.button("🚀 Run Simulation"):
                st.success("Simulation parameters updated!")
                
                # Simulate scenario adjustments
                scenario_data = data.copy() if not data.empty else pd.DataFrame()
                
                if not scenario_data.empty and selected_symbol in scenario_data['Symbol'].values:
                    symbol_data = scenario_data[scenario_data['Symbol'] == selected_symbol].copy()
                    
                    # Apply adjustments to the data
                    if 'returns' in symbol_data.columns:
                        # Adjust returns
                        symbol_data['adjusted_returns'] = (
                            symbol_data['returns'] + (return_adjustment / 100)
                        )
                        
                        # Adjust volatility
                        if 'volatility_20' in symbol_data.columns:
                            symbol_data['adjusted_volatility'] = (
                                symbol_data['volatility_20'] * volatility_multiplier
                            )
                        
                        # Apply stress scenario
                        if stress_scenario:
                            symbol_data['adjusted_returns'] *= 1.5  # Increase volatility
                            symbol_data['adjusted_volatility'] *= 2.0
                    
                    # Store simulation results
                    st.session_state['simulation_data'] = symbol_data
                    st.session_state['simulation_params'] = {
                        'regime': selected_regime,
                        'volatility_multiplier': volatility_multiplier,
                        'return_adjustment': return_adjustment,
                        'correlation_strength': correlation_strength,
                        'stress_scenario': stress_scenario
                    }
        
        with col2:
            st.subheader("📊 Simulation Results")
            
            if 'simulation_data' in st.session_state:
                sim_data = st.session_state['simulation_data']
                sim_params = st.session_state['simulation_params']
                
                # Display scenario summary
                st.info(f"""
                **Current Scenario:** {sim_params['regime']} Market
                - Volatility: {sim_params['volatility_multiplier']:.1f}x normal
                - Return Shift: {sim_params['return_adjustment']:+.1f}%
                - Stress Mode: {'ON' if sim_params['stress_scenario'] else 'OFF'}
                """)
                
                # Show adjusted metrics
                if not sim_data.empty:
                    latest_sim = sim_data.iloc[-1]
                    
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        original_return = latest_sim.get('returns', 0)
                        adjusted_return = latest_sim.get('adjusted_returns', original_return)
                        st.metric(
                            "📈 Adjusted Return",
                            f"{adjusted_return:.2%}",
                            delta=f"{(adjusted_return - original_return):.2%}"
                        )
                    
                    with col_b:
                        original_vol = latest_sim.get('volatility_20', 0)
                        adjusted_vol = latest_sim.get('adjusted_volatility', original_vol)
                        st.metric(
                            "📊 Adjusted Volatility",
                            f"{adjusted_vol:.4f}",
                            delta=f"{(adjusted_vol - original_vol):.4f}"
                        )
                    
                    with col_c:
                        risk_score = adjusted_vol * abs(adjusted_return) * 100
                        st.metric("⚠️ Risk Score", f"{risk_score:.2f}")
                    
                    # Scenario comparison chart
                    if len(sim_data) > 1:
                        fig = go.Figure()
                        
                        # Original returns
                        if 'returns' in sim_data.columns:
                            fig.add_trace(go.Scatter(
                                x=sim_data.index,
                                y=sim_data['returns'],
                                mode='lines',
                                name='Original Returns',
                                line=dict(color='blue')
                            ))
                        
                        # Adjusted returns
                        if 'adjusted_returns' in sim_data.columns:
                            fig.add_trace(go.Scatter(
                                x=sim_data.index,
                                y=sim_data['adjusted_returns'],
                                mode='lines',
                                name='Scenario Returns',
                                line=dict(color='red', dash='dash')
                            ))
                        
                        fig.update_layout(
                            title="Return Comparison: Original vs Scenario",
                            xaxis_title="Time",
                            yaxis_title="Returns",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                # Scenario insights
                st.subheader("🧠 Scenario Insights")
                
                insights = []
                if sim_params['volatility_multiplier'] > 1.5:
                    insights.append("⚠️ High volatility environment may increase prediction uncertainty")
                if abs(sim_params['return_adjustment']) > 5:
                    insights.append("💡 Large return shifts suggest regime change likelihood")
                if sim_params['stress_scenario']:
                    insights.append("🚨 Stress conditions active - consider defensive positioning")
                if sim_params['correlation_strength'] > 0.8:
                    insights.append("🔗 High correlations may reduce diversification benefits")
                
                if insights:
                    for insight in insights:
                        st.write(insight)
                else:
                    st.write("📊 Current scenario represents normal market conditions")
            
            else:
                st.info("👆 Adjust parameters and click 'Run Simulation' to see results")
                
                # Show example scenario results
                st.markdown("### 💡 Example Scenarios")
                
                example_scenarios = [
                    {"name": "📈 Bull Market", "desc": "Low volatility, positive returns, high correlation"},
                    {"name": "📉 Bear Market", "desc": "High volatility, negative returns, high correlation"},
                    {"name": "🔀 Sideways Market", "desc": "Medium volatility, flat returns, low correlation"},
                    {"name": "💥 Crisis Mode", "desc": "Extreme volatility, negative returns, stress conditions"}
                ]
                
                for scenario in example_scenarios:
                    with st.expander(scenario["name"]):
                        st.write(scenario["desc"])
    
    # Footer
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📊 Data Points", len(data))
    with col2:
        st.metric("🏢 Symbols", len(data['Symbol'].unique()) if 'Symbol' in data.columns else 0)
    with col3:
        st.metric("📋 Features", len(data.columns))
    with col4:
        st.metric("🕐 Last Update", datetime.now().strftime("%H:%M:%S"))
    with col5:
        st.metric("🔮 Scenario Runs", len(st.session_state['simulation_data']) if 'simulation_data' in st.session_state else 0)
    
    st.markdown("**📊 Time Series Forecasting Dashboard** | Powered by Streamlit")

if __name__ == "__main__":
    main()
