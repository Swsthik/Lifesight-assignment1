"""
Marketing Intelligence Dashboard with Prophet Forecasting and Order Attribution

This comprehensive dashboard provides marketing analytics and business intelligence
for multi-channel marketing campaigns including Facebook, Google, and TikTok.

Key Features:
- Multi-channel marketing performance analysis (Facebook, Google, TikTok)
- Prophet-based AI forecasting for revenue and orders prediction
- Order-level attribution analysis with UTM parameter tracking
- Campaign and tactic performance insights with ROI/ROAS metrics
- Geographic performance breakdown by state
- Real-time KPI monitoring with customizable targets
- Growth trend analysis and efficiency tracking
- Interactive data filtering and visualization
- Cohort lifetime value analysis
- Business intelligence insights and recommendations

Technical Implementation:
- Framework: Streamlit for web application interface
- Forecasting: Facebook Prophet for time series predictions
- Visualization: Plotly for interactive charts and graphs
- Data Processing: Pandas for data manipulation and analysis
- UI Components: Streamlit metrics, charts, and filters

Data Sources:
- Marketing Data: Facebook.csv, Google.csv, TikTok.csv
- Business Data: Business.csv (orders, revenue, KPIs)
- Order Data: Orders.csv (optional, for detailed attribution)

Usage:
1. Load CSV files (local or upload mode)
2. Configure filters (date range, channels, states, tactics)
3. Monitor KPIs and ROAS performance
4. Analyze forecasts and trends
5. Review campaign performance and insights
6. Make data-driven marketing decisions

Analytics Capabilities:
- ROAS (Return on Ad Spend) analysis with target thresholds
- CTR (Click-Through Rate) optimization insights
- CPC (Cost Per Click) efficiency monitoring
- Attribution modeling with marketing share calculation
- Forecasting with confidence intervals and trend analysis
- Cohort analysis for customer lifetime value tracking

Author: Marketing Analytics Team
Version: 2.0 (Prophet Integration)
"""
Version: 2.0
Last Updated: September 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import timedelta
import warnings

# Configure Streamlit page settings for wide layout and professional appearance
st.set_page_config(layout="wide", page_title="Marketing Intelligence Dashboard (Prophet + Order Attribution)")

# Try to import Prophet for advanced time series forecasting
# Prophet is Facebook's open-source forecasting tool that handles seasonality well
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception as e:
    # Fall back to simple moving average if Prophet is not available
    PROPHET_AVAILABLE = False

# Suppress pandas and other library warnings for cleaner output
warnings.filterwarnings("ignore")

# ========================================================================
# HELPER FUNCTIONS AND DATA PROCESSING
# ========================================================================

def safe_div(a, b):
    """
    Safely divide two pandas Series/values, replacing division by zero with NaN.
    
    Args:
        a: Numerator (pandas Series or numeric value)
        b: Denominator (pandas Series or numeric value)
    
    Returns:
        Result of a/b with zeros in denominator replaced by NaN
    
    Example:
        safe_div(revenue_series, spend_series) returns ROAS with no division errors
    """
    return a / b.replace(0, np.nan)

@st.cache_data
def load_marketing(csv_paths):
    """
    Load and normalize marketing data from multiple CSV files (Facebook, Google, TikTok).
    
    This function:
    1. Reads CSV files with date parsing
    2. Adds channel identification 
    3. Normalizes column names and data types
    4. Handles common column name variations (impression vs impressions, etc.)
    
    Args:
        csv_paths: List of tuples [(file_path, channel_name), ...]
    
    Returns:
        pandas.DataFrame: Combined marketing data with normalized columns:
            - date: datetime
            - channel: string (Facebook, Google, TikTok)
            - impressions: numeric
            - clicks: numeric  
            - spend: numeric
            - attr_revenue: numeric (attributed revenue)
            - state: string
            - tactic: string
            - campaign: string
    """
    frames = []
    for p, channel_name in csv_paths:
        # Load CSV with automatic date parsing for 'date' column
        df = pd.read_csv(p, parse_dates=['date'])
        df['channel'] = channel_name
        frames.append(df)
    
    # Combine all marketing data into single DataFrame
    m = pd.concat(frames, ignore_index=True)
    
    # Normalize column names by stripping whitespace
    m.columns = [c.strip() for c in m.columns]
    
    # Convert key numeric columns, handling any non-numeric values gracefully
    for col in ['impression','impressions','clicks','spend','attributed revenue','attr_revenue']:
        if col in m.columns:
            m[col] = pd.to_numeric(m[col], errors='coerce').fillna(0)
    
    # Standardize column names across different data sources
    if 'attributed revenue' in m.columns:
        m = m.rename(columns={'attributed revenue':'attr_revenue'})
    if 'impression' in m.columns:
        m = m.rename(columns={'impression':'impressions'})
    
    return m

@st.cache_data
def load_business(path):
    """
    Load and normalize business/revenue data from CSV file.
    
    Processes daily business metrics including orders, revenue, profit, and costs.
    
    Args:
        path: File path to business data CSV
        
    Returns:
        pandas.DataFrame: Business data with columns:
            - date: datetime
            - # of orders: numeric
            - # of new orders: numeric  
            - new customers: numeric
            - total revenue: numeric
            - gross profit: numeric
            - COGS: numeric (Cost of Goods Sold)
    """
    # Load business data with date parsing
    b = pd.read_csv(path, parse_dates=['date'])
    
    # Clean column names
    b.columns = [c.strip() for c in b.columns]
    
    # Convert all non-date columns to numeric, handling errors gracefully
    for col in b.columns:
        if col != 'date':
            try:
                b[col] = pd.to_numeric(b[col], errors='coerce').fillna(0)
            except:
                pass  # Skip columns that can't be converted
    
    return b

def prepare_marketing_aggregates(mkt_df):
    """
    Process raw marketing data and calculate key performance metrics.
    
    This function:
    1. Ensures required columns exist (fills with 0 if missing)
    2. Parses campaign structure from campaign names
    3. Aggregates data by date, channel, state, tactic, and campaign
    4. Calculates essential marketing metrics (CTR, CPC, ROAS)
    
    Args:
        mkt_df: Raw marketing DataFrame
        
    Returns:
        pandas.DataFrame: Aggregated marketing data with calculated metrics:
            - All original grouping columns (date, channel, state, tactic, campaign)
            - Summed metrics (impressions, clicks, spend, attr_revenue)
            - Calculated metrics (ctr, cpc, roas)
    """
    # Ensure all required columns exist, filling missing ones with zeros
    for req in ['impressions','clicks','spend','attr_revenue']:
        if req not in mkt_df.columns:
            mkt_df[req] = 0
    
    # Parse campaign structure if campaign column exists
    # Many campaigns follow pattern: "Platform - Tactic - Identifier"
    if 'campaign' in mkt_df.columns:
        parsed = mkt_df['campaign'].astype(str).str.split('-', n=4, expand=True)
        for i in range(parsed.shape[1]):
            mkt_df[f'campaign_part_{i}'] = parsed[i].str.strip()
    
    # Aggregate data by key dimensions to avoid double-counting
    agg = mkt_df.groupby(['date','channel','state','tactic','campaign'], as_index=False).agg({
        'impressions':'sum','clicks':'sum','spend':'sum','attr_revenue':'sum'
    })
    
    # Calculate key marketing performance metrics
    agg['ctr'] = safe_div(agg['clicks'], agg['impressions'])  # Click-through rate
    agg['cpc'] = safe_div(agg['spend'], agg['clicks'])        # Cost per click
    agg['roas'] = safe_div(agg['attr_revenue'], agg['spend']) # Return on ad spend
    
    return agg

def merge_with_business(mkt_daily, business):
    """
    Merge marketing data with business data to enable comprehensive analysis.
    
    Creates a unified dataset that combines marketing spend/performance with
    business outcomes like total revenue, orders, and profit.
    
    Args:
        mkt_daily: Daily aggregated marketing data
        business: Daily business metrics data
        
    Returns:
        pandas.DataFrame: Combined dataset with both marketing and business metrics:
            - All business metrics (revenue, orders, profit, etc.)
            - Marketing metrics (spend, attributed revenue, etc.) 
            - Calculated ratios (marketing share of revenue, revenue per order)
    """
    # Aggregate marketing data to daily level for joining with business data
    daily_mkt = mkt_daily.groupby('date', as_index=False).agg({
        'impressions':'sum','clicks':'sum','spend':'sum','attr_revenue':'sum'
    }).rename(columns={'attr_revenue':'mkt_attr_revenue', 'spend':'mkt_spend'})
    
    # Left join business data with marketing data (business data is primary)
    df = pd.merge(business, daily_mkt, on='date', how='left').fillna(0)
    
    # Calculate additional business intelligence metrics
    df['marketing_share_of_revenue'] = safe_div(df['mkt_attr_revenue'], df['total revenue'])
    df['revenue_per_order'] = safe_div(df['total revenue'], df['# of orders'])
    
    return df

# ========================================================================
# FORECASTING FUNCTIONS
# ========================================================================

def moving_average_forecast(series, periods=7, forecast_horizon=14):
    """
    Generate simple moving average forecast as fallback when Prophet is unavailable.
    
    Creates a basic forecast by averaging recent values and projecting forward.
    While simple, this method is reliable and doesn't require external dependencies.
    
    Args:
        series: pandas.Series with datetime index containing values to forecast
        periods: Number of recent periods to average (default: 7 days)
        forecast_horizon: Number of future periods to predict (default: 14 days)
        
    Returns:
        pandas.DataFrame: Forecast with columns:
            - ds: Future dates
            - yhat: Predicted values (constant based on recent average)
    """
    # Calculate average of recent non-null values
    last_avg = series.dropna().tail(periods).mean()
    
    # Get the last date in the series to start forecasting from
    idx_last = series.index.max()
    
    # Generate future dates for the forecast horizon
    future_index = [idx_last + timedelta(days=i) for i in range(1, forecast_horizon+1)]
    
    # Create constant forecast values (simple assumption)
    forecast_values = [last_avg]*forecast_horizon
    
    # Return in format consistent with Prophet output
    forecast_df = pd.DataFrame({'ds':future_index, 'yhat':forecast_values})
    return forecast_df

def prophet_forecast(series, forecast_horizon=14, changepoint_prior_scale=0.5):
    """
    Generate advanced time series forecast using Facebook's Prophet algorithm.
    
    Prophet is particularly good at handling:
    - Seasonal patterns (daily, weekly, yearly)
    - Holiday effects
    - Trend changes
    - Missing data
    
    Args:
        series: pandas.Series indexed by date with values to forecast
        forecast_horizon: Number of days to forecast into the future
        changepoint_prior_scale: Controls trend flexibility (0.5 = moderate)
        
    Returns:
        pandas.DataFrame or None: Forecast with columns:
            - ds: Future dates  
            - yhat: Predicted values
            - yhat_lower: Lower confidence bound
            - yhat_upper: Upper confidence bound
        Returns None if Prophet unavailable or insufficient data
    """
    # Check if Prophet is available
    if not PROPHET_AVAILABLE:
        return None
    
    # Prepare data in Prophet's required format (ds, y columns)
    if series.name != 'y':
        df = series.reset_index().rename(columns={'date':'ds', series.name:'y'})
    else:
        df = series.reset_index().rename(columns={'date':'ds'})
    df.columns = ['ds', 'y']
    
    # Validate data quality for Prophet
    if df['y'].isna().all() or len(df) < 14:
        return None
    
    # Initialize and configure Prophet model
    m = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,  # Trend flexibility
        daily_seasonality=True  # Enable daily patterns
    )
    
    # Fit the model to historical data
    m.fit(df)
    
    # Create future dataframe including forecast horizon
    future = m.make_future_dataframe(periods=forecast_horizon)
    
    # Generate predictions with confidence intervals
    forecast = m.predict(future)[['ds','yhat','yhat_lower','yhat_upper']]
    
    return forecast

def order_attribution_from_order_file(orders_df):
    """
    Process order-level data for detailed attribution analysis.
    
    This function normalizes order data to enable campaign-level attribution
    by extracting UTM parameters and revenue information.
    
    Args:
        orders_df: Raw order data DataFrame
        
    Returns:
        pandas.DataFrame: Normalized order data with columns:
            - order_revenue: Numeric revenue per order
            - order_date: Datetime of order
            - utm_source: Campaign source (normalized)
            - utm_medium: Campaign medium (normalized)  
            - utm_campaign: Campaign identifier (normalized)
            - utm_term: Campaign term (normalized)
            - utm_content: Campaign content (normalized)
    
    Note:
        This enables more precise attribution than aggregate daily data
        by tracking individual order sources.
    """
    df = orders_df.copy()
    
    # Try to find revenue column with various possible names
    revenue_cols = [c for c in df.columns if c.lower() in 
                   ('order_revenue','revenue','total','amount','order_total','total_revenue')]
    if revenue_cols:
        df['order_revenue'] = pd.to_numeric(df[revenue_cols[0]], errors='coerce').fillna(0)
    else:
        # Fallback if no revenue column found
        df['order_revenue'] = 0.0

    # Find order_date column with various possible names
    date_cols = [c for c in df.columns if 
                ('date' in c.lower() and 'order' in c.lower()) or 
                c.lower()=='order_date' or c.lower()=='date']
    if date_cols:
        df['order_date'] = pd.to_datetime(df[date_cols[0]])
    else:
        st.warning("No order_date found in orders file; attribution will be limited.")
        df['order_date'] = pd.NaT

    # Normalize UTM tracking parameters for attribution
    # These parameters track the marketing source of each order
    for u in ['utm_source','utm_medium','utm_campaign','utm_term','utm_content']:
        if u not in df.columns:
            # Try to find variations of UTM parameter names
            candidates = [c for c in df.columns if c.lower().endswith(u)]
            if candidates:
                df[u] = df[candidates[0]]
            else:
                df[u] = None
    
    return df

# ========================================================================
# DATA LOADING AND CONFIGURATION
# ========================================================================
# This section handles loading all data sources and configuring the dashboard

# Configure sidebar for data input controls
st.sidebar.header("Data Input & Configuration")

# Toggle between local files and file uploads
# This provides flexibility for different deployment environments
use_upload = st.sidebar.checkbox("Upload CSVs instead of using local data (marketing, business, orders)", value=False)

if use_upload:
    # FILE UPLOAD MODE: Allows users to upload their own data files
    # This is useful for custom datasets or when deploying in cloud environments
    st.info("📁 Upload marketing files (Facebook.csv, Google.csv, TikTok.csv), business file (Business.csv), and optionally an order-level file (Orders.csv).")
    
    # Individual file upload widgets for each data source
    fb_file = st.sidebar.file_uploader("Facebook.csv", type=["csv"])
    g_file = st.sidebar.file_uploader("Google.csv", type=["csv"])
    tt_file = st.sidebar.file_uploader("TikTok.csv", type=["csv"])
    b_file = st.sidebar.file_uploader("Business.csv", type=["csv"])
    orders_file = st.sidebar.file_uploader("Orders (optional).csv", type=["csv"])
    
    # Validate that required files are uploaded before proceeding
    if not (fb_file and g_file and tt_file and b_file):
        st.warning("⚠️ Upload the three marketing files and the business file to proceed.")
        st.stop()
    
    # Load uploaded files into DataFrames with proper date parsing
    fb = pd.read_csv(fb_file, parse_dates=['date'])
    g = pd.read_csv(g_file, parse_dates=['date'])
    b = pd.read_csv(b_file, parse_dates=['date'])
    
    # Add channel identification to each platform's data
    # This enables cross-platform comparison and aggregation
    fb['channel']='Facebook'; g['channel']='Google'; tt['channel']='TikTok'
    
    # Combine all marketing data into unified DataFrame
    marketing_raw = pd.concat([fb,g,tt], ignore_index=True)
    business = b
    
    # Handle optional orders file for attribution analysis
    orders_df = None
    if orders_file:
        orders_df = pd.read_csv(orders_file)
        
else:
    # LOCAL FILE MODE: Use files from current directory
    # This is the default mode for development and when files are pre-deployed
    data_folder = Path(".")
    st.sidebar.write("📂 Using local CSVs in current folder. You can toggle to upload.")
    
    # Define paths to local data files
    fb_path = data_folder / "Facebook.csv"
    g_path = data_folder / "Google.csv"
    tt_path = data_folder / "TikTok.csv"
    b_path = data_folder / "Business.csv"
    orders_path = data_folder / "Orders.csv"
    
    # Validate that required local files exist
    if not (fb_path.exists() and g_path.exists() and tt_path.exists() and b_path.exists()):
        st.error("❌ Local CSVs not found in current directory. Toggle 'Upload CSVs' or place files in current directory.")
        st.stop()
    
    # Load data using helper functions
    marketing_raw = load_marketing([(fb_path,'Facebook'), (g_path,'Google'), (tt_path,'TikTok')])
    business = load_business(b_path)
    
    # Load optional orders file if it exists
    orders_df = None
    if orders_path.exists():
        orders_df = pd.read_csv(orders_path)

# ========================================================================
# DATA PREPROCESSING AND NORMALIZATION  
# ========================================================================
# This section standardizes column names and data formats across all sources

# Normalize marketing column names to ensure consistency
# Different platforms may use slightly different naming conventions
marketing_raw.columns = [c.strip() for c in marketing_raw.columns]

# Standardize common column name variations
if 'attributed revenue' in marketing_raw.columns:
    marketing_raw = marketing_raw.rename(columns={'attributed revenue':'attr_revenue'})
if 'impression' in marketing_raw.columns:
    marketing_raw = marketing_raw.rename(columns={'impression':'impressions'})

# ========================================================================
# DATA AGGREGATION AND MERGING
# ========================================================================
# This section processes and combines marketing and business data

# Aggregate marketing data to daily level with calculated metrics
marketing_agg = prepare_marketing_aggregates(marketing_raw)

# Clean business data column names  
business_clean = business.copy()
business_clean.columns = [c.strip() for c in business_clean.columns]

# Merge marketing and business data for comprehensive analysis
merged = merge_with_business(marketing_agg, business_clean)

# ========================================================================
# USER INTERFACE AND FILTERING
# ========================================================================
# This section creates interactive filters for data exploration

st.sidebar.header("📊 Data Filters")

# Date range filter - allows users to focus on specific time periods
min_date = merged['date'].min()
max_date = merged['date'].max()
date_range = st.sidebar.date_input("📅 Date range", [min_date, max_date], min_value=min_date, max_value=max_date)

# Channel filter - enables single or multi-platform analysis
channels = sorted(marketing_agg['channel'].unique().tolist())
sel_channels = st.sidebar.multiselect("📱 Channel", channels, default=channels)

# Geographic filter - allows state-level analysis
states = sorted(marketing_agg['state'].dropna().unique().tolist())
sel_states = st.sidebar.multiselect("🗺️ State", states, default=None)

# Campaign tactic filter - enables strategy-level analysis  
tactics = sorted(marketing_agg['tactic'].dropna().unique().tolist())
sel_tactics = st.sidebar.multiselect("🎯 Tactic", tactics, default=None)

# ========================================================================
# DATA FILTERING APPLICATION
# ========================================================================
# Apply user-selected filters to create filtered dataset

# Apply date range filter
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
mkt_f = marketing_agg[(marketing_agg['date'] >= start_date) & (marketing_agg['date'] <= end_date)]

# Apply channel filter if channels are selected
if sel_channels:
    mkt_f = mkt_f[mkt_f['channel'].isin(sel_channels)]

# Apply state filter if states are selected
if sel_states:
    mkt_f = mkt_f[mkt_f['state'].isin(sel_states)]

# Apply tactic filter if tactics are selected    
if sel_tactics:
    mkt_f = mkt_f[mkt_f['tactic'].isin(sel_tactics)]

# Apply same date filter to merged dataset for consistency
merged_f = merged[(merged['date'] >= start_date) & (merged['date'] <= end_date)]

# ========================================================================
# DASHBOARD HEADER AND LAYOUT
# ========================================================================
# This section creates the main dashboard interface and layout

st.title("📈 Marketing Intelligence Dashboard")
st.caption("Advanced analytics with Prophet forecasting and order attribution")

# Create a placeholder container for AI insights that will update dynamically
# This allows insights to render first and refresh automatically as data changes
insights_container = st.empty()

# ========================================================================
# KEY PERFORMANCE INDICATORS (KPIs)
# ========================================================================
# This section calculates and displays top-level marketing metrics

# Create column layout for KPI display
k1, k2, k3, k4 = st.columns(4)

# Calculate primary marketing KPIs from filtered data
total_spend = mkt_f['spend'].sum()
total_attr_rev = mkt_f['attr_revenue'].sum()
agg_roas = total_attr_rev / total_spend if total_spend > 0 else np.nan

# Calculate business KPIs from merged dataset
total_orders = merged_f['# of orders'].sum()
total_revenue = merged_f['total revenue'].sum()
gross_profit = merged_f['gross profit'].sum()

# Display KPI metrics in organized columns
k1.metric("💰 Marketing Spend", f"${total_spend:,.0f}")
k2.metric("📈 Attributed Revenue", f"${total_attr_rev:,.0f}",
          delta=f"ROAS {agg_roas:.2f}" if not np.isnan(agg_roas) else "ROAS N/A")
k3.metric("🛒 Total Orders", f"{int(total_orders):,}")
k4.metric("💵 Total Revenue", f"${total_revenue:,.0f}")

# ========================================================================
# BUSINESS INSIGHTS SECTION
# ========================================================================
# This section provides AI-powered insights and ROAS analysis

# Populate the insights container with dynamic business intelligence
with insights_container:
    st.markdown("## 💡 Key Business Insights")

    # Helper function for consistent money formatting
    def fmt_money(x):
        """Format numbers as currency with commas"""
        return f"${x:,.0f}"

    # Interactive ROAS targeting for performance analysis
    target_roas = st.sidebar.slider("🎯 Target ROAS", 0.0, 5.0, 2.0, 0.1)
    
    # Calculate daily ROAS performance for trend analysis
    roas_df = mkt_f.groupby('date', as_index=False)[['attr_revenue','spend']].sum()
    roas_df['roas'] = roas_df['attr_revenue'] / roas_df['spend'].replace(0, np.nan)
    avg_roas = roas_df['roas'].mean(skipna=True)

    # Display ROAS performance metrics in organized layout
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("📊 Average ROAS", f"{avg_roas:.2f}")
    
    # Calculate days below target for performance monitoring
    low_days = (roas_df['roas'] < target_roas).sum()
    kpi2.metric("⚠️ Days Below Target", f"{low_days}")
    kpi3.metric("🎯 Target ROAS", f"{target_roas:.2f}",
                delta="Below Target!" if low_days else "On Track!")

    # Calculate and display growth trends for strategic insights
    if len(merged_f) > 7:
        first_week = merged_f.head(7)
        last_week = merged_f.tail(7)
        
        # Calculate week-over-week growth rates
        rev_growth = ((last_week['total revenue'].mean() - first_week['total revenue'].mean())
                      / first_week['total revenue'].mean()) * 100
        spend_growth = ((last_week['mkt_spend'].mean() - first_week['mkt_spend'].mean())
                        / first_week['mkt_spend'].mean()) * 100
        
        # Display growth trend insights
        st.info(
            f"📈 **Growth Trend Analysis:** Revenue changed by **{rev_growth:+.1f}%** "
            f"and marketing spend by **{spend_growth:+.1f}%** vs. the first week."
        )

# ========================================================================
# VISUALIZATION SECTIONS  
# ========================================================================
# This section creates interactive charts and data visualizations

# ========================================================================
# TIME SERIES TRENDS VISUALIZATION
# ========================================================================

st.subheader("📊 Trends — Spend vs Attributed Revenue vs Business Revenue")
st.caption("Interactive visualization comparing marketing investment with attributed and total revenue over time")

# Prepare time series data for visualization
time_df = merged_f[['date','mkt_spend','mkt_attr_revenue','total revenue']].sort_values('date')

# Create dual-axis chart to compare spend (bars) with revenue (lines)
fig = go.Figure()

# Add marketing spend as bars (left axis)
fig.add_trace(go.Bar(
    x=time_df['date'], 
    y=time_df['mkt_spend'], 
    name='💰 Marketing Spend', 
    yaxis='y1', 
    opacity=0.6,
    hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Amount: $%{y:,.0f}<extra></extra>'
))

# Add marketing attributed revenue as line (right axis)
fig.add_trace(go.Scatter(
    mode='lines', 
    x=time_df['date'], 
    y=time_df['mkt_attr_revenue'], 
    name='📈 Marketing Attributed Revenue', 
    yaxis='y2',
    hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Revenue: $%{y:,.0f}<extra></extra>'
))

# Add total business revenue as dashed line (right axis)
fig.add_trace(go.Scatter(
    mode='lines', 
    x=time_df['date'], 
    y=time_df['total revenue'], 
    name='💵 Business Total Revenue', 
    yaxis='y2', 
    line=dict(dash='dash'),
    hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Revenue: $%{y:,.0f}<extra></extra>'
))

# Configure dual-axis layout for better readability
fig.update_layout(
    xaxis_title="📅 Date",
    yaxis=dict(title="💰 Spend ($)", side='left', showgrid=False),
    yaxis2=dict(title="💵 Revenue ($)", overlaying='y', side='right'),
    legend=dict(x=0, y=1.12, orientation='h'),
    margin=dict(t=40),
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

# ========================================================================
# PROPHET FORECASTING SECTION
# ========================================================================

st.subheader("🔮 Forecast: Prophet AI-Powered Predictions")
st.caption("Advanced time series forecasting for orders and revenue using Facebook's Prophet algorithm")

if PROPHET_AVAILABLE:
    st.success("✅ Prophet available — using advanced AI forecasting.")
else:
    st.warning("⚠️ Prophet is not installed. Forecasting will use simple moving-average. To enable Prophet: `pip install prophet`")

# ========================================================================
# FORECAST DATA PREPARATION
# ========================================================================

# Prepare time series data for forecasting analysis
# Resample data to daily frequency and fill missing dates with zeros
orders_series = merged_f.set_index('date')['# of orders'].resample('D').sum().reindex(
    pd.date_range(start_date, end_date, freq='D')).fillna(0)
revenue_series = merged_f.set_index('date')['total revenue'].resample('D').sum().reindex(
    pd.date_range(start_date, end_date, freq='D')).fillna(0)

# Configure forecast horizon (number of future days to predict)
forecast_horizon = 14

# ========================================================================
# ORDERS FORECASTING
# ========================================================================

st.markdown("### 🛒 Orders Forecast")

# Generate orders forecast using Prophet if available and sufficient data
if PROPHET_AVAILABLE and len(orders_series.dropna()) >= 30:
    with st.spinner("🔮 Fitting Prophet model for orders..."):
        forecast_orders = prophet_forecast(orders_series, forecast_horizon=forecast_horizon)
    
    if forecast_orders is not None:
        # Create interactive chart with historical data and Prophet forecast
        figo = go.Figure()
        
        # Add historical orders data
        figo.add_trace(go.Scatter(
            x=orders_series.index, 
            y=orders_series.values, 
            name='📊 Historical Orders',
            hovertemplate='<b>Historical Orders</b><br>Date: %{x}<br>Orders: %{y}<extra></extra>'
        ))
        
        # Add Prophet forecast line
        figo.add_trace(go.Scatter(
            x=forecast_orders['ds'], 
            y=forecast_orders['yhat'], 
            name='🔮 Prophet Forecast', 
            line=dict(dash='dash', color='red'),
            hovertemplate='<b>Prophet Forecast</b><br>Date: %{x}<br>Predicted Orders: %{y:.0f}<extra></extra>'
        ))
        
        # Add confidence intervals if available
        if 'yhat_upper' in forecast_orders.columns and 'yhat_lower' in forecast_orders.columns:
            figo.add_trace(go.Scatter(
                x=forecast_orders['ds'], 
                y=forecast_orders['yhat_upper'], 
                name='Upper Bound', 
                line=dict(width=0), 
                showlegend=False
            ))
            figo.add_trace(go.Scatter(
                x=forecast_orders['ds'], 
                y=forecast_orders['yhat_lower'], 
                name='Lower Bound', 
                line=dict(width=0), 
                fill='tonexty', 
                fillcolor='rgba(255,0,0,0.1)', 
                showlegend=False
            ))
        
        figo.update_layout(
            title="🛒 Orders — Prophet 14-day Forecast with Confidence Intervals",
            xaxis_title="📅 Date", 
            yaxis_title="📊 Orders",
            hovermode='x unified'
        )
        st.plotly_chart(figo, use_container_width=True)
    else:
        st.info("ℹ️ Not enough history to run a robust Prophet model for orders. Falling back to moving-average.")
        fo = moving_average_forecast(orders_series, periods=7, forecast_horizon=forecast_horizon)
        figo = go.Figure()
        figo.add_trace(go.Scatter(mode='lines', x=orders_series.index, y=orders_series.values, name='📊 Historical Orders'))
        figo.add_trace(go.Scatter(mode='lines', x=fo['ds'], y=fo['yhat'], name='📈 MA Forecast', line=dict(dash='dash')))
        figo.update_layout(title="🛒 Orders — Moving Average Forecast", xaxis_title="📅 Date", yaxis_title="📊 Orders")
        st.plotly_chart(figo, use_container_width=True)
else:
    # Fallback to moving average forecast when Prophet is unavailable
    fo = moving_average_forecast(orders_series, periods=7, forecast_horizon=forecast_horizon)
    figo = go.Figure()
    figo.add_trace(go.Scatter(mode='lines', x=orders_series.index, y=orders_series.values, name='📊 Historical Orders'))
    figo.add_trace(go.Scatter(mode='lines', x=fo['ds'], y=fo['yhat'], name='📈 MA Forecast', line=dict(dash='dash')))
    figo.update_layout(title="🛒 Orders — Moving Average Forecast", xaxis_title="📅 Date", yaxis_title="📊 Orders")
    st.plotly_chart(figo, use_container_width=True)

# ========================================================================
# REVENUE FORECASTING  
# ========================================================================

# Generate revenue forecast using Prophet if available and sufficient data
if PROPHET_AVAILABLE and len(revenue_series.dropna()) >= 30:
    with st.spinner("🔮 Fitting Prophet model for revenue..."):
        forecast_revenue = prophet_forecast(revenue_series, forecast_horizon=forecast_horizon)
    
    if forecast_revenue is not None:
        # Create interactive chart with historical data and Prophet forecast
        figr = go.Figure()
        
        # Add historical revenue data
        figr.add_trace(go.Scatter(
            mode='lines', 
            x=revenue_series.index, 
            y=revenue_series.values, 
            name='💵 Historical Revenue',
            hovertemplate='<b>Historical Revenue</b><br>Date: %{x}<br>Revenue: $%{y:,.0f}<extra></extra>'
        ))
        
        # Add Prophet forecast line
        figr.add_trace(go.Scatter(
            mode='lines', 
            x=forecast_revenue['ds'], 
            y=forecast_revenue['yhat'], 
            name='🔮 Prophet Forecast', 
            line=dict(dash='dash', color='green'),
            hovertemplate='<b>Prophet Forecast</b><br>Date: %{x}<br>Predicted Revenue: $%{y:,.0f}<extra></extra>'
        ))
        
        # Add confidence intervals if available
        if 'yhat_upper' in forecast_revenue.columns and 'yhat_lower' in forecast_revenue.columns:
            figr.add_trace(go.Scatter(
                x=forecast_revenue['ds'], 
                y=forecast_revenue['yhat_upper'], 
                name='Upper Bound', 
                line=dict(width=0), 
                showlegend=False
            ))
            figr.add_trace(go.Scatter(
                x=forecast_revenue['ds'], 
                y=forecast_revenue['yhat_lower'], 
                name='Lower Bound', 
                line=dict(width=0), 
                fill='tonexty', 
                fillcolor='rgba(0,255,0,0.1)', 
                showlegend=False
            ))
        
        figr.update_layout(
            title="💵 Revenue — Prophet 14-day Forecast with Confidence Intervals",
            xaxis_title="📅 Date", 
            yaxis_title="💵 Revenue ($)",
            hovermode='x unified'
        )
        st.plotly_chart(figr, use_container_width=True)
    else:
        st.info("ℹ️ Not enough history to run a robust Prophet model for revenue. Falling back to moving-average.")
        fr = moving_average_forecast(revenue_series, periods=7, forecast_horizon=forecast_horizon)
        figr = go.Figure()
        figr.add_trace(go.Scatter(mode='lines', x=revenue_series.index, y=revenue_series.values, name='💵 Historical Revenue'))
        figr.add_trace(go.Scatter(mode='lines', x=fr['ds'], y=fr['yhat'], name='📈 MA Forecast', line=dict(dash='dash')))
        figr.update_layout(title="💵 Revenue — Moving Average Forecast", xaxis_title="📅 Date", yaxis_title="💵 Revenue ($)")
        st.plotly_chart(figr, use_container_width=True)
else:
    # Fallback to moving average forecast when Prophet is unavailable
    fr = moving_average_forecast(revenue_series, periods=7, forecast_horizon=forecast_horizon)
    figr = go.Figure()
    figr.add_trace(go.Scatter(mode='lines', x=revenue_series.index, y=revenue_series.values, name='💵 Historical Revenue'))
    figr.add_trace(go.Scatter(mode='lines', x=fr['ds'], y=fr['yhat'], name='📈 MA Forecast', line=dict(dash='dash')))
    figr.update_layout(title="💵 Revenue — Moving Average Forecast", xaxis_title="📅 Date", yaxis_title="💵 Revenue ($)")
    st.plotly_chart(figr, use_container_width=True)

# ========================================================================
# CAMPAIGN PERFORMANCE ANALYSIS
# ========================================================================

st.subheader("📊 Channel & Campaign Performance Analysis")
st.caption("Detailed breakdown of marketing performance by channel and individual campaigns")

# ========================================================================
# CHANNEL-LEVEL PERFORMANCE
# ========================================================================

# Aggregate performance metrics by marketing channel
ch_agg = mkt_f.groupby('channel', as_index=False).agg({
    'impressions':'sum','clicks':'sum','spend':'sum','attr_revenue':'sum'
})

# Calculate derived metrics for channel analysis
ch_agg['ctr'] = safe_div(ch_agg['clicks'], ch_agg['impressions'])  # Click-through rate
ch_agg['cpc'] = safe_div(ch_agg['spend'], ch_agg['clicks'])        # Cost per click
ch_agg['roas'] = safe_div(ch_agg['attr_revenue'], ch_agg['spend']) # Return on ad spend

# Create comparative visualization for spend vs revenue by channel
fig_ch = px.bar(
    ch_agg.melt(id_vars='channel', value_vars=['spend','attr_revenue'], var_name='metric', value_name='value'),
    x='channel', y='value', color='metric', barmode='group', 
    title='💰 Spend vs 📈 Attributed Revenue by Channel',
    labels={'value': 'Amount ($)', 'channel': 'Marketing Channel', 'metric': 'Metric Type'}
)
fig_ch.update_layout(hovermode='x unified')
st.plotly_chart(fig_ch, use_container_width=True)

# ========================================================================
# CAMPAIGN-LEVEL PERFORMANCE
# ========================================================================

st.markdown("#### 🎯 Campaign-Level Performance Table")

# Aggregate performance metrics by channel and campaign
camp_agg = mkt_f.groupby(['channel','campaign'], as_index=False).agg({
    'impressions':'sum','clicks':'sum','spend':'sum','attr_revenue':'sum'
})

# Calculate derived metrics for campaign analysis
camp_agg['ctr'] = safe_div(camp_agg['clicks'], camp_agg['impressions'])  # Click-through rate
camp_agg['cpc'] = safe_div(camp_agg['spend'], camp_agg['clicks'])        # Cost per click
camp_agg['roas'] = safe_div(camp_agg['attr_revenue'], camp_agg['spend']) # Return on ad spend

# Display top campaigns sorted by spend with formatted metrics
st.dataframe(
    camp_agg.sort_values('spend', ascending=False).head(200).style.format({
        'spend':"${:,.2f}", 
        'attr_revenue':"${:,.2f}", 
        'ctr':'{:.2%}', 
        'cpc':"${:,.2f}", 
        'roas':'{:.2f}',
        'impressions':'{:,.0f}',
        'clicks':'{:,.0f}'
    }),
    use_container_width=True
)

# ========================================================================
# ORDER-LEVEL ATTRIBUTION ANALYSIS
# ========================================================================

st.subheader("🛒 Order-Level Attribution & Cohort Analysis")
st.caption("Advanced attribution tracking and customer lifetime value analysis (when order data is available)")

orders_df = None
if 'orders_df' in locals() and locals()['orders_df'] is not None:
    orders_df = locals()['orders_df']
# if user uploads in the session via file_uploader earlier, they are in variable orders_df

uploaded = st.file_uploader("Upload Order-level CSV (optional). Columns: order_id, order_date, order_revenue (or total), utm_* fields", type=['csv'])
if uploaded is not None:
    orders_df = pd.read_csv(uploaded)

if orders_df is not None:
    orders_norm = order_attribution_from_order_file(orders_df)
    # attribute orders by utm_campaign if present; fallback to utm_source; fallback to 'Unknown'
    orders_norm['utm_campaign'] = orders_norm['utm_campaign'].fillna('').astype(str)
    orders_norm['utm_source'] = orders_norm['utm_source'].fillna('').astype(str)
    orders_norm['attrib_campaign'] = np.where(orders_norm['utm_campaign']!='', orders_norm['utm_campaign'],
                                             np.where(orders_norm['utm_source']!='', orders_norm['utm_source'], 'Unknown'))
    # aggregate campaign-level actuals
    orders_norm['order_date'] = pd.to_datetime(orders_norm['order_date'])
    orders_norm = orders_norm[(orders_norm['order_date'] >= pd.to_datetime(start_date)) & (orders_norm['order_date'] <= pd.to_datetime(end_date))]
    campaign_orders = orders_norm.groupby('attrib_campaign').agg({
        'order_revenue':'sum',
        'order_date':'count'
    }).rename(columns={'order_date':'orders', 'order_revenue':'revenue'}).reset_index().sort_values('revenue', ascending=False)
    st.write("Campaign-level actual orders & revenue from order-level data")
    st.dataframe(campaign_orders.style.format({'revenue':"${:,.2f}", 'orders':'{:.0f}'}))

    # Cohort LTV: assign cohort = order_date (or first order date per customer if customer id exists)
    if 'customer_id' in orders_norm.columns:
        # use first-order cohort per customer
        first_order = orders_norm.groupby('customer_id')['order_date'].min().reset_index().rename(columns={'order_date':'cohort_date'})
        orders_norm = orders_norm.merge(first_order, on='customer_id', how='left')
        cohort_col = 'cohort_date'
    else:
        # use order_date as cohort (cohort by week)
        orders_norm['cohort_week'] = orders_norm['order_date'].dt.to_period('W').apply(lambda r: r.start_time)
        cohort_col = 'cohort_week'
    st.write("Cohort LTV sample (sum revenue by cohort over time)")
    if cohort_col in orders_norm.columns:
        cohort = orders_norm.groupby([cohort_col, 'attrib_campaign']).agg({'order_revenue':'sum','order_date':'count'}).reset_index()
        st.dataframe(cohort.head(200).style.format({'order_revenue':"${:,.2f}", 'order_date':'{:.0f}'}))
    else:
        st.info("Not enough cohort info (no customer_id and cohorting disabled).")
else:
    st.info("No order-level file found. Upload `Orders.csv` or provide it via the Upload toggle to enable exact attribution and cohort LTVs. The dashboard will continue to use advertised-attributed revenue for aggregate metrics.")

# ======================================================================
# ===  COMPACT INSIGHTS SECTION  (text-focused, minimal graphs) ========
# ======================================================================

st.markdown("## 💡 Key Business Insights")

# Helper for money format
def fmt_money(x): 
    return f"${x:,.0f}"

# --- ROAS Target & Alerts (no chart, just a KPI and warning)
target_roas = st.sidebar.slider("Target ROAS", 0.0, 5.0, 2.0, 0.1)
roas_df = mkt_f.groupby('date', as_index=False)[['attr_revenue','spend']].sum()
roas_df['roas'] = roas_df['attr_revenue'] / roas_df['spend'].replace(0, np.nan)
avg_roas = roas_df['roas'].mean(skipna=True)

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Average ROAS", f"{avg_roas:.2f}")
low_days = (roas_df['roas'] < target_roas).sum()
kpi2.metric("Days Below Target", f"{low_days}")
if low_days > 0:
    kpi3.metric("Target ROAS", f"{target_roas:.2f}", delta="Below Target!" if low_days else "")

# --- Growth Trend vs First Week
if len(merged_f) > 7:
    first_week = merged_f.head(7)
    last_week = merged_f.tail(7)
    rev_growth = ((last_week['total revenue'].mean() - first_week['total revenue'].mean())
                  / first_week['total revenue'].mean()) * 100
    spend_growth = ((last_week['mkt_spend'].mean() - first_week['mkt_spend'].mean())
                    / first_week['mkt_spend'].mean()) * 100
    st.info(
        f"📈 **Growth Trend:** Revenue changed by **{rev_growth:+.1f}%** "
        f"and marketing spend by **{spend_growth:+.1f}%** vs. the first week."
    )

# --- Best & Worst Days
best_day = merged_f.loc[merged_f['total revenue'].idxmax()]
worst_day = merged_f.loc[merged_f['total revenue'].idxmin()]
st.write(
    f"📅 **Best day**: {best_day['date'].strftime('%Y-%m-%d')} with revenue {fmt_money(best_day['total revenue'])}. "
    f"**Worst day**: {worst_day['date'].strftime('%Y-%m-%d')} with revenue {fmt_money(worst_day['total revenue'])}."
)

# --- Top & Bottom Channel Efficiency
if not mkt_f.empty:
    ch_eff = mkt_f.groupby('channel', as_index=False).agg({
        'spend':'sum','attr_revenue':'sum'
    })
    ch_eff['efficiency'] = ch_eff['attr_revenue'] / ch_eff['spend'].replace(0, np.nan)
    best = ch_eff.loc[ch_eff['efficiency'].idxmax()]
    worst = ch_eff.loc[ch_eff['efficiency'].idxmin()]
    st.write(
        f"🚀 **Most Efficient Channel:** {best['channel']} "
        f"({best['efficiency']:.2f} revenue per $1 spend). "
        f"⚠️ Least efficient: {worst['channel']} ({worst['efficiency']:.2f})."
    )

# --- Spend vs Revenue Correlation
if len(merged_f) > 2:
    corr = merged_f['mkt_spend'].corr(merged_f['total revenue'])
    relation = ("strong positive" if corr > 0.5 else
                "moderate" if corr > 0.2 else
                "weak or negative")
    st.write(
        f"🔗 **Spend–Revenue Correlation:** {corr:.2f} "
        f"({relation} relationship)."
    )

# --- Overall ROAS Takeaway
if avg_roas > 3:
    takeaway = "Excellent marketing efficiency — campaigns are returning over 3× spend."
elif avg_roas > 1:
    takeaway = "Marketing is profitable but could be scaled further."
else:
    takeaway = "ROAS is below 1; consider optimizing campaigns or reducing spend."
st.success(f"📊 **Overall ROAS Insight:** {takeaway}")
# ======================================================================
# ===  DEEP-DIVE TEXT INSIGHTS  ========================================
# ======================================================================

st.markdown("## 🧐 Deeper Narrative Insights")

# 1️⃣ Top-performing tactic
if not mkt_f.empty and "tactic" in mkt_f.columns:
    tactic_perf = mkt_f.groupby("tactic")["roas"].mean().dropna()
    if not tactic_perf.empty:
        best_tac = tactic_perf.idxmax()
        st.write(
            f"🎯 **Best Tactic:** `{best_tac}` with an average ROAS of "
            f"{tactic_perf[best_tac]:.2f}. Consider allocating more spend here."
        )

# 2️⃣ State-level winner
if "state" in mkt_f.columns and not mkt_f["state"].dropna().empty:
    state_perf = mkt_f.groupby("state")["roas"].mean().dropna()
    if not state_perf.empty:
        top_state = state_perf.idxmax()
        st.write(
            f"🗺️ **Top State:** `{top_state}` achieved the highest mean ROAS "
            f"of {state_perf[top_state]:.2f}."
        )

# 3️⃣ Click-Through Rate benchmark
ctr_df = mkt_f.groupby("channel")["ctr"].mean().dropna()
if not ctr_df.empty:
    best_ctr_channel = ctr_df.idxmax()
    st.write(
        f"💡 **Highest Engagement:** `{best_ctr_channel}` leads in average CTR "
        f"at {ctr_df[best_ctr_channel]:.2%}."
    )

# 4️⃣ Marketing share of revenue
if not merged_f.empty:
    share = merged_f["marketing_share_of_revenue"].mean() * 100
    st.write(
        f"📦 **Marketing-Driven Revenue:** On average, marketing campaigns "
        f"accounted for **{share:.1f}%** of total revenue during the selected period."
    )

# 5️⃣ Spend Efficiency Over Time (text summary only)
if len(merged_f) > 14:
    first_half = merged_f.head(len(merged_f)//2)
    second_half = merged_f.tail(len(merged_f)//2)
    eff_change = (
        (second_half["mkt_attr_revenue"].sum() / second_half["mkt_spend"].sum()) -
        (first_half["mkt_attr_revenue"].sum() / first_half["mkt_spend"].sum())
    )
    st.write(
        f"⏱️ **Efficiency Shift:** ROAS changed by "
        f"{eff_change:+.2f} points comparing the first half vs. second half of your selection."
    )

# 6️⃣ Cost per Order
if not merged_f.empty:
    cpo = (merged_f["mkt_spend"].sum() /
           merged_f["# of orders"].sum()) if merged_f["# of orders"].sum() else np.nan
    if not np.isnan(cpo):
        st.write(f"💲 **Cost per Order:** Average marketing spend per order is **${cpo:,.2f}**.")
