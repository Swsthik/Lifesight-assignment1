# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import timedelta
import warnings

st.set_page_config(layout="wide", page_title="Marketing Intelligence Dashboard (Prophet + Order Attribution)")

# Try to import Prophet; if unavailable we'll fall back to moving-average forecast
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception as e:
    PROPHET_AVAILABLE = False

warnings.filterwarnings("ignore")

# ---------- Helpers ----------
def safe_div(a, b):
    return a / b.replace(0, np.nan)

@st.cache_data
def load_marketing(csv_paths):
    frames = []
    for p, channel_name in csv_paths:
        df = pd.read_csv(p, parse_dates=['date'])
        df['channel'] = channel_name
        frames.append(df)
    m = pd.concat(frames, ignore_index=True)
    m.columns = [c.strip() for c in m.columns]
    for col in ['impression','impressions','clicks','spend','attributed revenue','attr_revenue']:
        if col in m.columns:
            m[col] = pd.to_numeric(m[col], errors='coerce').fillna(0)
    if 'attributed revenue' in m.columns:
        m = m.rename(columns={'attributed revenue':'attr_revenue'})
    if 'impression' in m.columns:
        m = m.rename(columns={'impression':'impressions'})
    return m

@st.cache_data
def load_business(path):
    b = pd.read_csv(path, parse_dates=['date'])
    b.columns = [c.strip() for c in b.columns]
    for col in b.columns:
        if col != 'date':
            try:
                b[col] = pd.to_numeric(b[col], errors='coerce').fillna(0)
            except:
                pass
    return b

def prepare_marketing_aggregates(mkt_df):
    for req in ['impressions','clicks','spend','attr_revenue']:
        if req not in mkt_df.columns:
            mkt_df[req] = 0
    if 'campaign' in mkt_df.columns:
        parsed = mkt_df['campaign'].astype(str).str.split('-', n=4, expand=True)
        for i in range(parsed.shape[1]):
            mkt_df[f'campaign_part_{i}'] = parsed[i].str.strip()
    agg = mkt_df.groupby(['date','channel','state','tactic','campaign'], as_index=False).agg({
        'impressions':'sum','clicks':'sum','spend':'sum','attr_revenue':'sum'
    })
    agg['ctr'] = safe_div(agg['clicks'], agg['impressions'])
    agg['cpc'] = safe_div(agg['spend'], agg['clicks'])
    agg['roas'] = safe_div(agg['attr_revenue'], agg['spend'])
    return agg

def merge_with_business(mkt_daily, business):
    daily_mkt = mkt_daily.groupby('date', as_index=False).agg({
        'impressions':'sum','clicks':'sum','spend':'sum','attr_revenue':'sum'
    }).rename(columns={'attr_revenue':'mkt_attr_revenue', 'spend':'mkt_spend'})
    df = pd.merge(business, daily_mkt, on='date', how='left').fillna(0)
    df['marketing_share_of_revenue'] = safe_div(df['mkt_attr_revenue'], df['total revenue'])
    df['revenue_per_order'] = safe_div(df['total revenue'], df['# of orders'])
    return df

def moving_average_forecast(series, periods=7, forecast_horizon=14):
    last_avg = series.dropna().tail(periods).mean()
    idx_last = series.index.max()
    future_index = [idx_last + timedelta(days=i) for i in range(1, forecast_horizon+1)]
    forecast_values = [last_avg]*forecast_horizon
    forecast_df = pd.DataFrame({'ds':future_index, 'yhat':forecast_values})
    return forecast_df

def prophet_forecast(series, forecast_horizon=14, changepoint_prior_scale=0.5):
    """series: pd.Series indexed by date"""
    if not PROPHET_AVAILABLE:
        return None
    df = series.reset_index().rename(columns={'date':'ds', series.name:'y'}) if series.name != 'y' else series.reset_index().rename(columns={'date':'ds'})
    df.columns = ['ds', 'y']
    if df['y'].isna().all() or len(df) < 14:
        return None
    m = Prophet(changepoint_prior_scale=changepoint_prior_scale, daily_seasonality=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=forecast_horizon)
    forecast = m.predict(future)[['ds','yhat','yhat_lower','yhat_upper']]
    return forecast

def order_attribution_from_order_file(orders_df):
    """
    Normalize columns and ensure we have: order_id, order_date (datetime), order_revenue (float),
    utm_source/utm_medium/utm_campaign (optional)
    """
    df = orders_df.copy()
    # Try to find revenue column
    revenue_cols = [c for c in df.columns if c.lower() in ('order_revenue','revenue','total','amount','order_total','total_revenue')]
    if revenue_cols:
        df['order_revenue'] = pd.to_numeric(df[revenue_cols[0]], errors='coerce').fillna(0)
    else:
        # fallback: if business daily revenue exists we'll use aggregated mapping; otherwise zero
        df['order_revenue'] = 0.0

    # find order_date column
    date_cols = [c for c in df.columns if 'date' in c.lower() and 'order' in c.lower() or c.lower()=='order_date' or c.lower()=='date']
    if date_cols:
        df['order_date'] = pd.to_datetime(df[date_cols[0]])
    else:
        st.warning("No order_date found in orders file; attribution will be limited.")
        df['order_date'] = pd.NaT

    # normalize utm columns if present
    for u in ['utm_source','utm_medium','utm_campaign','utm_term','utm_content']:
        if u not in df.columns:
            # try variants
            candidates = [c for c in df.columns if c.lower().endswith(u)]
            if candidates:
                df[u] = df[candidates[0]]
            else:
                df[u] = None
    return df

# ---------- Load marketing & business ----------
st.sidebar.header("Data input")
use_upload = st.sidebar.checkbox("Upload CSVs instead of using local data (marketing, business, orders)", value=False)

if use_upload:
    st.info("Upload marketing files (Facebook.csv, Google.csv, TikTok.csv), business file (Business.csv), and optionally an order-level file (Orders.csv).")
    fb_file = st.sidebar.file_uploader("Facebook.csv", type=["csv"])
    g_file = st.sidebar.file_uploader("Google.csv", type=["csv"])
    tt_file = st.sidebar.file_uploader("TikTok.csv", type=["csv"])
    b_file = st.sidebar.file_uploader("Business.csv", type=["csv"])
    orders_file = st.sidebar.file_uploader("Orders (optional).csv", type=["csv"])
    if not (fb_file and g_file and tt_file and b_file):
        st.warning("Upload the three marketing files and the business file to proceed.")
        st.stop()
    fb = pd.read_csv(fb_file, parse_dates=['date'])
    g = pd.read_csv(g_file, parse_dates=['date'])
    tt = pd.read_csv(tt_file, parse_dates=['date'])
    b = pd.read_csv(b_file, parse_dates=['date'])
    fb['channel']='Facebook'; g['channel']='Google'; tt['channel']='TikTok'
    marketing_raw = pd.concat([fb,g,tt], ignore_index=True)
    business = b
    orders_df = None
    if orders_file:
        orders_df = pd.read_csv(orders_file)
else:
    data_folder = Path(".")
    st.sidebar.write("Using local CSVs in current folder. You can toggle to upload.")
    fb_path = data_folder / "Facebook.csv"
    g_path = data_folder / "Google.csv"
    tt_path = data_folder / "TikTok.csv"
    b_path = data_folder / "Business.csv"
    orders_path = data_folder / "Orders.csv"
    if not (fb_path.exists() and g_path.exists() and tt_path.exists() and b_path.exists()):
        st.error("Local CSVs not found in current directory. Toggle 'Upload CSVs' or place files in current directory.")
        st.stop()
    marketing_raw = load_marketing([(fb_path,'Facebook'), (g_path,'Google'), (tt_path,'TikTok')])
    business = load_business(b_path)
    orders_df = None
    if orders_path.exists():
        orders_df = pd.read_csv(orders_path)

# Normalize marketing column names
marketing_raw.columns = [c.strip() for c in marketing_raw.columns]
if 'attributed revenue' in marketing_raw.columns:
    marketing_raw = marketing_raw.rename(columns={'attributed revenue':'attr_revenue'})
if 'impression' in marketing_raw.columns:
    marketing_raw = marketing_raw.rename(columns={'impression':'impressions'})

marketing_agg = prepare_marketing_aggregates(marketing_raw)
business_clean = business.copy()
business_clean.columns = [c.strip() for c in business_clean.columns]
merged = merge_with_business(marketing_agg, business_clean)

# ---------- Filters ----------
st.sidebar.header("Filters")
min_date = merged['date'].min()
max_date = merged['date'].max()
date_range = st.sidebar.date_input("Date range", [min_date, max_date], min_value=min_date, max_value=max_date)

channels = sorted(marketing_agg['channel'].unique().tolist())
sel_channels = st.sidebar.multiselect("Channel", channels, default=channels)

states = sorted(marketing_agg['state'].dropna().unique().tolist())
sel_states = st.sidebar.multiselect("State", states, default=None)

tactics = sorted(marketing_agg['tactic'].dropna().unique().tolist())
sel_tactics = st.sidebar.multiselect("Tactic", tactics, default=None)

start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
mkt_f = marketing_agg[(marketing_agg['date'] >= start_date) & (marketing_agg['date'] <= end_date)]
if sel_channels:
    mkt_f = mkt_f[mkt_f['channel'].isin(sel_channels)]
if sel_states:
    mkt_f = mkt_f[mkt_f['state'].isin(sel_states)]
if sel_tactics:
    mkt_f = mkt_f[mkt_f['tactic'].isin(sel_tactics)]

merged_f = merged[(merged['date'] >= start_date) & (merged['date'] <= end_date)]

# ---------- Top KPIs ----------
st.title("Marketing Intelligence Dashboard (Prophet + Order Attribution)")


# --- Create a placeholder container so the insights render first and auto-update
insights_container = st.empty()

# Calculate KPIs as before
k1,k2,k3,k4 = st.columns(4)
total_spend = mkt_f['spend'].sum()
total_attr_rev = mkt_f['attr_revenue'].sum()
agg_roas = total_attr_rev / total_spend if total_spend > 0 else np.nan
total_orders = merged_f['# of orders'].sum()
total_revenue = merged_f['total revenue'].sum()
gross_profit = merged_f['gross profit'].sum()

k1.metric("Marketing Spend", f"${total_spend:,.0f}")
k2.metric("Attributed Revenue (marketing)", f"${total_attr_rev:,.0f}",
          delta=f"ROAS {agg_roas:.2f}" if not np.isnan(agg_roas) else "ROAS N/A")
k3.metric("Orders (total)", f"{int(total_orders):,}")
k4.metric("Revenue (total)", f"${total_revenue:,.0f}")

# --- Now fill the placeholder with the Key Business Insights section
with insights_container:
    st.markdown("## 💡 Key Business Insights")

    def fmt_money(x):
        return f"${x:,.0f}"

    target_roas = st.sidebar.slider("Target ROAS", 0.0, 5.0, 2.0, 0.1)
    roas_df = mkt_f.groupby('date', as_index=False)[['attr_revenue','spend']].sum()
    roas_df['roas'] = roas_df['attr_revenue'] / roas_df['spend'].replace(0, np.nan)
    avg_roas = roas_df['roas'].mean(skipna=True)

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Average ROAS", f"{avg_roas:.2f}")
    low_days = (roas_df['roas'] < target_roas).sum()
    kpi2.metric("Days Below Target", f"{low_days}")
    kpi3.metric("Target ROAS", f"{target_roas:.2f}",
                delta="Below Target!" if low_days else "")

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


k1,k2,k3,k4 = st.columns(4)
total_spend = mkt_f['spend'].sum()
total_attr_rev = mkt_f['attr_revenue'].sum()
agg_roas = total_attr_rev / total_spend if total_spend>0 else np.nan
total_orders = merged_f['# of orders'].sum()
total_revenue = merged_f['total revenue'].sum()
gross_profit = merged_f['gross profit'].sum()

k1.metric("Marketing Spend", f"${total_spend:,.0f}")
k2.metric("Attributed Revenue (marketing)", f"${total_attr_rev:,.0f}", delta=f"ROAS {agg_roas:.2f}" if not np.isnan(agg_roas) else "ROAS N/A")
k3.metric("Orders (total)", f"{int(total_orders):,}")
k4.metric("Revenue (total)", f"${total_revenue:,.0f}")

# ---------- Time series (historical) ----------
st.subheader("Trends — Spend vs Attributed Revenue vs Business Revenue")
time_df = merged_f[['date','mkt_spend','mkt_attr_revenue','total revenue']].sort_values('date')
fig = go.Figure()
fig.add_trace(go.Bar(x=time_df['date'], y=time_df['mkt_spend'], name='Marketing Spend', yaxis='y1', opacity=0.6))
fig.add_trace(go.Scatter(mode='lines', x=time_df['date'], y=time_df['mkt_attr_revenue'], name='Marketing Attributed Revenue', yaxis='y2'))
fig.add_trace(go.Scatter(mode='lines', x=time_df['date'], y=time_df['total revenue'], name='Business Total Revenue', yaxis='y2', line=dict(dash='dash')))
fig.update_layout(xaxis_title="Date",
                  yaxis=dict(title="Spend", side='left', showgrid=False),
                  yaxis2=dict(title="Revenue", overlaying='y', side='right'),
                  legend=dict(x=0, y=1.12, orientation='h'),
                  margin=dict(t=40))
st.plotly_chart(fig, use_container_width=True)

# ---------- Stronger forecasting with Prophet (orders & revenue) ----------
st.subheader("Forecast: Prophet (if available) — Orders & Revenue")
if PROPHET_AVAILABLE:
    st.success("Prophet available — using Prophet for forecasts.")
else:
    st.warning("Prophet is not installed. Forecasting will use simple moving-average. To enable Prophet install the package in requirements (`pip install prophet`).")

# Prepare series
orders_series = merged_f.set_index('date')['# of orders'].resample('D').sum().reindex(pd.date_range(start_date, end_date, freq='D')).fillna(0)
revenue_series = merged_f.set_index('date')['total revenue'].resample('D').sum().reindex(pd.date_range(start_date, end_date, freq='D')).fillna(0)

forecast_horizon = 14

# Orders forecast
if PROPHET_AVAILABLE and len(orders_series.dropna()) >= 30:
    with st.spinner("Fitting Prophet for orders..."):
        forecast_orders = prophet_forecast(orders_series, forecast_horizon=forecast_horizon)
    if forecast_orders is not None:
        figo = go.Figure()
        figo.add_trace(go.Scatter(x=orders_series.index, y=orders_series.values, name='Historical Orders'))
        figo.add_trace(go.Scatter(x=forecast_orders['ds'], y=forecast_orders['yhat'], name='Prophet Forecast', line=dict(dash='dash')))
        figo.add_trace(go.Scatter(x=forecast_orders['ds'], y=forecast_orders['yhat_upper'], name='Upper', line=dict(width=0), showlegend=False))
        figo.add_trace(go.Scatter(x=forecast_orders['ds'], y=forecast_orders['yhat_lower'], name='Lower', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,0,0,0.05)', showlegend=False))
        figo.update_layout(title="Orders — Prophet 14-day forecast", xaxis_title="Date", yaxis_title="Orders")
        st.plotly_chart(figo, use_container_width=True)
    else:
        st.info("Not enough history to run a robust Prophet model for orders. Falling back to moving-average.")
        fo = moving_average_forecast(orders_series, periods=7, forecast_horizon=forecast_horizon)
        figo = go.Figure()
        figo.add_trace(go.Scatter(mode='lines', x=orders_series.index, y=orders_series.values, name='Historical Orders'))
        figo.add_trace(go.Scatter(mode='lines', x=fo['ds'], y=fo['yhat'], name='MA Forecast', line=dict(dash='dash')))
        st.plotly_chart(figo, use_container_width=True)
else:
    fo = moving_average_forecast(orders_series, periods=7, forecast_horizon=forecast_horizon)
    figo = go.Figure()
    figo.add_trace(go.Scatter(mode='lines', x=orders_series.index, y=orders_series.values, name='Historical Orders'))
    figo.add_trace(go.Scatter(mode='lines', x=fo['ds'], y=fo['yhat'], name='MA Forecast', line=dict(dash='dash')))
    st.plotly_chart(figo, use_container_width=True)

# Revenue forecast
if PROPHET_AVAILABLE and len(revenue_series.dropna()) >= 30:
    with st.spinner("Fitting Prophet for revenue..."):
        forecast_revenue = prophet_forecast(revenue_series, forecast_horizon=forecast_horizon)
    if forecast_revenue is not None:
        figr = go.Figure()
        figr.add_trace(go.Scatter(mode='lines', x=revenue_series.index, y=revenue_series.values, name='Historical Revenue'))
        figr.add_trace(go.Scatter(mode='lines', x=forecast_revenue['ds'], y=forecast_revenue['yhat'], name='Prophet Forecast', line=dict(dash='dash')))
        figr.add_trace(go.Scatter(x=forecast_revenue['ds'], y=forecast_revenue['yhat_upper'], name='Upper', line=dict(width=0), showlegend=False))
        figr.add_trace(go.Scatter(x=forecast_revenue['ds'], y=forecast_revenue['yhat_lower'], name='Lower', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,0,0,0.05)', showlegend=False))
        figr.update_layout(title="Revenue — Prophet 14-day forecast", xaxis_title="Date", yaxis_title="Revenue")
        st.plotly_chart(figr, use_container_width=True)
    else:
        st.info("Not enough history to run a robust Prophet model for revenue. Falling back to moving-average.")
        fr = moving_average_forecast(revenue_series, periods=7, forecast_horizon=forecast_horizon)
        figr = go.Figure()
        figr.add_trace(go.Scatter(mode='lines', x=revenue_series.index, y=revenue_series.values, name='Historical Revenue'))
        figr.add_trace(go.Scatter(mode='lines', x=fr['ds'], y=fr['yhat'], name='MA Forecast', line=dict(dash='dash')))
        st.plotly_chart(figr, use_container_width=True)
else:
    fr = moving_average_forecast(revenue_series, periods=7, forecast_horizon=forecast_horizon)
    figr = go.Figure()
    figr.add_trace(go.Scatter(mode='lines', x=revenue_series.index, y=revenue_series.values, name='Historical Revenue'))
    figr.add_trace(go.Scatter(mode='lines', x=fr['ds'], y=fr['yhat'], name='MA Forecast', line=dict(dash='dash')))
    st.plotly_chart(figr, use_container_width=True)

# ---------- Campaign & tactic performance ----------
st.subheader("Channel & Campaign Performance")
ch_agg = mkt_f.groupby('channel', as_index=False).agg({
    'impressions':'sum','clicks':'sum','spend':'sum','attr_revenue':'sum'
})
ch_agg['ctr'] = safe_div(ch_agg['clicks'], ch_agg['impressions'])
ch_agg['cpc'] = safe_div(ch_agg['spend'], ch_agg['clicks'])
ch_agg['roas'] = safe_div(ch_agg['attr_revenue'], ch_agg['spend'])
fig_ch = px.bar(ch_agg.melt(id_vars='channel', value_vars=['spend','attr_revenue'], var_name='metric', value_name='value'),
                x='channel', y='value', color='metric', barmode='group', title='Spend vs Attributed Revenue by Channel')
st.plotly_chart(fig_ch, use_container_width=True)

camp_agg = mkt_f.groupby(['channel','campaign'], as_index=False).agg({
    'impressions':'sum','clicks':'sum','spend':'sum','attr_revenue':'sum'
})
camp_agg['ctr'] = safe_div(camp_agg['clicks'], camp_agg['impressions'])
camp_agg['cpc'] = safe_div(camp_agg['spend'], camp_agg['clicks'])
camp_agg['roas'] = safe_div(camp_agg['attr_revenue'], camp_agg['spend'])
st.dataframe(camp_agg.sort_values('spend', ascending=False).head(200).style.format({
    'spend':"${:,.2f}", 'attr_revenue':"${:,.2f}", 'ctr':'{:.2%}', 'cpc':"${:,.2f}", 'roas':'{:.2f}'
}))

# ---------- Order-level attribution & cohort LTV ----------
st.subheader("Order-level Attribution & Cohort LTV (optional)")

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
