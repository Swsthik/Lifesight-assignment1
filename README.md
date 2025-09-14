# Marketing Intelligence Dashboard 📈

A comprehensive marketing analytics dashboard with AI-powered forecasting using Facebook's Prophet algorithm.

## 🚀 Live Demo

**Deployed on Streamlit Cloud**: https://lifesight-bidashboard.streamlit.app/

## ✨ Features

- **Multi-channel Marketing Analysis**: Facebook, Google, TikTok campaign performance
- **Prophet AI Forecasting**: Advanced time series predictions for revenue and orders  
- **Order Attribution**: UTM parameter tracking and customer journey analysis
- **Interactive Visualizations**: Plotly charts with drill-down capabilities
- **ROAS Optimization**: Real-time performance monitoring with customizable targets
- **Business Intelligence**: Growth trends, efficiency metrics, and strategic insights

## 📊 Key Metrics

- Marketing spend and attributed revenue tracking
- Click-through rates (CTR) and cost-per-click (CPC) analysis
- Return on advertising spend (ROAS) optimization
- Geographic performance breakdown
- Campaign and tactic effectiveness analysis

## 🛠️ Technical Stack

- **Framework**: Streamlit
- **Forecasting**: Facebook Prophet
- **Visualization**: Plotly
- **Data Processing**: Pandas, NumPy
- **Deployment**: Streamlit Cloud

## 📋 Usage

1. Upload your marketing data (Facebook.csv, Google.csv, TikTok.csv)
2. Upload business metrics (Business.csv)
3. Optionally upload order-level data for detailed attribution
4. Use filters to analyze specific time periods, channels, or campaigns
5. Review forecasts and business insights for data-driven decisions

## 🔧 Local Development

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

## 📁 Data Format

The dashboard expects CSV files with the following structure:

### Marketing Data
- `date`: Date of campaign activity
- `channel`: Platform name (Facebook, Google, TikTok)
- `spend`: Marketing spend amount
- `impressions`: Number of ad impressions
- `clicks`: Number of clicks
- `attr_revenue`: Attributed revenue

### Business Data  
- `date`: Date of business activity
- `# of orders`: Number of orders
- `total revenue`: Total business revenue
- `gross profit`: Gross profit amount

### Order Data (Optional)
- `order_date`: Date of order
- `order_revenue`: Revenue from order
- `utm_source`, `utm_medium`, `utm_campaign`: Attribution parameters

---

**Built with ❤️ Swsthik**