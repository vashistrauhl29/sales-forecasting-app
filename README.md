# 📊 Quarterly Sales Forecasting App

This Streamlit app lets you upload quarterly sales data and generates forecasts using dynamic time series models (ETS or ARIMA). Features include:

- 📈 Trend and seasonality analysis  
- 🤖 Auto-selection of best forecasting model  
- 🔮 Forecast visualization and downloadable results  
- 📬 Email export and PDF report  
- 🧠 Smart insights and recommendations

### 🔧 How to use

1. Upload your Excel file (use the provided sample format).
2. Choose the forecast horizon (number of future quarters).
3. View interactive plots and insights.
4. Download forecasts or email the report.

### 📁 File format

The uploaded Excel must include these columns:
- `Year`
- `Quarter`
- `Time Period` (numeric ordering)
- `Actual Sales`

Use `template_sales_data.xlsx` for guidance.

---

### 🚀 Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py

