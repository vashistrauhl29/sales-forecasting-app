import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import io

st.set_page_config(page_title="Sales Forecasting Tool", layout="wide")

st.title("📊 Quarterly Sales Forecasting Tool")
st.markdown("Upload your Excel file in the correct format to analyze trends and generate forecasts.")

# Upload section
st.sidebar.markdown("### 📎 Upload Sales Excel")
uploaded_file = st.sidebar.file_uploader("Upload your sales data (.xlsx)", type=["xlsx"])

# Forecast horizon slider
st.sidebar.markdown("## 🔧 Forecast Settings")
forecast_horizon = st.sidebar.slider(
    "Select forecast horizon (in quarters):",
    min_value=1,
    max_value=12,
    value=4,
    step=1
)

# Sample download
with open("template_sales_data.xlsx", "rb") as f:
    st.sidebar.download_button("📥 Download Sample Excel Template", f, file_name="template_sales_data.xlsx")

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)

        # Standardize column names
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

        required_cols = {"year", "quarter", "time_period", "actual_sales"}
        if not required_cols.issubset(df.columns):
            st.error(f"❌ Required columns missing. Found columns: {df.columns}")
        else:
            df = df.sort_values("time_period")
            df["period_label"] = df["year"].astype(str) + "-Q" + df["quarter"].astype(str)

            # Step 1: Visualize sales trend
            st.markdown("### 🔍 Step 1: Visualize Sales Trend")
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            ax1.plot(df["period_label"], df["actual_sales"], marker='o', linewidth=2)
            ax1.set_title("Quarterly Sales Trend", fontsize=16)
            ax1.set_xlabel("Quarter", fontsize=12)
            ax1.set_ylabel("Sales", fontsize=12)
            ax1.grid(True, linestyle='--', alpha=0.5)
            ax1.tick_params(axis='x', rotation=45)
            fig1.tight_layout()
            st.pyplot(fig1)

            # Step 2: Stationarity
            st.markdown("### 📉 Step 2: Stationarity Check")
            diff = df["actual_sales"].diff().dropna()
            if diff.std() / df["actual_sales"].std() < 0.5:
                st.success("✅ Likely Stationary Data")
                stationary = True
            else:
                st.warning("⚠️ Likely Non-Stationary Data")
                stationary = False

            # Step 3: Seasonality detection
            st.markdown("### 📊 Step 3: Seasonality Detection")
            quarterly_avg = df.groupby("quarter")["actual_sales"].mean()
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            bars = ax2.bar(quarterly_avg.index.astype(str), quarterly_avg.values, color='skyblue')
            ax2.set_title("Average Sales by Quarter", fontsize=14)
            ax2.set_xlabel("Quarter")
            ax2.set_ylabel("Average Sales")
            ax2.grid(axis='y', linestyle='--', alpha=0.4)
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
            st.pyplot(fig2)

            seasonal_strength = quarterly_avg.max() - quarterly_avg.min()
            if seasonal_strength > 0.05 * df["actual_sales"].mean():
                st.success("✅ Seasonality Detected")
                seasonality = True
            else:
                st.warning("⚠️ No Strong Seasonality Detected")
                seasonality = False

            # Step 4: Additive vs Multiplicative
            st.markdown("### ➕✖️ Step 4: Additive or Multiplicative")
            if seasonality:
                mid = df["time_period"].median()
                early = df[df["time_period"] < mid]
                late = df[df["time_period"] >= mid]

                range_early = early["actual_sales"].max() - early["actual_sales"].min()
                range_late = late["actual_sales"].max() - late["actual_sales"].min()

                if abs(range_late - range_early) / range_early < 0.5:
                    st.info("🔎 Additive Seasonality Detected")
                    seasonality_type = "additive"
                else:
                    st.info("🔎 Multiplicative Seasonality Detected")
                    seasonality_type = "multiplicative"
            else:
                seasonality_type = "none"

            # Optional Insight for Stationarity + Multiplicative Seasonality
            if stationary and seasonality_type == "multiplicative":
                with st.expander("🤔 How can the data be both stationary and have multiplicative seasonality?"):
                    st.markdown("""
            **Great observation!**  
            Yes, a time series can be both **stationary** and have **multiplicative seasonality**. Here's how:

            - 📉 **Stationary** means the trend has been removed and the data fluctuates around a stable mean.
            - ✖️ **Multiplicative seasonality** means seasonal effects scale with the level of the series — the percentage impact is consistent over time.
            - So, even if the absolute values change, their **relative seasonal pattern remains constant**, and the differenced series appears stable.

            ✅ This is perfectly valid and often seen in real-world business data like sales, revenue, and demand cycles.
                    """)


            # Step 5: Forecasting
            st.markdown("### 🚀 Step 5: Forecast")

            import warnings
            from statsmodels.tsa.arima.model import ARIMA
            from sklearn.metrics import mean_squared_error

            warnings.filterwarnings("ignore")

            # --- ETS Model ---
            ets_model = ExponentialSmoothing(
                df["actual_sales"],
                seasonal="add" if seasonality_type == "additive" else "mul" if seasonality_type == "multiplicative" else None,
                seasonal_periods=4,
                trend="add",
                initialization_method="estimated"
            ).fit()
            ets_forecast = ets_model.forecast(forecast_horizon)
            ets_rmse = np.sqrt(mean_squared_error(df["actual_sales"], ets_model.fittedvalues))

            # --- ARIMA Model ---
            try:
                arima_model = ARIMA(df["actual_sales"], order=(1, 1, 1)).fit()
                arima_predicted = arima_model.predict(start=1, end=len(df["actual_sales"]) - 1)
                arima_actual = df["actual_sales"].iloc[1:]
                arima_rmse = np.sqrt(mean_squared_error(arima_actual, arima_predicted))
            except:
                arima_model = None
                arima_rmse = float("inf")

            # --- Choose Best Model ---
            if arima_rmse + 0.01 < ets_rmse:
                chosen_model = "ARIMA(1,1,1)"
                model_fit = arima_model
                forecast = arima_model.forecast(forecast_horizon)
                model_reason = f"📈 **ARIMA** chosen due to better in-sample fit (RMSE: {arima_rmse:.2f}) vs ETS (RMSE: {ets_rmse:.2f})"
            else:
                chosen_model = "Exponential Smoothing"
                model_fit = ets_model
                forecast = ets_forecast
                model_reason = f"📊 **ETS** selected (RMSE: {ets_rmse:.2f}) vs ARIMA (RMSE: {arima_rmse:.2f})"

            st.info(model_reason)

            # --- Future period labels ---
            last_year = df["year"].iloc[-1]
            last_quarter = df["quarter"].iloc[-1]
            future_labels = []
            for i in range(1, forecast_horizon + 1):
                next_q = last_quarter + i
                next_y = last_year + (next_q - 1) // 4
                next_q = ((next_q - 1) % 4) + 1
                future_labels.append(f"{next_y}-Q{next_q}")

            forecast_df = pd.DataFrame({
                "Period": future_labels,
                "Forecasted Sales": forecast.round(2)
            })
            st.dataframe(forecast_df)

            # --- Forecast Plot ---
            st.markdown("### 📊 Forecast Plot")
            full_x = list(df["period_label"]) + future_labels
            full_y_actual = list(df["actual_sales"])
            full_y_forecast = [None] * len(df["actual_sales"]) + list(forecast)

            fig3, ax3 = plt.subplots(figsize=(12, 6))
            ax3.plot(full_x, full_y_actual + [None] * forecast_horizon, label="Actual", marker="o", linewidth=2)
            ax3.plot(full_x, full_y_forecast, label="Forecast", marker="o", linestyle="--", linewidth=2, color="orange")

            # --- Confidence Intervals (ARIMA only) ---
            try:
                if chosen_model.startswith("ARIMA"):
                    pred_ci = model_fit.get_prediction(start=len(df), end=len(df) + forecast_horizon - 1).conf_int()
                    ax3.fill_between(future_labels, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='orange', alpha=0.2, label="95% CI")
                else:
                    st.info("ℹ️ Forecast confidence intervals are only shown when ARIMA is selected.")
            except Exception as e:
                st.warning(f"⚠️ Confidence interval calculation failed: {e}")

            ax3.set_title("Actual vs Forecasted Sales", fontsize=16)
            ax3.set_xlabel("Time Period", fontsize=12)
            ax3.set_ylabel("Sales", fontsize=12)
            ax3.grid(True, linestyle='--', alpha=0.5)
            ax3.legend()
            ax3.tick_params(axis='x', rotation=45)
            fig3.tight_layout()
            st.pyplot(fig3)

            # Step 6: Model Performance Summary
            st.markdown("### 📈 Step 6: Model Performance Summary")

            from sklearn.metrics import mean_absolute_error

            actual_vals = df["actual_sales"]
            fitted_vals = model_fit.fittedvalues

            mae = mean_absolute_error(actual_vals, fitted_vals)
            rmse = np.sqrt(mean_squared_error(actual_vals, fitted_vals))

            st.write(f"**MAE (Mean Absolute Error):** {mae:.2f}")
            st.write(f"**RMSE (Root Mean Squared Error):** {rmse:.2f}")
            if rmse < 0.1 * actual_vals.mean():
                st.success("✅ Forecast model fits well!")
            else:
                st.warning("⚠️ Consider refining the model — moderate forecast error.")

            # Residual plot
            residuals = actual_vals - fitted_vals
            fig4, ax4 = plt.subplots(figsize=(10, 4))
            ax4.plot(df["period_label"], residuals, marker='o', linestyle='-', color='crimson')
            ax4.axhline(0, color='gray', linestyle='--', linewidth=1)
            ax4.set_title("Residual Plot", fontsize=14)
            ax4.set_xlabel("Quarter")
            ax4.set_ylabel("Residual")
            ax4.grid(True, linestyle='--', alpha=0.4)
            ax4.tick_params(axis='x', rotation=45)
            fig4.tight_layout()
            st.pyplot(fig4)


            # Step 9: Insights & Recommendations
            st.markdown("### 💡 Step 9: Insights & Recommendations")

            insights = []

            # Trend detection
            if df["actual_sales"].iloc[-1] > df["actual_sales"].iloc[0]:
                insights.append("📈 Sales show an **upward trend** over time.")
            elif df["actual_sales"].iloc[-1] < df["actual_sales"].iloc[0]:
                insights.append("📉 Sales show a **downward trend** over time.")
            else:
                insights.append("➖ Sales appear **stable** over the observed period.")

            # Forecast direction using slope and range analysis
            forecast_trend_slope = np.polyfit(range(len(forecast)), forecast, 1)[0]
            first_forecast = forecast.iloc[0]
            last_forecast = forecast.iloc[-1]
            last_actual = df["actual_sales"].iloc[-1]

            if forecast_trend_slope > 0.5:
                insights.append("🔮 The model predicts a **strong upward trend** in sales over the forecast period.")
            elif forecast_trend_slope < -0.5:
                insights.append("🔮 The model forecasts a **declining trend** in sales over the forecast period.")
            elif abs(last_forecast - first_forecast) < 0.05 * last_actual:
                insights.append("🔮 Sales are expected to **remain relatively stable** over the forecast period.")
            else:
                insights.append("🔮 The model anticipates **minor fluctuations** but no clear trend in future sales.")

            # Model quality
            if rmse < 0.1 * actual_vals.mean():
                insights.append("✅ Forecast model is **highly reliable** based on RMSE.")
            else:
                insights.append("⚠️ Forecast model has **moderate errors**. Consider using more data or reviewing outliers.")

            # Seasonality note
            if seasonality:
                insights.append(f"🔁 Seasonal pattern is **{seasonality_type}**, with noticeable variation across quarters.")
            else:
                insights.append("📊 No strong seasonality was detected in the data.")

             # Recommendations
            st.markdown("#### 📌 Summary:")
            for insight in insights:
                st.write(insight)

            # Step 7: Download Forecast
            st.markdown("### 💾 Download Forecast")
            csv = forecast_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Forecast as CSV",
                data=csv,
                file_name="forecast_output.csv",
                mime="text/csv"
            )

            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                forecast_df.to_excel(writer, sheet_name="Forecast", index=False)
            excel_buffer.seek(0)
            st.download_button(
                label="📥 Download Forecast as Excel",
                data=excel_buffer,
                file_name="forecast_output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            import pdfkit
            import tempfile
            import streamlit.components.v1 as components

            st.markdown("### 🖨️ Export Full Page as PDF")

            if st.button("📄 Download Full Report (Web View)"):
                try:
                    # Create a temporary HTML snapshot
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_html:
                        # Capture everything shown on the page into HTML
                        html_content = f"""
                        <html>
                        <head>
                            <meta charset="UTF-8">
                            <title>Forecast Report</title>
                        </head>
                        <body>
                            {st.session_state['_rendered_page']}
                        </body>
                        </html>
                        """
                        tmp_html.write(html_content.encode("utf-8"))
                        tmp_html_path = tmp_html.name

                    # Convert HTML to PDF
                    pdf_output_path = tmp_html_path.replace(".html", ".pdf")
                    pdfkit.from_file(tmp_html_path, pdf_output_path)

                    # Offer download
                    with open(pdf_output_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download Full Report (PDF)",
                            data=f,
                            file_name="streamlit_report.pdf",
                            mime="application/pdf"
                        )
                except Exception as e:
                    st.error(f"❌ Failed to generate full PDF: {e}")

            import yagmail
            # Step 10: Generate PDF Report
            st.markdown("### 📤 Step 10: Email Full Report")

            import tempfile
            from fpdf import FPDF
            import yagmail
            import base64
            import os
            import matplotlib.pyplot as plt

            # Emoji-safe replacement
            emoji_map = {
                "📈": "[Upward Trend]",
                "📉": "[Downward Trend]",
                "🔮": "[Forecast]",
                "📊": "[Model]",
                "✅": "[Good Fit]",
                "⚠️": "[Warning]",
                "🔁": "[Seasonality]",
                "➖": "[Stable]",
                "📧": "[Email]",
                "📌": "[Summary]",
            }

            def replace_emojis(text):
                for emoji, word in emoji_map.items():
                    text = text.replace(emoji, word)
                return text

            # Email input field
            email_recipient = st.text_input("Enter your email address to receive full report (PDF format):")

            if st.button("📧 Send Report"):
                try:
                    # 1. Generate PDF using FPDF
                    pdf = FPDF()
                    pdf.set_auto_page_break(auto=True, margin=15)
                    pdf.add_page()
                    pdf.set_font("Arial", 'B', 16)
                    pdf.cell(200, 10, "Quarterly Sales Forecast Report", ln=True, align='C')
                    pdf.ln(10)

                    # Add key insights (emoji-safe)
                    pdf.set_font("Arial", size=12)
                    for insight in insights:
                        clean_text = replace_emojis(insight)
                        pdf.multi_cell(0, 10, clean_text)

                    # Save plots as images
                    plot_paths = []
                    for i, fig in enumerate([fig1, fig2, fig3, fig4]):
                        img_path = os.path.join(tempfile.gettempdir(), f"plot_{i}.png")
                        fig.savefig(img_path, dpi=300, bbox_inches='tight')
                        plot_paths.append(img_path)

                    for img_path in plot_paths:
                        pdf.add_page()
                        pdf.image(img_path, x=10, w=190)

                    # Save final report
                    report_path = os.path.join(tempfile.gettempdir(), "forecast_report.pdf")
                    pdf.output(report_path)

                    # 2. Send email (via .streamlit/secrets.toml)
                    email_user = st.secrets["general"]["email_user"]
                    email_pass = st.secrets["general"]["email_pass"]
                    yag = yagmail.SMTP(user=email_user, password=email_pass)
                    yag.send(
                        to=email_recipient,
                        subject="Your Quarterly Forecast Report",
                        contents="Attached is your full sales forecasting report in PDF format.",
                        attachments=report_path
                    )
                    st.success(f"📧 Report sent successfully to {email_recipient}!")

                except Exception as e:
                    st.error(f"❌ Failed to send report: {e}")


            # Email Export Section
            st.markdown("### 📧 Email Forecast")

            recipient_email = st.text_input("Enter recipient email address:")
            if st.button("✉️ Send Forecast via Email"):
                if recipient_email:
                    try:
                        # Save Excel to disk temporarily
                        with open("forecast_output.xlsx", "wb") as f:
                            f.write(excel_buffer.getbuffer())

                        email_user = st.secrets["general"]["email_user"]
                        email_pass = st.secrets["general"]["email_pass"]
                        yag = yagmail.SMTP(user=email_user, password=email_pass)
                        yag.send(
                            to=recipient_email,
                            subject="Your Sales Forecast from Streamlit App",
                            contents="Hi,\n\nAttached is your forecasted sales data generated using our Streamlit tool.\n\nBest,\nSales Forecasting Tool",
                            attachments="forecast_output.xlsx"
                        )
                        st.success(f"📤 Forecast sent to {recipient_email}!")
                    except Exception as e:
                        st.error(f"❌ Failed to send email: {e}")
                else:
                    st.warning("⚠️ Please enter a valid email address.")

    except Exception as e:
        st.error(f"❌ Forecasting failed: {e}")

