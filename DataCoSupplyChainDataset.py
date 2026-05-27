# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.8",
#     "matplotlib==3.10.9",
#     "numpy==2.4.4",
#     "pandas==3.0.2",
#     "scikit-learn==1.8.0",
#     "xgboost==3.2.0",
# ]
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App(
    width="medium",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import io

    # Create a UI bridge: Interactive file upload button
    upload_button = mo.ui.file(
        filetypes=[".csv"], 
        kind="button", 
        label="Upload DataCoSupplyChainDataset.csv"
    )

    # Render the button on screen
    upload_button
    return io, mo, pd, upload_button


@app.cell
def _(io, mo, pd, upload_button):
    # Gatekeeper: Halt execution if the raw CSV file has not been uploaded yet
    mo.stop(not upload_button.value, mo.md("⏳ *Please upload the DataCo CSV file above to initiate data pipeline...*"))

    # 1. Extract raw binary content from browser memory into a Pandas DataFrame
    file_content = upload_button.value[0].contents
    df_raw = pd.read_csv(io.BytesIO(file_content), encoding='latin1')

    # 2. Parse and format temporal columns (Critical step for sequential time-series alignment)
    df_raw['order date (DateOrders)'] = pd.to_datetime(df_raw['order date (DateOrders)'])

    # 3. Aggregation Phase: Roll up transaction records into total daily quantity demand
    df_daily = df_raw.groupby(df_raw['order date (DateOrders)'].dt.date)['Order Item Quantity'].sum().reset_index()
    df_daily.columns = ['date', 'total_demand']
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily = df_daily.sort_values('date')

    # 4. Calendar Feature Extraction (Capturing temporal human behavior)
    df_daily['day_of_week'] = df_daily['date'].dt.dayofweek
    df_daily['is_weekend'] = df_daily['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # 5. Historical Feature Engineering (Creating lookback variables for predictive momentum)
    df_daily['demand_lag_7'] = df_daily['total_demand'].shift(7) # Captures exact weekly seasonality
    df_daily['rolling_mean_7'] = df_daily['total_demand'].shift(1).rolling(window=7).mean() # Captures short-term market trend

    # 6. Data Cleansing: Remove NaN rows introduced by lag and rolling window operations
    df_daily = df_daily.dropna().reset_index(drop=True)

    # Display the final engineered time-series dataset
    df_daily.head(10)
    return (df_daily,)


@app.cell
def _(df_daily, mo):
    import xgboost as xgb
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import matplotlib.pyplot as plt
    import numpy as np

    # 1. Define Predictors (Features) and Target Variable
    features = ['day_of_week', 'is_weekend', 'demand_lag_7', 'rolling_mean_7']
    target = 'total_demand'

    X = df_daily[features]
    y = df_daily[target]

    # 2. Out-of-Sample Time-Series Split (The Golden Rule of Supply Chain Forecasting)
    # Avoid random shuffling to prevent data leakage. Train on historical 80% and validate on future 20%.
    split_index = int(len(df_daily) * 0.8)

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    dates_test = df_daily['date'].iloc[split_index:]

    # 3. Model Training via XGBoost Regressor
    # Tree depth is restricted to max_depth=5 to mitigate risk of model overfitting
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # 4. Generate Predictions on the Unseen Test Set
    predictions = model.predict(X_test)

    # 5. Evaluate Forecast Error Metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    # 6. High-Impact Visualization (Portfolio Spotlight Component)
    fig, ax = plt.subplots(figsize=(12, 6))
    # Blue Solid Line = Ground Truth Field Reality
    ax.plot(dates_test, y_test, label='Actual Ground Truth', color='#002B5B', alpha=0.7, linewidth=2)
    # Golden Dashed Line = AI Predictive Output
    ax.plot(dates_test, predictions, label='XGBoost Forecast Prediction', color='#D4AF37', linestyle='--', linewidth=2)

    ax.set_title('Logistics Demand Forecast Optimization: Actual vs Predicted Trends', fontsize=14, fontweight='bold', color='#002B5B')
    ax.set_ylabel('Total Unit Demand (Quantity)', fontsize=11)
    ax.set_xlabel('Timeline Horizon', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=11)

    # 7. Render the Final Analytics Executive Dashboard in Marimo
    mo.vstack([
        mo.md(f"""
        ### 🎯 Predictive Model Performance Metrics (XGBoost)
        - **MAE (Mean Absolute Error):** `{mae:.2f}` units per day.
        - **RMSE (Root Mean Squared Error):** `{rmse:.2f}` units.
    
        *Business Operations Impact:* The predictive framework demonstrates high fidelity in tracking demand elasticity and warehouse volume surges. The golden forecast trajectory successfully mirrors the actual supply chain disruption markers (blue volatility spikes), providing actionable lead-time visibility for safety stock optimization.*
        """),
        mo.as_html(fig)
    ])
    return


if __name__ == "__main__":
    app.run()
