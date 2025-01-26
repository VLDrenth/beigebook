import pandas as pd
from src.config.config import Config
from sklearn.linear_model import LinearRegression

def prepare_data():
    """
    Prepares a DataFrame containing GDP predictions and actuals across multiple forecast
    horizons, using a linear regression model trained on historical data.
    
    Returns:
        pd.DataFrame: Contains columns for forecast_date, horizon, predicted_value,
                     actual_value, and is_test
    """
    
    df_sentiment = pd.read_parquet(Config.SENTIMENT_OUTPUT_DIR / "sentiment_scores_clean.parquet")
    print(df_sentiment)
    gdp = pd.read_csv(Config.EXTERNAL_PATH /  "gdp.csv", 
                         index_col='observation_date', 
                         parse_dates=['observation_date'])
    
    # Adjust GDP dates to end of quarter
    gdp.index = gdp.index + pd.DateOffset(months=3) + pd.offsets.MonthEnd()
    
    results = []
    
    for forecast_date in pd.date_range(start="2010-01-31",
                                       end="2023-01-31",
                                       freq="M"):
        for horizon in Config.HORIZONS:
            target_date = forecast_date + pd.DateOffset(months=horizon)
            
            if target_date not in gdp.index:
                continue

            # Get available dates up to forecast_date
            available_dates = gdp.index[gdp.index < forecast_date]
            train_dates = [date for date in (available_dates - pd.DateOffset(months=horizon)) if date in df_sentiment.index]

            # Create train set
            X_train = df_sentiment.loc[train_dates].values
            y_test = gdp.loc[target_date].values
            y_train = gdp.loc[available_dates].values

            # Prepare test data (reshape for single sample prediction)
            X_test = df_sentiment.loc[[forecast_date]]

            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Make prediction
            y_pred = model.predict(X_test)[0]  # Extract scalar from array
            
            results.append({
                'forecast_date': forecast_date,
                'horizon': horizon,
                'predicted_value': y_pred,
                'actual_value': y_test
            })
                    
    return pd.DataFrame(results)

if __name__ == "__main__":
    print(prepare_data())