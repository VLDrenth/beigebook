import pandas as pd
import numpy as np

from tqdm import tqdm
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
    df_sentiment = df_sentiment[["index"]].diff().bfill().ewm(alpha=0.8).mean()
    gdp = pd.read_csv(Config.EXTERNAL_PATH / "gdp.csv", 
                         index_col='observation_date', 
                         parse_dates=['observation_date'])
    
    # Adjust GDP dates to end of quarter
    gdp.index = gdp.index + pd.DateOffset(months=3) + pd.offsets.MonthEnd()
    gdp = ((1 + np.log(gdp).diff())**4 -1) * 100
    gdp_col = "GDPC1"

    df_aligned = pd.concat([df_sentiment, gdp], axis=1)
    df_aligned = df_aligned[df_aligned.index >= "2000-01-31"]

    results = []
    
    for forecast_date in tqdm(pd.date_range(start="2019-01-31",
                                       end="2023-01-31",
                                       freq="M"), desc="forecast date"):
        for horizon in tqdm(Config.HORIZONS, desc="Forecast Horizon", total=len(Config.HORIZONS)):
            target_date = forecast_date + pd.DateOffset(months=horizon)
            
            if target_date not in gdp.index:
                continue
            temp_df = df_aligned.copy()

            y = temp_df[gdp_col].shift(-horizon)
            temp_df.drop(columns=gdp_col, inplace=True)

            X_train = temp_df.loc[temp_df.index < forecast_date]
            y_train = y.loc[temp_df.index < forecast_date]

            valid_indices = ~y_train.isna()
            X_train = X_train[valid_indices]
            y_train = y_train[valid_indices]

            X_test = pd.DataFrame(temp_df.loc[forecast_date, :].values.reshape(1, -1), columns=X_train.columns)
            y_test = y.loc[forecast_date]

            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Make prediction
            y_pred = model.predict(X_test)
            
            results.append({
                'forecast_date': forecast_date,
                'horizon': horizon,
                'predicted_value': y_pred,
                'actual_value': y_test
            })
                    
    res = pd.DataFrame(results)
    return res

def get_performance():
    """
    Compute performance metrics over the backtest for each forecasting horizon.
    """
    metrics_list = []
    
    def _calculate_metrics(group: pd.DataFrame) -> dict:
        """
        Calculate performance metrics for a group of predictions
        """
        errors = (group['actual_value'] - group['predicted_value']).values
        n = len(errors)
        
        if n == 0:
            return {
                'horizon': group.name,
                'rmse': np.nan,
                'mae': np.nan,
                'r2': np.nan,
                'mape': np.nan,
                'count': 0
            }
            
        rmse = np.sqrt(np.mean(errors ** 2)).item()
        mae = np.mean(np.abs(errors)).item()
        
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((group['actual_value'] - group['actual_value'].mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot).item() if ss_tot != 0 else np.nan
        
        non_zero_mask = group['actual_value'] != 0
        mape = np.mean(np.abs(errors[non_zero_mask] / group['actual_value'][non_zero_mask])).item() * 100 if any(non_zero_mask) else np.nan
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'count': n
        }
    
    try:
        results = pd.read_parquet("results.parquet")
        
        # Process each group and collect results
        for horizon, group in results.groupby("horizon")[["predicted_value", "actual_value"]]:
            metrics = _calculate_metrics(group)
            metrics["horizon"] = horizon
            metrics_list.append(metrics)
        
        # Create DataFrame from list of dictionaries
        performance_df = pd.DataFrame(metrics_list)
        performance_df.set_index('horizon', inplace=True)

        return performance_df
        
    except Exception as e:
        raise RuntimeError(f"Error calculating performance metrics: {str(e)}")
    
if __name__ == "__main__":
    prepare_data()
    print(get_performance())