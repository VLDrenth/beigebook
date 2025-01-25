import pandas as pd

class ProcessData:

    def __init__(self, file_path):
        df = pd.read_parquet(file_path)
        df.set_index("date", drop=True, inplace=True)
        df.index = pd.to_datetime(df.index)
        self.df = df

def transform_data(self):
    """
    Transforms dataframe to a monthly frequency, and imputes missing values
    using forward fill followed by backward fill to handle edge cases.
    
    Returns:
        pd.DataFrame: Transformed dataframe with monthly frequency and imputed values
    """
    # Resample to monthly frequency taking the mean for numeric columns
    monthly_df = self.df.resample('M').last()
        
    # Ensure all dates are set to end of month
    monthly_df.index = monthly_df.index + pd.offsets.MonthEnd(0)

    # Identify numeric columns for appropriate imputation
    numeric_cols = monthly_df.select_dtypes(include=['float64', 'int64']).columns
    
    # Impute missing values for numeric columns
    monthly_df[numeric_cols] = (
        monthly_df[numeric_cols]
        .fillna(method='ffill')  # Forward fill
        .fillna(method='bfill')  # Backward fill for any remaining NAs at the start
    )
    
    # Store transformed data
    self.df = monthly_df
    
    return self.df



