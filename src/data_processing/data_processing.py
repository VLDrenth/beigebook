import pandas as pd

class ProcessData:

    def __init__(self, file_path, save: bool = False):
        df = pd.read_parquet(file_path / "sentiment_scores.parquet")
        df.set_index("date", drop=True, inplace=True)
        df.index = pd.to_datetime(df.index)

        self.df = df
        self.save = save
        self.file_path = file_path / "sentiment_scores_clean.parquet"

    def transform_data(self):
        """
        Transforms dataframe to a monthly frequency, and imputes missing values
        using forward fill followed by backward fill to handle edge cases.
        
        Returns:
            pd.DataFrame: Transformed dataframe with monthly frequency and imputed values
        """
        df = self.df.pivot(columns='region', values='score')
        df["index"] = df.mean(1)

        # Resample to monthly frequency taking the mean for numeric columns
        df = df.resample('M').mean()
 
        # Ensure all dates are set to end of month
        df.index = df.index + pd.offsets.MonthEnd(0)
        
        # Impute missing values for numeric columns
        df = (
            df \
            .ffill()  # Forward fill
            .bfill()  # Backward fill for any remaining NAs at the start
        )

        df = (df - df.rolling(window=12).mean()) / df.rolling(window=12).std()

        # Store transformed data
        self.df = df

        if self.save:
            df.to_parquet(self.file_path)

        return self.df



