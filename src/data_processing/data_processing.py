import pandas as pd

class ProcessData:
    def __init__(self, file_path, alpha: float = 0.05, save: bool = False):
        self.df = pd.read_parquet(file_path / "sentiment_scores.parquet")
        self.df.set_index("date", drop=True, inplace=True)
        self.df.index = pd.to_datetime(self.df.index)
        
        self.save = save
        self.alpha = alpha/2
        self.file_path = file_path / "sentiment_scores_clean.parquet"

    def transform_data(self):
        """
        Transforms dataframe to a monthly frequency, and imputes missing values
        using forward fill followed by backward fill to handle missing starting values.
        
        Returns:
            pd.DataFrame: Transformed dataframe with monthly frequency and imputed values
        """
        # Ignore neutral sentiments
        self.df = self.df[self.df["label"] != "neutral"]

        # Create normalized scores
        self.df['normalized_score'] = self.df.apply(
            lambda x: x['score'] * len(x['text']) if x["label"] == "positive" else -x['score'] * len(x['text']), 
            axis=1
        )

        # Calculate quantiles for filtering
        lower_quantile = self.df['normalized_score'].quantile(self.alpha)
        upper_quantile = self.df['normalized_score'].quantile(1 - self.alpha)

        # Filter out extreme values
        self.df = self.df[
            (self.df['normalized_score'] >= lower_quantile) & 
            (self.df['normalized_score'] <= upper_quantile)
        ]

        # Calculate mean scores by region and date
        mean_scores = (self.df
                      .groupby(['date', 'region'])['normalized_score']
                      .mean()
                      .reset_index())
        
        # Pivot to get regions as columns
        df_pivot = mean_scores.pivot(index='date', columns='region', values='normalized_score')
        
        # Resample to monthly frequency
        df_monthly = df_pivot.resample('ME').mean()
        
        # Ensure all dates are set to end of month
        df_monthly.index = df_monthly.index + pd.offsets.MonthEnd(0)

        df_monthly["index"] = df_monthly.mean(axis=1)
        
        # Impute missing values
        df_imputed = (df_monthly
                     .ffill()  # Forward fill
                     .bfill()  # Backward fill for any remaining NAs at the start
                    )
        
        # Calculate z-scores using expanding window
        df_normalized = ((df_imputed - df_imputed.expanding(min_periods=24).mean()) 
                        / df_imputed.expanding(min_periods=24).std())
        
        df_normalized = df_normalized.ewm(alpha=0.7).mean()
        self.df = df_normalized

        if self.save:
            self.df.to_parquet(self.file_path)

        return self.df