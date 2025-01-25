import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List

class ScoreAnalyzer:
    def __init__(self, filepath: str, output_dir: str = 'output'):
        self.df = pd.read_csv(filepath / "sentiment_scores.csv", parse_dates=['date'])
        self._validate_data()
        self.output_dir = Path(output_dir)
        self._setup_output_directory()
    
    def _setup_output_directory(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_regional_time_series(self) -> None:
        regions = self.df['region'].unique()
        
        for region in regions:
            region_data = self.df[self.df['region'] == region].copy()
            region_data = region_data.sort_values('date')
            
            plt.figure(figsize=(12, 6))
            
            # Plot raw data
            plt.plot(region_data['date'], region_data['score'], 
                    color='blue', alpha=0.5, label='Raw Data')
                        
            plt.title(f'Score Trend Over Time - {region.upper()}')
            plt.xlabel('Date')
            plt.ylabel('Score')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            
            output_path = self.output_dir / f'time_series_{region}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

    def _validate_data(self) -> None:
        """Validates that all required columns are present in the dataset."""
        required_columns = {'date', 'score', 'region', 'year', 'month'}
        if not required_columns.issubset(self.df.columns):
            missing = required_columns - set(self.df.columns)
            raise ValueError(f"Missing required columns: {missing}")
