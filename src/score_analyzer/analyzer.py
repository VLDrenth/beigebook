import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List

class ScoreAnalyzer:
    def __init__(self, filepath: str, output_dir: str = 'output'):
        self.df = pd.read_parquet(filepath / "sentiment_scores_clean.parquet")
        self.output_dir = Path(output_dir)
        self._setup_output_directory()
    
    def _setup_output_directory(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_regional_time_series(self) -> None:
        regions = self.df.columns
        
        for region in regions:
            region_data = self.df[region]
            region_data = region_data.sort_index()
            
            plt.figure(figsize=(12, 6))
            
            # Plot raw data
            plt.plot(region_data.index, region_data.values, 
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