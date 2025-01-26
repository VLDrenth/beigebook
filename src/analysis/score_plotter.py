import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

class ScorePlotter:
    def __init__(self, filepath: str, output_dir: str = 'output'):
        self.df = pd.read_parquet(filepath / "sentiment_scores_clean.parquet")
        self.output_dir = Path(output_dir)
        self._setup_output_directory()
    
    def _setup_output_directory(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_regional_time_series(self) -> None:
        plt.style.use('seaborn-v0_8-whitegrid')
        
        for region in self.df.columns:
            region_data = self.df[region].sort_index()
            
            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
            
            ax.plot(region_data.index, region_data.values, 
                    color='#2E5A88', linewidth=1.5, label='Sentiment Score')
            
            # Formatting
            ax.set_xlabel('Date', fontsize=11, fontweight='bold')
            ax.set_ylabel('Sentiment Score', fontsize=11, fontweight='bold')
            ax.set_title(f'Regional Sentiment Analysis: {region.title()}', 
                        fontsize=12, fontweight='bold', pad=15)
            
            # Clean up ticks
            ax.tick_params(axis='both', labelsize=10)
            plt.xticks(rotation=45, ha='right')
            
            # Adjust grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Legend
            ax.legend(frameon=True, fancybox=True, framealpha=0.95, fontsize=10)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'time_series_{region}.png')
            plt.close()