from typing import Dict, List
import pandas as pd
import os
import logging
import re
from multiprocessing import Pool, cpu_count
from pathlib import Path
from functools import partial
from tqdm import tqdm

from src.config.config import Config
from src.sentiment.sentiment_scoring import SentimentScorer

logger = logging.getLogger("processor_logger")

class TextProcessor:
    """
    Process text files to extract sentiment scores using multiprocessing.
    
    This class handles the parallel processing of text files organized by year/month/region,
    calculating sentiment scores and aggregating results into a structured format.
    """

    def __init__(self, num_processes: int = None):
        """
        Initialize the TextProcessor with configurable number of processes.
        
        Args:
            num_processes (int, optional): Number of processes to use. 
                                        Defaults to CPU count - 1 if not specified.
        """
        self.output_path = Path(Config.SENTIMENT_OUTPUT_DIR)
        self.input_path = Path(Config.SCRAPED_TEXT_DIR)
        self.num_processes = num_processes or max(1, cpu_count() - 1)
        self.sentiment_scorer = SentimentScorer()

    def process_all(self, end_year: int) -> pd.DataFrame:
        """
        Process all texts using multiprocessing to find sentiment scores.
        
        Returns:
            pd.DataFrame: DataFrame containing processed results with date, score, and region.
        """
        logger.info("Starting the processing of sentiment scoring")
        # Generate jobs for parallel processing
        jobs = [
            {"year": year, "month": month, "base_path": self.input_path}
            for year in os.listdir(self.input_path)
            for month in os.listdir(self.input_path / year) 
            if int(year) <= end_year
        ]
        
        # Create a process pool and map jobs
        with Pool(processes=self.num_processes) as pool:
            # Use tqdm to show progress
            results = list(tqdm(
                pool.imap(self.process_date, jobs),
                total=len(jobs),
                desc="Processing files"
            ))
        
        # Flatten results and convert to DataFrame
        flat_results = [item for sublist in results for item in sublist]
        df = pd.DataFrame(flat_results)
        
        # Save results
        self._save_results(df)
        
        return df

    def process_date(self, job: Dict) -> List[Dict]:
        """
        Process all regions for a specific year/month combination.
        
        Args:
            job (Dict): Dictionary containing year, month, and base_path information.
            
        Returns:
            List[Dict]: List of dictionaries containing processed results.
        """
        year = job["year"]
        month = job["month"]
        base_path = job["base_path"]
        
        try:
            region_paths = list((base_path / year / month).glob("*.txt"))
            output = []

            for region_path in region_paths:
                try:
                    date = self.extract_date(region_path.name)
                    region = self.extract_identifier(region_path.name)
                    logger.info(f"Processing Region {region} at date {date}")
                    score = self.sentiment_score(region_path)
                    
                    output.append({
                        "date": date,
                        "score": score,
                        "region": region,
                        "year": year,
                        "month": month
                    })
                except Exception as e:
                    logger.warning(f"Failed to process {region_path}: {str(e)}")
                    continue
            logger.info(f"Succesfully processed region {region} at date {date}")
            return output
            
        except Exception as e:
            logger.error(f"Failed to process {year}/{month}: {str(e)}")
            return []

    def sentiment_score(self, region_path: Path) -> float:
        """
        Calculate sentiment score for text in the given path.
        
        Args:
            region_path (Path): Path to the text file.
            
        Returns:
            float: Calculated sentiment score.
            
        Raises:
            Exception: If file cannot be read or processed.
        """
        try:
            text = region_path.read_text(encoding='utf-8')
            score = self.sentiment_scorer.score(text=text)
            return score
        except Exception as e:
            logger.warning(f"Could not process {region_path}")
            raise e

    def _save_results(self, df: pd.DataFrame) -> None:
        """
        Save results in both parquet and CSV formats.
        
        Args:
            df (pd.DataFrame): DataFrame containing processed results.
        """
        # Ensure output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as parquet
        df.to_parquet(self.output_path / "sentiment_scores.parquet")
        # Save as CSV for easier viewing
        df.to_csv(self.output_path / "sentiment_scores.csv", index=False)

        logger.info("Succesfully saved results")

    @staticmethod
    def extract_identifier(filename: str) -> str:
        """Extract region identifier from filename."""
        match = re.search(r'\d{4}-\d{2}-\d{2}-(.+?)\.', filename)
        if not match:
            raise ValueError(f"Could not extract identifier from {filename}")
        return match.group(1)

    @staticmethod
    def extract_date(filename: str) -> str:
        """Extract date from filename."""
        match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
        if not match:
            raise ValueError(f"Could not extract date from {filename}")
        return match.group(1)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize and run processor
    processor = TextProcessor()
    results_df = processor.process_all(end_year=2002)
    logger.info(f"Processed {len(results_df)} files successfully")