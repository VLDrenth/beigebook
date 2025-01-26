import pandas as pd
import os
import logging
import re
from multiprocessing import Pool, cpu_count
from pathlib import Path
from tqdm import tqdm
from typing import List

from src.config.config import Config
from src.sentiment.sentiment_scoring import HuggingFaceSentimentScorer, OpenAISentimentScorer, DocumentSentiment

logger = logging.getLogger("processor_logger")

class TextProcessor:
    """
    Process text files to extract sentiment scores using multiprocessing.
    
    This class handles the parallel processing of text files organized by year/month/region,
    calculating sentiment scores and aggregating results into a structured format.
    """

    def __init__(self, token: str = None, num_processes: int = 4):
        """
        Initialize the TextProcessor with configurable number of processes.
        
        Args:
            num_processes (int, optional): Number of processes to use. 
                                        Defaults to CPU count - 1 if not specified.
        """

        self.output_path = Path(Config.SENTIMENT_OUTPUT_DIR)
        self.input_path = Path(Config.SCRAPED_TEXT_DIR)
        self.num_processes = num_processes or max(1, cpu_count() - 1)
        self.sentiment_scorer = HuggingFaceSentimentScorer(token=token)

    def process_file(self, region_path: Path) -> List[dict]:
        try:
            text = region_path.read_text(encoding='utf-8')
            sentiment_scores = self.sentiment_scorer.score(text)
            date = self.extract_date(region_path.name)
            region = self.extract_identifier(region_path.name)
            year = region_path.parent.parent.name
            month = region_path.parent.name
            
            # Create a row for each chunk's sentiment scores
            return [
                {
                    "date": date,
                    "region": region,
                    "year": year,
                    "month": month,
                    "chunk_id": score.chunk_id,
                    "text": score.text,
                    "label": score.label,
                    "score": score.score
                }
                for score in sentiment_scores
            ]
        except Exception as e:
            logger.warning(f"Failed to process {region_path}: {str(e)}")
            return None

    def process_all(self, end_year: int) -> pd.DataFrame:
        logger.info("Starting the processing of sentiment scoring")
        jobs = []
        for year in os.listdir(self.input_path):
            if int(year) > end_year:
                continue
            year_path = self.input_path / year
            for month in os.listdir(year_path):
                month_path = year_path / month
                jobs.extend(list(month_path.glob("*.txt")))
        
        with Pool(processes=self.num_processes) as pool:
            results = list(tqdm(
                pool.imap(self.process_file, jobs, chunksize=100),
                total=len(jobs),
                desc="Processing files"
            ))

        logger.info("Finished processing files, preparing dataframe")
        
        # Flatten the list of lists into a single list of dictionaries
        flattened_results = [
            row for result in results 
            if result is not None 
            for row in result
        ]
        
        df = pd.DataFrame(flattened_results)
        self._save_results(df)
        return df
        
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