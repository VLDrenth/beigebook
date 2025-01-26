"""
Beige Book Analysis Pipeline

This script runs a configurable pipeline for analyzing Federal Reserve Beige Book reports.
It supports multiple stages including data scraping, sentiment analysis, data cleaning,
and visualization generation.

Usage:
    python script.py [--scrape] [--sentiment] [--clean] [--analyze]

Arguments:
    --scrape        Fetch new Beige Book reports from the source
    --sentiment     Run sentiment analysis on the fetched reports
    --clean         Clean and transform the sentiment analysis results
    --analyze       Generate regional time series visualization plots

Examples:
    # Run complete pipeline
    python script.py --scrape --sentiment --clean --analyze
    
    # Only fetch new data
    python script.py --scrape
    
    # Process existing data without scraping
    python script.py --sentiment --clean --analyze
"""


import argparse
from pathlib import Path
from src.config.config import Config
from src.pipeline.processing import TextProcessor
from src.scraper.scraper import BeigeBookScraper
from src.score_analyzer.analyzer import ScoreAnalyzer
from src.data_processing.data_processing import ProcessData

def parse_arguments():
    """
    Parse and validate command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Beige Book Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--scrape",
        action="store_true",
        help="Fetch Beige Book data from source"
    )
    
    parser.add_argument(
        "--sentiment",
        action="store_true",
        help="Perform sentiment analysis on fetched data"
    )
    
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean and transform sentiment analysis results"
    )
    
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Generate regional time series analysis plots"
    )
    
    return parser.parse_args()

def main():
    """
    Execute the Beige Book analysis pipeline based on command line arguments.
    
    The pipeline consists of four optional stages:
    1. Data scraping
    2. Sentiment analysis
    3. Data cleaning
    4. Time series analysis
    """
    args = parse_arguments()
    
    if not any(vars(args).values()):
        print("No operations specified. Use --help for usage information.")
        return
    
    try:
        if args.scrape:
            scraper = BeigeBookScraper()
            scraper.fetch_all()
            
        if args.sentiment:
            processor = TextProcessor(token=Config.HF_TOKEN)
            results = processor.process_all(end_year=Config.END_YEAR)
            print(results)

        if args.clean:
            processor = ProcessData(
                file_path=Config.SENTIMENT_OUTPUT_DIR,
                save=True
            )
            processor.transform_data()

        if args.analyze:
            analyzer = ScoreAnalyzer(
                filepath=Config.SENTIMENT_OUTPUT_DIR,
                output_dir=Config.PLOTS_OUTPUT_DIR
            )
            analyzer.plot_regional_time_series()
            
    except Exception as e:
        print(f"Error in pipeline execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()