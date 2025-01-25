from pathlib import Path
from src.config.config import Config
from src.pipeline.processing import TextProcessor
from src.scraper.scraper import BeigeBookScraper
from src.score_analyzer.analyzer import ScoreAnalyzer
from src.data_processing.data_processing import ProcessData

def main(scrape: bool = False, sentiment: bool = False, clean: bool = False, analyze: bool = False):
    """
    Function that uses the modules to scrape the data and apply sentiment analysis.
    """

    if scrape:
        BeigeBookScraper().fetch_all()
        
    if sentiment:
        processor = TextProcessor(hf_token=Config.HF_TOKEN)
        print(processor.process_all(end_year=Config.END_YEAR))

    if clean:
        ProcessData(file_path=Config.SENTIMENT_OUTPUT_DIR, save=True).transform_data()

    if analyze:
        analyzer = ScoreAnalyzer(filepath=Config.SENTIMENT_OUTPUT_DIR, output_dir=Config.PLOTS_OUTPUT_DIR)
        analyzer.plot_regional_time_series()

if __name__ == "__main__":
    main(analyze=True)