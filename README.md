# beigebook
Scraper and analyzer for the Federal Reserve's Beige book publications

## Project Structure

- pipeline: Contains module to take the scraped text, apply sentiment scoring, and produce a time-series parquet file
- scraper: Contains the scraper module used to derive the raw text files
- data_processing: Contains the methods used to clean and process the data. Too simple at the current stage to merit its own module, but will expand soon.
- sentiment: Contains a wrapper for Language models to provide a sentiment score
- score_analyzer: Code used to analyze and produce plots for the scraped sentiment data
- config: Contains some basic setup concerning paths and some parameters for running the programs