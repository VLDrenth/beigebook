# BeigeBook: Federal Reserve Economic Sentiment Analysis

An automated tool for analyzing Federal Reserve Beige Book publications, providing systematic sentiment analysis of regional economic conditions across time.

## Overview

BeigeBook extracts, processes, and analyzes sentiment from Federal Reserve Beige Book publications. This uses NLP techniques to convert qualitative economic assessments into quantitative sentiment indicators, enabling systematic analysis of economic conditions across all Federal Reserve districts.

## Project Architecture

The project follows a has the following core functionalities into distinct components:

### Data Collection and Processing
- **Scraper Module**: Extracts raw text from Federal Reserve Beige Book publications
- **Data Processing Module**: Implements text cleaning and standardization procedures 
- **Pipeline Module**: Orchestrates the end-to-end workflow from text extraction to sentiment scoring

### Analysis Components
- **Sentiment Module**: Provides a wrapper for language models to generate sentiment scores
- **Score Analyzer**: Generates visualizations and analytical insights from processed sentiment data

### Configuration
- **Config Module**: Contains environment settings, file paths, and runtime parameters

## Technical Implementation

The system processes Beige Book publications through several stages:
1. Automated collection of Federal Reserve documents
2. Text preprocessing and standardization
3. Sentiment analysis using specialized language models
4. Time series generation and storage in Parquet format
5. Visualization and analysis of sentiment patterns over time

## Future Development

Planned enhancements include:
- Integration of additional fed/economic data to more fully capture sentiment
- Development of more sophisticated sentiment analysis techniques, like modernBERT

## Installation and Usage

[Coming Soon].