"""
Beige Book Scraper - A tool to fetch Federal Reserve Beige Book reports.

This module provides functionality to scrape Beige Book reports from the Federal Reserve's
Minneapolis website and related economic data from FRED.
"""

import re
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Iterator

import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BeigeBookReport:
    """Container for Beige Book report data."""
    year: int
    month: int
    region: str
    text: Optional[str] = None
    date: Optional[datetime] = None

class BeigeBookScraper:
    """Main scraper class for Federal Reserve Beige Book reports."""

    # Beige Book regions
    REGIONS = [
        'at',  # Atlanta
        'bo',  # Boston
        'ch',  # Chicago
        'cl',  # Cleveland
        'da',  # Dallas
        'kc',  # Kansas City
        'mi',  # Minneapolis
        'ny',  # New York
        'ph',  # Philadelphia
        'ri',  # Richmond
        'sf',  # San Francisco
        'sl',  # St. Louis
        'su'   # National Summary
    ]

    # The Beige Book is published eight times per year
    RELEASE_MONTHS = [1, 3, 4, 6, 7, 9, 10, 12]

    BASE_URL = "https://www.minneapolisfed.org/beige-book-reports"
    OUTPUT_DIR = Path("out")

    def __init__(self):
        """Initialize scraper and ensure output directories exist."""
        self.session = requests.Session()
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Create necessary output directories if they don't exist."""
        (self.OUTPUT_DIR / "csv").mkdir(parents=True, exist_ok=True)
        (self.OUTPUT_DIR / "txt").mkdir(parents=True, exist_ok=True)

    def _make_url(self, report: BeigeBookReport) -> str:
        """Generate URL for a specific report."""
        return f"{self.BASE_URL}/{report.year}/{report.year}-{report.month:02d}-{report.region}"

    def _fetch_page(self, url: str) -> BeautifulSoup:
        """Fetch and parse a page, handling common errors."""
        response = self.session.get(url)
        if response.status_code == 404:
            raise ValueError(f"Page not found: {url}")
        response.raise_for_status()
        return BeautifulSoup(response.text, features="html5lib")

    def _extract_text(self, soup: BeautifulSoup) -> Tuple[str, Optional[str]]:
        """Extract main text and date from BeautifulSoup object."""
        div = soup.find("div", class_="col-sm-12 col-lg-8 offset-lg-1")
        if not div:
            raise ValueError("Could not find main content div")

        content = re.sub(r"\s*\n\s*", "\n", div.text).strip()
        parts = content.split("\n", 3)

        date_str = parts[2] if len(parts) > 2 else None
        text = parts[3] if len(parts) > 3 else None

        return text, date_str

    def fetch_report(self, report: BeigeBookReport) -> BeigeBookReport:
        """Fetch a single report and update the BeigeBookReport object."""
        url = self._make_url(report)
        try:
            soup = self._fetch_page(url)
            text, date_str = self._extract_text(soup)
            report.text = text
            # Parse date if available
            if date_str:
                # Add date parsing logic here if needed
                pass
            return report
        except Exception as e:
            logger.error(f"Error fetching report {report}: {str(e)}")
            raise

    def save_report(self, report: BeigeBookReport) -> None:
        """Save report text to file."""
        if not report.text:
            raise ValueError("Report has no text content")

        output_path = (self.OUTPUT_DIR / "txt" / str(report.year) /
                      f"{report.month:02d}" / f"{report.year}-{report.month:02d}-{report.region}.txt")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report.text)

    def fetch_latest(self) -> None:
        """Fetch the most recent Beige Book report."""
        # Get current date
        now = datetime.now()
        # Beige Book is released 8 times per year
        # You might want to adjust this logic based on the actual release schedule
        report = BeigeBookReport(
            year=now.year,
            month=now.month,
            region="su"  # Summary report
        )

        try:
            report = self.fetch_report(report)
            self.save_report(report)
            logger.info(f"Successfully fetched and saved latest report: {report}")
        except Exception as e:
            logger.error(f"Failed to fetch latest report: {str(e)}")

def _get_valid_dates(self, start_date: Optional[datetime] = None, 
                      end_date: Optional[datetime] = None) -> Iterator[Tuple[int, int]]:
    """
    Generate valid year-month combinations for Beige Book reports between start_date and end_date.
    
    Args:
        start_date: Optional starting date. If None, uses earliest available date
        end_date: Optional end date. If None, uses current date
        
    Yields:
        Tuples of (year, month) for valid Beige Book release dates
    """
    if end_date is None:
        end_date = datetime.now()
    
    if start_date is None:
        # You might want to adjust this based on actual earliest available date
        start_date = datetime(2000, 1, 1)
    
    current_year = start_date.year
    while current_year <= end_date.year:
        for month in self.RELEASE_MONTHS:
            current_date = datetime(current_year, month, 1)
            if start_date <= current_date <= end_date:
                yield (current_year, month)
        current_year += 1

def fetch_all(self, start_date: Optional[datetime] = None, 
              end_date: Optional[datetime] = None,
              regions: Optional[list[str]] = None) -> list[BeigeBookReport]:
    """
    Fetch all Beige Book reports between start_date and end_date for specified regions.
    
    Args:
        start_date: Optional starting date. If None, uses earliest available date
        end_date: Optional end date. If None, uses current date
        regions: Optional list of region codes. If None, fetches all regions
        
    Returns:
        List of BeigeBookReport objects containing the fetched reports
    """
    if regions is None:
        regions = self.REGIONS
    
    reports = []
    
    for year, month in self._get_valid_dates(start_date, end_date):
        for region in regions:
            try:
                report = BeigeBookReport(year=year, month=month, region=region)
                fetched_report = self.fetch_report(report)
                self.save_report(fetched_report)
                reports.append(fetched_report)
                logger.info(f"Successfully fetched report: {year}-{month:02d}-{region}")
            except Exception as e:
                logger.error(f"Failed to fetch report {year}-{month:02d}-{region}: {str(e)}")
                continue
    
    return reports

def main():
    """Main entry point for the scraper."""
    scraper = BeigeBookScraper()

    # Fetch latest Beige Book report
    scraper.fetch_latest()

if __name__ == "__main__":
    main()