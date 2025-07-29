import requests
import os
import json
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self, config):
        self.alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_KEY") or config.get('alpha_vantage_api_key')
        self.fred_api_key = os.getenv("FRED_API_KEY") or config.get('fred_api_key')
        self.config = config

    def fetch_alpha_vantage_news(self, symbols=None, topics=None, time_from=None, time_to=None):
        if not self.alpha_vantage_api_key:
            logger.warning("ALPHA_VANTAGE_KEY not set. Skipping Alpha Vantage news fetch.")
            return ""

        base_url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "apikey": self.alpha_vantage_api_key,
            "sort": "RELEVANCE",
            "limit": 50
        }
        if symbols:
            params["symbols"] = ",".join(symbols)
        if topics:
            params["topics"] = ",".join(topics)
        if time_from:
            params["time_from"] = time_from
        if time_to:
            params["time_to"] = time_to

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            articles = [
                f"Title: {item['title']}\nSummary: {item['summary']}"
                for item in data.get('feed', [])
            ]
            return data.get('feed', [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Alpha Vantage news: {e}")
            return ""

    def fetch_fred_data(self, series_id):
        if not self.fred_api_key:
            logger.warning("FRED_API_KEY not set. Skipping FRED data fetch.")
            return ""

        base_url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self.fred_api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 1
        }
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            observations = data.get('observations', [])
            if observations:
                latest = observations[0]
                return f"{series_id}: Date={latest['date']}, Value={latest['value']}"
            return ""
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching FRED data: {e}")
            return ""
