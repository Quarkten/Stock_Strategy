import os
import time
import yaml
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime
from src.data.database import DatabaseManager

class ForexFactoryScraper:
    def __init__(self, config):
        self.config = config
        self.db = DatabaseManager(config['db_path'])
        chrome_options = webdriver.ChromeOptions()
        if config.get('headless', True):
            chrome_options.add_argument("--headless")
        self.driver = webdriver.Chrome(options=chrome_options)
        
    def scrape_calendar(self):
        self.driver.get(self.config['forex_factory_url'])
        WebDriverWait(self.driver, 20).until(
            EC.presence_of_element_located((By.CLASS_NAME, "calendar__row"))
        )
        
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        rows = soup.find_all("tr", class_="calendar__row")
        
        for row in rows:
            # ... parsing logic from previous implementation
            self.db.log_economic_event(event_data)
        
        self.driver.quit()
        return True