import os
import time
import random
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import undetected_chromedriver as uc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ScrapingConfig:
    """Configuration class for scraper settings"""
    base_url: str = "https://www.forexfactory.com/calendar"
    output_dir: str = "output"
    max_retries: int = 3
    page_load_timeout: int = 30
    element_timeout: int = 15
    respect_robots: bool = True
    rate_limit_delay: Tuple[float, float] = (2.0, 5.0)
    user_data_dir: Optional[str] = None
    profile_directory: Optional[str] = None
    # CSS selectors for better maintainability
    table_selector: str = ".calendar__table"
    row_selector: str = "tr"
    cell_selector: str = "td"

class ForexCalendarScraper:
    """Enhanced Forex Factory Calendar Scraper with better practices"""
    FIELD_MAPPINGS = {
        "calendar__date": "date",
        "calendar__time": "time",
        "calendar__currency": "currency",
        "calendar__impact": "impact",
        "calendar__event": "event",
        "calendar__actual": "actual",
        "calendar__forecast": "forecast",
        "calendar__previous": "previous",
    }
    IMPACT_COLORS = {
        "icon--ff-impact-yel": "yellow",
        "icon--ff-impact-ora": "orange",
        "icon--ff-impact-red": "red",
        "icon--ff-impact-gra": "gray",
        "icon--ff-impact-hol": "holiday",
    }
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15"
    ]

    def __init__(self, config: ScrapingConfig = None):
        self.config = config or ScrapingConfig()
        self.session = requests.Session()
        self.driver = None

    def check_robots_txt(self) -> bool:
        """Check robots.txt for scraping permissions"""
        if not self.config.respect_robots:
            return True
        try:
            robots_url = "https://www.forexfactory.com/robots.txt"
            response = self.session.get(robots_url, timeout=10)
            # Simple check - in production, use robotparser module
            robots_content = response.text.lower()
            if "disallow: /calendar" in robots_content:
                logger.warning("Calendar scraping may be disallowed by robots.txt")
                return False
        except Exception as e:
            logger.warning(f"Could not check robots.txt: {e}")
        return True

    def get_random_user_agent(self) -> str:
        """Get a random modern user agent"""
        return random.choice(self.USER_AGENTS)

    def setup_chrome_options(self, user_data_dir: Optional[str] = None, profile_directory: Optional[str] = None) -> Options:
        """Configure Chrome options for better stability"""
        options = Options()
        # Essential options
        options.add_argument(f"--user-agent={self.get_random_user_agent()}")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-extensions")
        # Performance options
        options.add_argument("--memory-pressure-off")
        options.add_argument("--max_old_space_size=4096")
        # Privacy options
        
        
        # Set window size for consistency
        options.add_argument("--window-size=1920,1080")
        if user_data_dir:
            options.add_argument(f"--user-data-dir={user_data_dir}")
        if profile_directory:
            options.add_argument(f"--profile-directory={profile_directory}")
        return options

    @contextmanager
    def get_driver(self):
        """Context manager for driver lifecycle"""
        driver = None
        try:
            options = uc.ChromeOptions()
            options.add_argument(f'--user-data-dir={self.config.user_data_dir}')
            options.add_argument(f'--profile-directory={self.config.profile_directory}')
            driver = uc.Chrome(options=options, use_subprocess=True)
            logger.info("Using undetected_chromedriver.")
            # Set timeouts
            driver.set_page_load_timeout(self.config.page_load_timeout)
            driver.implicitly_wait(5)
            # Remove webdriver property
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            yield driver
        except Exception as e:
            logger.error(f"Driver initialization failed: {e}")
            raise
        finally:
            if driver:
                try:
                    time.sleep(1) # Add a small delay for graceful shutdown
                    driver.quit()
                    if driver.service.process:
                        driver.service.stop()
                except Exception as e:
                    logger.error(f"Error closing driver: {e}")

    def human_like_delay(self):
        """Add human-like delays between actions"""
        delay = random.uniform(*self.config.rate_limit_delay)
        time.sleep(delay)

    def scroll_page_naturally(self, driver):
        """Scroll page in a natural way"""
        try:
            # Get page height
            total_height = driver.execute_script("return document.body.scrollHeight")
            viewport_height = driver.execute_script("return window.innerHeight")
            current_position = 0
            scroll_step = random.randint(300, 500)

            while current_position < total_height - viewport_height:
                # Scroll with smooth behavior
                driver.execute_script(f"""
                    window.scrollTo({{
                        top: {current_position + scroll_step},
                        behavior: 'smooth'
                    }});
                """)
                current_position += scroll_step
                time.sleep(random.uniform(0.5, 1.5))

                # Random small scroll variations
                if random.random() < 0.3:
                    mini_scroll = random.randint(-50, 50)
                    driver.execute_script(f"window.scrollBy(0, {mini_scroll})")
                    time.sleep(0.2)

        except Exception as e:
            logger.warning(f"Scrolling error: {e}")

    def extract_impact_level(self, element) -> str:
        """Extract impact level from element"""
        try:
            # Look for impact icons
            for icon_class, color in self.IMPACT_COLORS.items():
                if element.find_elements(By.CSS_SELECTOR, f".{icon_class}"):
                    return color
            # Check class names directly
            class_names = element.get_attribute("class") or ""
            for icon_class, color in self.IMPACT_COLORS.items():
                if icon_class in class_names:
                    return color
            return "unknown"
        except Exception as e:
            logger.debug(f"Error extracting impact level: {e}")
            return "unknown"

    def parse_table_row(self, row) -> Dict[str, str]:
        """Parse a single table row into structured data"""
        row_data = {}
        try:
            cells = row.find_elements(By.TAG_NAME, "td")
            for cell in cells:
                class_names = cell.get_attribute("class") or ""
                # Find matching field type
                field_type = None
                for css_class, field_name in self.FIELD_MAPPINGS.items():
                    if css_class in class_names:
                        field_type = field_name
                        break
                if not field_type:
                    continue

                # Extract cell content
                if field_type == "impact":
                    row_data[field_type] = self.extract_impact_level(cell)
                else:
                    text_content = cell.text.strip()
                    row_data[field_type] = text_content

        except Exception as e:
            logger.debug(f"Error parsing row: {e}")
        return row_data

    def scrape_calendar_data(self, driver, month: str, year: str) -> List[Dict[str, str]]:
        """Scrape calendar data from the page"""
        data = []
        try:
            # Wait for calendar table to load
            wait = WebDriverWait(driver, self.config.element_timeout)
            table = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, self.config.table_selector))
            )
            logger.info("Calendar table loaded successfully")

            # Natural page interaction
            self.scroll_page_naturally(driver)
            self.human_like_delay()

            # Parse table rows
            rows = table.find_elements(By.CSS_SELECTOR, self.config.row_selector)
            logger.info(f"Found {len(rows)} table rows")

            current_date = None  # Track current date for rows without dates
            for i, row in enumerate(rows):
                try:
                    row_data = self.parse_table_row(row)
                    if not row_data:
                        continue

                    # Handle date continuity (some rows don't repeat the date)
                    if row_data.get("date"):
                        current_date = row_data["date"]
                    elif current_date and "event" in row_data:
                        row_data["date"] = current_date

                    # Add metadata
                    row_data["scraped_month"] = month
                    row_data["scraped_year"] = year
                    row_data["row_index"] = i
                    data.append(row_data)

                    # Small delay between rows
                    if i % 10 == 0:
                        time.sleep(random.uniform(0.1, 0.3))

                except Exception as e:
                    logger.debug(f"Error processing row {i}: {e}")
                    continue

            logger.info(f"Successfully parsed {len(data)} data rows")

        except TimeoutException:
            logger.error("Timeout waiting for calendar table to load")
        except Exception as e:
            logger.error(f"Error scraping calendar data: {e}")
        return data

    def save_data(self, data: List[Dict[str, str]], month: str, year: str) -> str:
        """Save scraped data to CSV file"""
        if not data:
            logger.warning("No data to save")
            return ""
        try:
            # Create output directory
            os.makedirs(self.config.output_dir, exist_ok=True)

            # Create DataFrame and clean data
            df = pd.DataFrame(data)
            # Basic data cleaning
            df = df.dropna(how='all')  # Remove completely empty rows
            df = df.drop_duplicates()  # Remove exact duplicates

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config.output_dir}/forex_calendar_{month}_{year}_{timestamp}.csv"

            # Save to CSV
            df.to_csv(filename, index=False, encoding='utf-8')
            logger.info(f"Data saved to {filename} ({len(df)} records)")
            return filename
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            return ""

    def calculate_target_month(self, month_name: Optional[str] = None) -> Tuple[str, str]:
        """Calculate target month and year"""
        now = datetime.now()
        if not month_name:
            return now.strftime("%B"), str(now.year)
        try:
            # Parse target month
            target_date = datetime.strptime(month_name.title(), "%B")
            target_month_num = target_date.month

            # Determine year (if month is in the past, assume next year)
            year = now.year
            if target_month_num < now.month:
                year += 1
            return month_name.title(), str(year)
        except ValueError:
            logger.error(f"Invalid month name: {month_name}")
            return now.strftime("%B"), str(now.year)

    def build_calendar_url(self, month: str, year: str, use_current: bool = False) -> str:
        """Build the calendar URL for the target month"""
        if use_current:
            month_param = "this"
        else:
            month_param = f"{month.lower()}.{year}"
        return f"{self.config.base_url}?month={month_param}"

    def scrape_with_retries(self, driver, url: str, month: str, year: str) -> List[Dict[str, str]]:
        """Scrape with retry logic"""
        last_exception = None
        for attempt in range(self.config.max_retries):
            try:
                logger.info(f"Scraping attempt {attempt + 1}/{self.config.max_retries}: {url}")
                # Navigate to URL
                driver.get(url)
                # Wait for page load
                self.human_like_delay()
                # Scrape data
                data = self.scrape_calendar_data(driver, month, year)
                if data:
                    return data
                else:
                    logger.warning(f"No data found on attempt {attempt + 1}")
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    delay = (attempt + 1) * 2
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
        if last_exception:
            raise last_exception
        else:
            raise Exception("All retry attempts failed with no data")

    def get_news(self, target_month: Optional[str] = None) -> List[Dict[str, str]]:
        """Scrapes calendar data and returns it as a list of dictionaries."""
        try:
            if not self.check_robots_txt():
                logger.error("Scraping may violate robots.txt. Aborting.")
                return []

            month, year = self.calculate_target_month(target_month)
            use_current = target_month is None

            url = self.build_calendar_url(month, year, use_current)
            logger.info(f"Target: {month} {year}")
            logger.info(f"URL: {url}")

            with self.get_driver() as driver:
                data = self.scrape_with_retries(driver, url, month, year)

            if data:
                logger.info(f"Scraping completed successfully. {len(data)} events found.")
                return data
            else:
                logger.error("No data was scraped")
                return []

        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            return []

    def main():
        config = ScrapingConfig()
        scraper = ForexCalendarScraper(config)
        news_data = scraper.get_news()
        print(json.dumps(news_data))

if __name__ == "_