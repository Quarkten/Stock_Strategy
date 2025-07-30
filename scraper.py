import os
import time
import random
import argparse
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
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
    
    # CSS selectors for better maintainability
    table_selector: str = ".calendar__table"
    row_selector: str = "tr"
    cell_selector: str = "td"
    requires_login: bool = False # New flag to indicate if login is required
    login_url: str = "https://www.forexfactory.com/login" # Default login URL
    username_env_var: str = "FOREX_FACTORY_USERNAME"
    password_env_var: str = "FOREX_FACTORY_PASSWORD"
    username_field_id: str = "login_username"
    password_field_id: str = "login_password"
    login_button_id: str = "input.button.button--pressable"
    post_login_element_css: str = "meta[property='og:title'][content='Forums | Forex Factory']"
    news_url: str = "https://www.forexfactory.com/news"
    news_sections: dict = field(default_factory=lambda: {
        "latest_stories": ".news__section--latest-stories",
        "fundamental_analysis": ".news__section--fundamental-analysis",
        "breaking_news": ".news__section--breaking-news",
        "hottest_stories": ".news__section--hottest-stories",
        "breaking_news_most_viewed": "ul.body.flexposts"
    })

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

    

    

    

    def _get_chrome_options_for_temp_profile(self) -> Options:
        """Configures Chrome options for a temporary profile."""
        options = uc.ChromeOptions()
        # Essential options
        options.add_argument(f"--user-agent={self.get_random_user_agent()}")
        
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-extensions")
        # Performance options
        options.add_argument("--memory-pressure-off")
        options.add_argument("--max_old_space_size=4096")
        # Set window size for consistency
        options.add_argument("--window-size=1920,1080")
        return options

    @contextmanager
    def get_driver(self):
        """Context manager for driver lifecycle"""
        driver = None
        try:
            logger.info("Attempting to launch Chrome with a temporary profile...")
            time.sleep(1) # Add a small delay to allow Chrome to initialize
            options = self._get_chrome_options_for_temp_profile()
            driver = uc.Chrome(options=options, headless=False)
            logger.info("undetected_chromedriver launched Chrome with a temporary profile.")
            
            logger.info("Yielding driver to the scraping process.")
            yield driver

        except Exception as e:
            logger.error(f"Driver initialization failed: {e}")
            raise
        finally:
            if driver:
                logger.info("Attempting to quit Chrome driver.")
                try:
                    driver.quit()
                    logger.info("Chrome driver quit successfully.")
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
                driver.execute_script(f'''
                    window.scrollTo({{
                        top: {current_position + scroll_step},
                        behavior: 'smooth'
                    }});
                ''')
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

    def save_data_to_json(self, data: dict, filename: str) -> str:
        """Save scraped data to a JSON file."""
        if not data:
            logger.warning(f"No data to save to {filename}")
            return ""
        try:
            os.makedirs(self.config.output_dir, exist_ok=True)
            filepath = os.path.join(self.config.output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            logger.info(f"Data saved to {filepath} ({len(data)} records)")
            return filepath
        except Exception as e:
            logger.error(f"Error saving data to {filename}: {e}")
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

    def scrape_news_data(self, driver) -> Dict[str, List[Dict[str, str]]]:
        """Scrapes news data from the Forex Factory news page."""
        all_news_data = {}
        try:
            logger.info(f"Navigating to news URL: {self.config.news_url}")
            driver.get(self.config.news_url)
            self.human_like_delay()

            wait = WebDriverWait(driver, self.config.element_timeout)

            for section_name, section_selector in self.config.news_sections.items():
                logger.info(f"Attempting to scrape news section: {section_name} using selector: {section_selector}")
                try:
                    # Wait for the section to be present
                    section_element = wait.until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, section_selector))
                    )
                    logger.info(f"Found news section: {section_name}")

                    # Find individual news items within the section
                    news_items = section_element.find_elements(By.CSS_SELECTOR, ".flexposts__story, .flexposts__item")
                    logger.info(f"Found {len(news_items)} news items in section {section_name}")

                    section_news = []
                    for item in news_items:
                        try:
                            # Extract headline and link
                            headline_element = item.find_element(By.CSS_SELECTOR, ".flexposts__title a, .flexposts__story-title a")
                            headline = headline_element.text.strip()
                            link = headline_element.get_attribute("href")

                            # Extract date/time
                            news_time = ""
                            try:
                                time_element = item.find_element(By.CSS_SELECTOR, ".flexposts__time")
                                news_time = time_element.text.strip()
                            except NoSuchElementException:
                                timestamp = item.get_attribute("data-timestamp")
                                if timestamp:
                                    news_time = datetime.fromtimestamp(int(timestamp)).isoformat()

                            # Extract impact
                            impact = ""
                            try:
                                impact_element = item.find_element(By.CSS_SELECTOR, ".flexposts__storyimpact")
                                impact = " ".join(impact_element.get_attribute("class").split())
                            except NoSuchElementException:
                                pass

                            # Extract summary
                            summary = ""
                            try:
                                summary_element = item.find_element(By.CSS_SELECTOR, ".flexposts__preview .fader__original")
                                summary = summary_element.text.strip()
                            except NoSuchElementException:
                                pass

                            # Extract comments
                            comments = ""
                            try:
                                comments_element = item.find_element(By.CSS_SELECTOR, ".flexposts__caption a")
                                comments = comments_element.text.strip()
                            except NoSuchElementException:
                                pass


                            # You might need to extract more details like summary, author, etc.
                            section_news.append({
                                "section": section_name,
                                "headline": headline,
                                "link": link,
                                "time": news_time,
                                "impact": impact,
                                "summary": summary,
                                "comments": comments
                                # Add more fields as needed
                            })
                        except NoSuchElementException:
                            logger.warning("Could not find expected elements within a news item. Skipping.")
                            continue
                        except Exception as e:
                            logger.warning(f"Error processing news item: {e}")
                            continue
                    all_news_data[section_name] = section_news

                except TimeoutException:
                    logger.warning(f"Timeout waiting for news section: {section_name}. Skipping this section.")
                except NoSuchElementException:
                    logger.warning(f"Could not find news section: {section_name} using selector {section_selector}. Skipping.")
                except Exception as e:
                    logger.error(f"Error scraping news section {section_name}: {e}")
                    logger.exception("Full traceback for news section scraping error:")
                    continue

        except Exception as e:
            logger.error(f"Error navigating to or scraping news page: {e}")
            logger.exception("Full traceback for news page scraping error:")
        return all_news_data

    def get_news(self, target_month: Optional[str] = None) -> Tuple[List[Dict[str, str]], Dict[str, List[Dict[str, str]]]]:
        """Scrapes calendar data and news data and returns them."""
        try:
            if not self.check_robots_txt():
                logger.error("Scraping may violate robots.txt. Aborting.")
                return [], {}

            month, year = self.calculate_target_month(target_month)
            use_current = target_month is None

            calendar_url = self.build_calendar_url(month, year, use_current)
            logger.info(f"Target Calendar URL: {calendar_url}")

            scraped_calendar_events = []
            scraped_news_data = {}

            with self.get_driver() as driver:
                if self.config.requires_login:
                    logger.info("Attempting to log in...")
                    driver.get(self.config.login_url)

                    try:
                        username_field = WebDriverWait(driver, self.config.element_timeout).until(
                            EC.presence_of_element_located((By.ID, self.config.username_field_id))
                        )
                        password_field = driver.find_element(By.ID, self.config.password_field_id)
                        login_button = driver.find_element(By.CSS_SELECTOR, self.config.login_button_id)

                        username_field.send_keys(os.getenv(self.config.username_env_var))
                        password_field.send_keys(os.getenv(self.config.password_env_var))
                        login_button.click()

                        WebDriverWait(driver, self.config.element_timeout).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, self.config.post_login_element_css))
                        )
                        logger.info("Login successful.")
                        time.sleep(5) # Wait for 5 seconds after successful login

                    except TimeoutException as e:
                        logger.error(f"Login failed: Timeout waiting for login elements or post-login page. Exception: {e}")
                        logger.exception("Full traceback for TimeoutException during login:")
                        raise
                    except NoSuchElementException as e:
                        logger.error(f"Login failed: Could not find username, password, or login button. Exception: {e}")
                        logger.exception("Full traceback for NoSuchElementException during login:")
                        raise
                    except Exception as e:
                        logger.error(f"An unexpected error occurred during login: {e}")
                        logger.exception("Full traceback for unexpected Exception during login:")
                        raise

                # Scrape calendar data
                logger.info(f"Navigating to calendar URL: {calendar_url}")
                scraped_calendar_events = self.scrape_with_retries(driver, calendar_url, month, year)

                # Scrape news data
                scraped_news_data = self.scrape_news_data(driver)

            if scraped_calendar_events or scraped_news_data:
                logger.info(f"Scraping completed successfully. {len(scraped_calendar_events)} calendar events and {sum(len(v) for v in scraped_news_data.values())} news items found.")
                return scraped_calendar_events, scraped_news_data
            else:
                logger.error("No data was scraped from either calendar or news.")
                return [], {}

        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            logger.exception("Full traceback for scraping failure:")
            return [], {}

def main():
    config = ScrapingConfig()
    scraper = ForexCalendarScraper(config)
    calendar_data, news_data = scraper.get_news()
    
    # Save the data to JSON files
    if calendar_data:
        scraper.save_data_to_json({"calendar_data": calendar_data}, "calendar_data.json")
    if news_data:
        scraper.save_data_to_json({"news_data": news_data}, "news_data.json")

if __name__ == "__main__":
    main()