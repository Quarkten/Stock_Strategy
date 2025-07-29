FROM python:3.11-slim-buster
WORKDIR /app
COPY requirements_scraper.txt .
RUN pip install -r requirements_scraper.txt
COPY scraper.py .
CMD ["python", "scraper.py"]