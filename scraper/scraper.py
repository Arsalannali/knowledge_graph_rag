import os
import requests
from bs4 import BeautifulSoup
import time
import json
from urllib.parse import urljoin, urlparse
import logging
import argparse
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create data directory if it doesn't exist
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "scraped")
os.makedirs(DATA_DIR, exist_ok=True)

def get_domain(url):
    """Extract domain from URL"""
    parsed_uri = urlparse(url)
    return '{uri.scheme}://{uri.netloc}'.format(uri=parsed_uri)

def download_page(url):
    """Download page content with error handling and retry logic"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading {url}: {str(e)}")
            if attempt < max_retries - 1:
                sleep_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                logger.error(f"Failed to download {url} after {max_retries} attempts")
                return None

def extract_links(html, base_url):
    """Extract all links from HTML content"""
    soup = BeautifulSoup(html, 'html.parser')
    domain = get_domain(base_url)
    links = []
    
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        # Make relative URLs absolute
        full_url = urljoin(base_url, href)
        # Only include links from the same domain
        if get_domain(full_url) == domain:
            links.append(full_url)
    
    return list(set(links))  # Remove duplicates

def extract_text_content(html):
    """Extract main text content from HTML"""
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style", "nav", "footer", "header"]):
        script.extract()
    
    # Get text
    text = soup.get_text(separator=' ', strip=True)
    
    # Clean up text
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    return text

def scrape_website(start_url, max_pages=100):
    """Crawl website starting from start_url and store content"""
    visited = set()
    to_visit = [start_url]
    page_count = 0
    
    domain = get_domain(start_url)
    domain_name = urlparse(domain).netloc
    domain_dir = os.path.join(DATA_DIR, domain_name)
    os.makedirs(domain_dir, exist_ok=True)
    
    logger.info(f"Starting to scrape {start_url}")
    logger.info(f"Saving content to {domain_dir}")
    
    while to_visit and page_count < max_pages:
        url = to_visit.pop(0)
        
        if url in visited:
            continue
        
        logger.info(f"Scraping: {url}")
        
        html = download_page(url)
        if not html:
            visited.add(url)
            continue
        
        # Extract content
        text_content = extract_text_content(html)
        
        # Save content
        page_filename = f"page_{page_count:04d}.txt"
        with open(os.path.join(domain_dir, page_filename), 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        # Save metadata
        metadata = {
            'url': url,
            'title': BeautifulSoup(html, 'html.parser').title.string if BeautifulSoup(html, 'html.parser').title else 'No Title',
            'text_file': page_filename,
            'scrape_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(os.path.join(domain_dir, f"meta_{page_count:04d}.json"), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        # Find new links
        new_links = extract_links(html, url)
        for link in new_links:
            if link not in visited and link not in to_visit:
                to_visit.append(link)
        
        visited.add(url)
        page_count += 1
        
        # Be nice to the server
        time.sleep(1)
    
    logger.info(f"Scraping complete. Scraped {page_count} pages from {domain}")
    return {
        'domain': domain,
        'pages_scraped': page_count,
        'data_directory': domain_dir
    }

def main():
    """Main function to handle command-line arguments"""
    parser = argparse.ArgumentParser(description='Web scraper for RAG Application')
    parser.add_argument('url', nargs='?', help='URL to scrape')
    parser.add_argument('--max-pages', type=int, default=20, help='Maximum number of pages to scrape')
    
    args = parser.parse_args()
    
    if args.url:
        url = args.url
    else:
        url = input("Enter the URL to scrape: ")
    
    result = scrape_website(url, max_pages=args.max_pages)
    print(f"Scraped {result['pages_scraped']} pages from {result['domain']}")
    print(f"Data saved in {result['data_directory']}")

if __name__ == "__main__":
    main() 