import requests
import pdfplumber
import os
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import time
import json

visited_urls = set()  # Store visited URLs
extracted_data = []  # List to store content of all pages

def fetch_content(url):
    """Fetches the content from the given URL."""
    try:
        response = requests.get(url, timeout=10)  # Fetch page
        response.raise_for_status()  # Check for errors
        return response.text
    except requests.RequestException as e:
        print(f"Failed to access {url}: {e}")
        return None

def get_page_type(url):
    """Determines the type of content (HTML, PDF, or Plain Text) from the URL."""
    if url.endswith(".pdf"):
        return "PDF"
    elif url.endswith(".txt"):
        return "TEXT"
    else:
        return "HTML"

def download_pdf(url):
    """Downloads the PDF file to the local machine."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        filename = url.split("/")[-1]  # Extract the filename from the URL
        with open(filename, 'wb') as f:
            f.write(response.content)
        return filename
    except requests.RequestException as e:
        print(f"Failed to download PDF: {e}")
        return None

def extract_text_from_html(soup):
    """Extracts and returns text content from an HTML page."""
    return soup.get_text(separator='\n', strip=True)

def extract_text_from_pdf(file_path):
    """Extracts text from a PDF document (using pdfplumber)."""
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ''
            for page in pdf.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return ""

def extract_text_from_plain_text(url):
    """Extracts text content from a plain text document."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Failed to fetch plain text file: {e}")
        return ""

def get_all_pages(base_url, url, level=1, max_depth=1):
    """Fetches the content of the main page and first-level subpages."""
    if url in visited_urls:
        return
    visited_urls.add(url)  # Mark as visited
    
    # Fetch the content from the page
    page_content = fetch_content(url)
    if not page_content:
        return

    # Create the BeautifulSoup object to parse the HTML
    soup = BeautifulSoup(page_content, 'html.parser')

    # Determine the type of content
    page_type = get_page_type(url)
    
    # Extract text content based on the page type
    if page_type == 'HTML':
        extracted_content = extract_text_from_html(soup)
    elif page_type == 'PDF':
        pdf_filename = download_pdf(url)  # Download the PDF to local
        if pdf_filename:
            extracted_content = extract_text_from_pdf(pdf_filename)  # Extract text from the downloaded PDF
            os.remove(pdf_filename)  # Clean up by removing the downloaded PDF file
        else:
            extracted_content = "Failed to download PDF"
    elif page_type == 'TEXT':
        extracted_content = extract_text_from_plain_text(url)
    else:
        extracted_content = "Unsupported content type"
    
    # Prepare the data structure to store
    data = {
        "url": url,
        "type": page_type,
        "content": extracted_content
    }

    # Store the extracted content in the list
    extracted_data.append(data)

    # Fetch first-level sublinks (if the current page is HTML)
    if level < max_depth:  # Crawl only the first-level subpages
        for link in soup.find_all('a', href=True):  # Extract all anchor tags
            next_url = urljoin(base_url, link['href'])  # Construct absolute URL
            if urlparse(next_url).netloc == urlparse(base_url).netloc:  # Same domain
                get_all_pages(base_url, next_url, level+1, max_depth)
    
    time.sleep(0.5)  # Be polite and avoid server overload

def save_to_json():
    """Save the extracted content to a JSON file."""
    with open('extracted_data.json', 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=4)

# Example Usage
start_urls = ["https://www.heinz.cmu.edu/",
              "https://www.heinz.cmu.edu/heinz-shared/_files/img/current-students/heinz-academic-calendar-2024-25.pdf",
              "https://en.wikipedia.org/wiki/Pittsburgh",
              "https://en.wikipedia.org/wiki/History_of_Pittsburgh",
              "https://www.pittsburghpa.gov/Home",
              "https://www.britannica.com/place/Pittsburgh",
              "https://www.visitpittsburgh.com",
              "https://www.pittsburghpa.gov/City-Government/Finances-Budget/Taxes/Tax-Forms",
              "https://apps.pittsburghpa.gov/redtail/images/23255_2024_Operating_Budget.pdf",
              "https://www.cmu.edu/about/",
                
                # Events
                "https://pittsburgh.events",
                "https://downtownpittsburgh.com/events/",
                "https://www.pghcitypaper.com/pittsburgh/EventSearch?v=d",
                "https://events.cmu.edu",
                "https://www.cmu.edu/engage/alumni/events/campus/index.html",
                
                # Music and Culture
                "https://www.pittsburghsymphony.org",
                "https://pittsburghopera.org",
                "https://trustarts.org",
                "https://carnegiemuseums.org",
                "https://www.heinzhistorycenter.org",
                "https://www.thefrickpittsburgh.org",
                "https://en.wikipedia.org/wiki/List_of_museums_in_Pittsburgh",
                
                # Food-related events
                "https://www.visitpittsburgh.com/events-festivals/food-festivals/",
                "https://www.picklesburgh.com",
                "https://www.pghtacofest.com",
                "https://pittsburghrestaurantweek.com",
                "https://littleitalydays.com",
                "https://bananasplitfest.com",
                
                # Sports
                "https://www.visitpittsburgh.com/things-to-do/pittsburgh-sports-teams/",
                "https://www.mlb.com/pirates",
                "https://www.steelers.com",
                "https://www.nhl.com/penguins/"
              
]

for start_url in start_urls:
    get_all_pages(start_url, start_url, level=1, max_depth=2)  # Fetch main page and first subpages
    save_to_json()  # Save the extracted content to a JSON file     

# Output the stored data to check
print("Extracted data stored in 'extracted_data.json':")
print(json.dumps(extracted_data, indent=2))
