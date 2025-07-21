import os
import requests
from bs4 import BeautifulSoup
import time
import random
from urllib.parse import quote_plus

from utils import API_KEY_SPRINGER, API_KEY_ELSEVIER, QUERY, query

"""
Script for downloading papers from arXiv based on a search query.
params:
    download_dir: str, directory to save downloaded papers
    max_retries: int, number of retries for downloading a paper
methods:
    download_from_arxiv/springer/elsevier: download papers from arxiv/springer/elsevier based on a search query and save metadata
    _download_pdf: download a single paper from a given URL
    download_papers: download multiple papers based on a search query
returns:
    list of downloaded papers
"""

class PaperDownloader:
    def __init__(self, download_dir='downloaded_papers', max_retries=10):
        self.download_dir = download_dir
        self.max_retries = max_retries
        os.makedirs(download_dir, exist_ok=True)

    def _download_from_arxiv(self, query, num_papers=50):
        base_url = 'https://export.arxiv.org/api/query?'
        search_query = f'{base_url}search_query=all:{query}&start=0&max_results={num_papers}'

        response = requests.get(search_query)
        soup = BeautifulSoup(response.content, 'xml')

        # save metadata
        with open(os.path.join(self.download_dir, 'metadata', 'arxiv_metadata.txt'), 'w') as f:
            for entry in soup.find_all('entry')[:num_papers]:
                metadata = {
                    'title': entry.title.text,
                    'authors': [author.text for author in entry.find_all('author')],
                    'published': entry.published.text,
                    'summary': entry.summary.text,
                    'source': 'arxiv'
                }
                f.write(str(metadata) + '\n')


        papers = []
        for entry in soup.find_all('entry')[:num_papers]:
            title = entry.title.text
            pdf_link = entry.find('link', type='application/pdf')['href']

            try:
                self._download_pdf(pdf_link, title)
                papers.append(title)
            except Exception as e:
                print(f"Error downloading {title}: {e}")

            time.sleep(random.uniform(1, 3))

        return papers

    def _download_from_springer(self, query, num_papers=50):
        base_url = 'https://api.springernature.com/openaccess/json?'
        search_query = f'{base_url}api_key={API_KEY_SPRINGER}&callback=&s=1&p={num_papers}&q=(keyword:{query})'

        response_springer = requests.request("GET", search_query)
        response_springer.raise_for_status()
        data = response_springer.json()

        # Ensure metadata directory exists
        metadata_dir = os.path.join(self.download_dir, 'metadata')
        os.makedirs(metadata_dir, exist_ok=True)

        # Save metadata
        metadata_path = os.path.join(metadata_dir, 'springer_metadata.txt')
        with open(metadata_path, 'w') as f:
            for record in data.get('records', []):
                metadata = {
                    'title': record.get('title'),
                    'authors': [creator.get('creator') for creator in record.get('creators', [])],
                    'published': record.get('publicationDate'),
                    'summary': record.get('abstract', {}).get('p', ''),
                    'source': 'springer'
                }
                f.write(str(metadata) + '\n')

        papers = []
        for record in data.get('records', []):
            title = record.get('title')
            page_link = None

            for url in record.get('url', []):
                if url.get('value').startswith('http'):
                    page_link = url.get('value')
                    break

            if page_link:
                try:
                    pdf_link = self._get_pdf_link_from_page(page_link)
                    self._download_pdf(pdf_link, title)
                    papers.append(title)
                except Exception as e:
                    print(f"Error downloading {title}: {e}")

                # Rate-limiting to prevent excessive requests
                time.sleep(random.uniform(1, 3))
            else:
                print(f"No PDF link found for {title}")

        return papers

    def _download_from_elsevier(self, query, num_papers=50):
        base_url = 'https://api.elsevier.com/content/search/scopus?'
        encoded_query = quote_plus(query)
        url = f'{base_url}query={encoded_query}&count={min(num_papers, 50)}'

        headers = {
            'X-ELS-APIKey': API_KEY_ELSEVIER,
            #'X-ELS-Insttoken': 'YOUR_INSTITUTION_TOKEN', #todo ask for this
            'Accept': 'application/json'
        }

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            papers = []
            for entry in data.get('search-results', {}).get('entry', []):
                # Get Scopus ID
                scopus_id = entry.get('dc:identifier', '').replace('SCOPUS_ID:', '')
                if not scopus_id:
                    print(f"No Scopus ID found for paper: {entry.get('dc:title', 'No title')}")
                    continue

                # Get abstract and full-text links using Scopus ID
                abstract_url = f'https://api.elsevier.com/content/abstract/scopus_id/{scopus_id}'
                abstract_headers = {
                    'X-ELS-APIKey': API_KEY_ELSEVIER,
                    'Accept': 'application/json'
                }

                try:
                    # Get the abstract response to check for open access and get PDF link
                    abstract_response = requests.get(abstract_url, headers=abstract_headers)
                    abstract_response.raise_for_status()

                    soup = BeautifulSoup(abstract_response.content, 'xml')

                    # Check if paper is open access
                    is_open_access = soup.find('openaccess')
                    if not is_open_access or is_open_access.text != '1':
                        print(f"Paper not open access: {entry.get('dc:title', 'No title')}")
                        continue

                    # Try to get the PDF link
                    pdf_link = None
                    for link in soup.find_all('link'):
                        if link.get('ref') == 'full-text':
                            pdf_link = link.get('href')
                            break

                    if pdf_link:
                        try:
                            # Download the PDF
                            title = entry.get('dc:title', 'No title')
                            self._download_pdf(pdf_link, title)
                            papers.append(title)
                            print(f"Successfully processed: {title}")
                        except Exception as e:
                            print(f"Error downloading PDF: {e}")
                    else:
                        print(f"No PDF link found for paper: {entry.get('dc:title', 'No title')}")

                except requests.RequestException as e:
                    print(f"Error retrieving abstract data: {e}")

                time.sleep(random.uniform(1, 3))  # Polite delay between requests

            return papers

        except requests.RequestException as e:
            print(f"Error fetching search results from Elsevier: {e}")
            return []

    def _download_pdf(self, url, title):
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()

                # Create a safe filename
                safe_filename = "".join(x for x in title if x.isalnum() or x in [' ', '-']).rstrip()
                filename = os.path.join(self.download_dir, f"{safe_filename}_{attempt}.pdf")

                with open(filename, 'wb') as f:
                    f.write(response.content)

                print(f"Successfully downloaded: {filename}")
                return filename

            except requests.RequestException as e:
                print(f"Download attempt {attempt + 1} failed: {e}")
                time.sleep(2)

        raise Exception(f"Failed to download paper after {self.max_retries} attempts")

    def _get_pdf_link_from_page(self, page_url):
        """
        Scrape the article page to find the 'Download PDF' link.
        """
        response = requests.get(page_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')

        # Find the meta tag with the PDF URL
        pdf_meta = soup.find('meta', attrs={'name': 'citation_pdf_url'})

        if pdf_meta and pdf_meta.get('content'):
            pdf_link = pdf_meta['content']
            return pdf_link
        else:
            print(f"No PDF link found on the page: {page_url}")
            return None

    def download_papers(self, query, num_papers=300):
        print(f"Downloading papers for query: {query}")

        arxiv_papers = self._download_from_arxiv(query, num_papers)
        springer_papers = self._download_from_springer(query, num_papers)
        elsevier_papers = self._download_from_elsevier(query, num_papers)

        total_papers = len(springer_papers) + len(arxiv_papers) + len(elsevier_papers)
        print(f"Total papers downloaded: {total_papers}")



# Query for papers on species interaction networks
if __name__ == "__main__":
    downloader = PaperDownloader()
    """for query in QUERY:
        downloader.download_papers(query, num_papers=10)
        print(f"Finished downloading papers for query: {query}")
"""
    downloader.download_papers("Ecological Network", num_papers=10)