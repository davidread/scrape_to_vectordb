from typing import ClassVar, List

import scrapy
from scrapy.crawler import CrawlerProcess
import lancedb
import pandas as pd
import numpy as np
import pyarrow as pa
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup

class SecuritySpider(scrapy.Spider):
    name = 'security_spider'
    start_urls = ['https://security-guidance.service.justice.gov.uk/']
    allowed_domains = ['security-guidance.service.justice.gov.uk']
    documents: ClassVar[List[dict]] = []  # Class variable to store documents
    
    def should_follow(self, href: str) -> bool:
        # Skip non-HTML files and other unwanted links
        skip_patterns = (
            'mailto:', 'tel:', '#', 'javascript:',
            '.pdf', '.doc', '.docx', '.xls', '.xlsx',
            '.png', '.jpg', '.jpeg', '.gif', 
            'http://', 'https://', '//'
        )
        return href and not href.startswith(skip_patterns)
    
    def parse(self, response):
        if not response.headers.get('Content-Type', b'').startswith(b'text/html'):
            return
        
        # Extract content using BeautifulSoup for better parsing
        soup = BeautifulSoup(response.text, 'html.parser')
        main_content = soup.find('main') or soup.find('article') or soup.body
        
        if main_content:
            # Get title
            title = soup.find('h1')
            title_text = title.get_text(strip=True) if title else response.url
            
            # Get content
            content_parts = []
            for element in main_content.find_all(['p', 'li', 'h2', 'h3', 'h4', 'code']):
                text = element.get_text(strip=True)
                if text:
                    content_parts.append(text)
            
            if content_parts:
                SecuritySpider.documents.append({
                    'url': response.url,
                    'title': title_text,
                    'content': ' '.join(content_parts)
                })

        # Follow valid links
        for href in response.css('a::attr(href)').getall():
            if self.should_follow(href):
                yield response.follow(href, self.parse)

def ensure_float32_list(embedding):
    """Convert embedding to float32 and then to list."""
    return np.array(embedding, dtype=np.float32).tolist()


def create_database():
    # Create and run the spider
    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/5.0',
        'ROBOTSTXT_OBEY': True,
        'CONCURRENT_REQUESTS': 2,  # Be gentle with the server
        'DEPTH_LIMIT': 5  # Limit crawl depth if needed
    })

    # Clear any previous documents
    SecuritySpider.documents = []

    process.crawl(SecuritySpider)
    process.start()  # it blocks here until all crawling jobs are finished
    
    if not SecuritySpider.documents:
        print("No documents found!")
        return
    
    # Create embeddings
    print(f"Creating embeddings for {len(SecuritySpider.documents)} documents...")
    df = pd.DataFrame(SecuritySpider.documents)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    combined_texts = [f"{row['title']}: {row['content']}" for _, row in df.iterrows()]
    raw_embeddings = model.encode(combined_texts, show_progress_bar=True)
    embeddings = [ensure_float32_list(emb) for emb in raw_embeddings]

    df['embedding'] = embeddings
    
    # Create LanceDB database
    print("Creating LanceDB database...")
    db = lancedb.connect("./.lancedb")
    
    schema = pa.schema([
        ('url', pa.string()),
        ('title', pa.string()),
        ('content', pa.string()),
        ('embedding', pa.list_(pa.float32(), 384))
    ])

    table = db.create_table(
        "security_guidance",
        data=df,
        schema=schema,
        mode="overwrite"
    )
    
    # Insert data
    table.add(df)

    print(f"Done! Stored {len(df)} documents")
    return table

if __name__ == "__main__":
    create_database()