import os
import time

from langchain_community.document_loaders import WebBaseLoader

# Set user agent to avoid the warning
os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
my_site=[
    # 'https://iclg.com/practice-areas/data-protection-laws-and-regulations/nigeria',
    # 'https://iclg.com/practice-areas/employment-and-labour-laws-and-regulations/nigeria',
    # 'https://iclg.com/practice-areas/technology-sourcing-laws-and-regulations/nigeria',
#     #united states
#     'https://iclg.com/practice-areas/data-protection-laws-and-regulations/usa',
#     'https://iclg.com/practice-areas/employment-and-labour-laws-and-regulations/usa'
#     'https://iclg.com/practice-areas/real-estate-laws-and-regulations/usa',
#     #united_kg
#     'https://iclg.com/practice-areas/data-protection-laws-and-regulations/united-kingdom'
#     'https://iclg.com/practice-areas/data-protection-laws-and-regulations/denmark',
#     'https://iclg.com/practice-areas/data-protection-laws-and-regulations/finland',
#     'https://iclg.com/practice-areas/data-protection-laws-and-regulations/france',
#     'https://iclg.com/practice-areas/data-protection-laws-and-regulations/germany',
# 'https://iclg.com/practice-areas/employment-and-labour-laws-and-regulations/singapore',
#     'https://iclg.com/practice-areas/employment-and-labour-laws-and-regulations/spain',
#     'https://iclg.com/practice-areas/employment-and-labour-laws-and-regulations/sweden',
#     'https://iclg.com/practice-areas/employment-and-labour-laws-and-regulations/switzerland',
    'https://iclg.com/practice-areas/employment-and-labour-laws-and-regulations/united-kingdom',
    'https://iclg.com/practice-areas/cybersecurity-laws-and-regulations/netherlands',
    'https://iclg.com/practice-areas/cybersecurity-laws-and-regulations/nigeria',
    'https://iclg.com/practice-areas/cybersecurity-laws-and-regulations/singapore',
    'https://iclg.com/practice-areas/cybersecurity-laws-and-regulations/united-kingdom',
    'https://iclg.com/practice-areas/cybersecurity-laws-and-regulations/usa',

]

def scrape_url():
    print("Scrapping ICGE documentation...")
    loader= WebBaseLoader(my_site)
    docs= loader.load()
    print(f'Scraped {len(docs)} document')

    return docs

