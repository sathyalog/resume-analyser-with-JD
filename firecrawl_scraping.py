import os
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from firecrawl import Firecrawl

class ExtractedJobDescription(BaseModel):
    job_title: str = Field(default="Job Position", description="The title of the job position")
    years_experience_required: float = Field(default=0.0, description="Minimum total years of professional experience required")
    required_skills: list[str] = Field(default_factory=list, description="Key technical skills mentioned")
    job_overview: str = Field(default="", description="Summary of key duties and responsibilities")

def fallback_web_scrape(url: str) -> dict:
    """Fallback scraping method using requests and BeautifulSoup for Firecrawl-restricted domains."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()
                
            clean_text = soup.get_text(separator="\n", strip=True)
            
            if len(clean_text) > 100:
                return {
                    "job_title": "Position (Scraped via Web Fallback)",
                    "years_experience_required": 0.0,
                    "required_skills": [],
                    "job_overview": clean_text[:4000] # Limit tokens passed to LLM
                }
    except Exception as e:
        print(f"Fallback BeautifulSoup scrape failed: {e}")
        
    return {}

def extract_jd_from_url(url: str) -> dict:
    api_key = os.getenv("FIRECRAWL_API_KEY")
    
    # 1. Try Firecrawl Structured Extraction
    if api_key:
        try:
            app = Firecrawl(api_key=api_key)
            response = app.extract(
                urls=[url],
                schema=ExtractedJobDescription.model_json_schema(),
                prompt="Extract job_title, years_experience_required as a float, required_skills list, and job_overview."
            )
            
            data = getattr(response, "data", response)
            if isinstance(data, list) and len(data) > 0:
                data = data[0]
                
            if hasattr(data, "model_dump"):
                data = data.model_dump()
                
            if isinstance(data, dict) and (data.get("job_title") or data.get("job_overview")):
                return data

        except Exception as e:
            print(f"Firecrawl extract failed: {e}. Attempting Firecrawl markdown scrape...")

        # 2. Try Firecrawl Scrape Fallback
        try:
            app = Firecrawl(api_key=api_key)
            scrape_result = app.scrape(url=url, formats=["markdown"])
            markdown_text = getattr(scrape_result, "markdown", "") or scrape_result.get("markdown", "")
            
            if markdown_text:
                return {
                    "job_title": "Position (Scraped from URL)",
                    "years_experience_required": 0.0,
                    "required_skills": [],
                    "job_overview": markdown_text
                }
        except Exception as scrape_err:
            print(f"Firecrawl scrape failed or blocked site: {scrape_err}")

    # 3. Final Fallback: Direct BeautifulSoup HTTP Scraper (Bypasses Firecrawl Domain Restrictions)
    print("Executing BeautifulSoup fallback scraper...")
    return fallback_web_scrape(url)
