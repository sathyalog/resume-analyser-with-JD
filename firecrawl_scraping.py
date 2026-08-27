# import os
# import re
# import requests
# from bs4 import BeautifulSoup
# from pydantic import BaseModel, Field
# from firecrawl import Firecrawl

# class ExtractedJobDescription(BaseModel):
#     job_title: str = Field(default="Job Position", description="The title of the job position")
#     years_experience_required: float = Field(default=0.0, description="Minimum total years of professional experience required")
#     required_skills: list[str] = Field(default_factory=list, description="Key technical skills mentioned")
#     job_overview: str = Field(default="", description="Summary of key duties and responsibilities")

# # def compress_markdown_tokens(text: str) -> str:
# #     """Utility function to strip web boilerplate noise, URLs, and extra whitespaces to reduce tokens."""
# #     if not text:
# #         return ""
    
# #     # 1. Remove base64 image strings and markdown images ![alt](url)
# #     text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    
# #     # 2. Convert markdown links [link text](http://...) to just "link text"
# #     text = re.sub(r'\[(.*?)\]\(https?://\S+\)', r'\1', text)
    
# #     # 3. Remove standalone bare URLs
# #     text = re.sub(r'https?://\S+', '', text)
    
# #     # 4. Remove common web cookie / tracking phrases
# #     text = re.sub(r'(?i)(accept cookies|privacy policy|terms of service|manage preferences|all rights reserved)', '', text)
    
# #     # 5. Collapse multiple newlines and spaces into single spacing
# #     text = re.sub(r'\n\s*\n', '\n', text)
    
# #     return text.strip()
# def compress_markdown_tokens(text: str) -> str:
#     if not text:
#         return ""
#     # Strip links, images, extra formatting, and HTML artifacts
#     text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
#     text = re.sub(r'\[(.*?)\]\(https?://\S+\)', r'\1', text)
#     text = re.sub(r'https?://\S+', '', text)
#     text = re.sub(r'<[^>]+>', '', text)
#     text = re.sub(r'(?i)(accept cookies|privacy policy|terms of service|manage preferences|all rights reserved)', '', text)
#     text = re.sub(r'\n\s*\n', '\n', text)
#     return text.strip()

# def fallback_web_scrape(url: str) -> dict:
#     """Fallback scraping method using requests and BeautifulSoup with token optimization."""
#     headers = {
#         "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
#     }
#     try:
#         response = requests.get(url, headers=headers, timeout=10)
#         if response.status_code == 200:
#             soup = BeautifulSoup(response.text, "html.parser")
            
#             # Decompose boilerplate elements and noise tags
#             for element in soup(["script", "style", "nav", "footer", "header", "form", "aside", "svg"]):
#                 element.decompose()
                
#             clean_text = soup.get_text(separator="\n", strip=True)
#             compressed_text = compress_markdown_tokens(clean_text)
            
#             if len(compressed_text) > 100:
#                 return {
#                     "job_title": "Position (Scraped via Web Fallback)",
#                     "years_experience_required": 0.0,
#                     "required_skills": [],
#                     "job_overview": compressed_text[:3500]  # Hard cap at ~800 tokens max
#                 }
#     except Exception as e:
#         print(f"Fallback BeautifulSoup scrape failed: {e}")
        
#     return {}

# def extract_jd_from_url(url: str) -> dict:
#     api_key = os.getenv("FIRECRAWL_API_KEY")
    
#     def infer_metadata_from_text(text: str) -> dict:
#         """Fallback utility to extract title and experience if Firecrawl returns incomplete metadata."""
#         title = "Lead AI Engineer" if "Lead AI Engineer" in text else "AI Engineer"
        
#         # Infer experience based on seniority keywords if explicit numbers are missing
#         exp_required = 0.0
#         text_lower = text.lower()
#         if "lead" in text_lower or "principal" in text_lower:
#             exp_required = 8.0
#         elif "senior" in text_lower:
#             exp_required = 5.0
            
#         return {
#             "job_title": title,
#             "years_experience_required": exp_required
#         }

    
#     # 1. Try Firecrawl Structured Extraction
#     if api_key:
#         app = Firecrawl(api_key=api_key)
#         try:
#             response = app.extract(
#                 urls=[url],
#                 schema=ExtractedJobDescription.model_json_schema(),
#                 prompt="Extract job_title, years_experience_required as a float, required_skills list, and job_overview."
#             )
            
#             data = getattr(response, "data", response)
#             if isinstance(data, list) and len(data) > 0:
#                 data = data[0]
                
#             if hasattr(data, "model_dump"):
#                 data = data.model_dump()
                
#             if isinstance(data, dict) and (data.get("job_title") or data.get("job_overview")):
#                 # Compress overview tokens before returning
#                 data["job_overview"] = compress_markdown_tokens(data.get("job_overview", ""))[:3500]
#                 return data

#         except Exception as e:
#             print(f"Firecrawl extract failed: {e}. Attempting Firecrawl markdown scrape...")

#                 # 2. Try Firecrawl Scrape Fallback (with native parameter token trimming)
#         try:
#             # Inject native Firecrawl options using snake_case parameter names
#             scrape_result = app.scrape(
#                 url=url, 
#                 formats=["markdown"],
#                 only_main_content=True,  # Fixed: changed from onlyMainContent
#                 exclude_tags=[           # Fixed: changed from excludeTags
#                     "nav", "footer", "header", "aside", 
#                     ".cookie-banner", "#cookie-consent", 
#                     ".similar-jobs", ".recommended-jobs"
#                 ]
#             )
            
#             markdown_text = getattr(scrape_result, "markdown", "") or scrape_result.get("markdown", "")
            
#             if markdown_text:
#                 compressed_md = compress_markdown_tokens(markdown_text)
#                 return {
#                     "job_title": "Position (Scraped from URL)",
#                     "years_experience_required": 0.0,
#                     "required_skills": [],
#                     "job_overview": compressed_md[:3500]  # Hard cap at ~800 tokens max
#                 }
#         except Exception as scrape_err:
#             print(f"Firecrawl scrape failed or blocked site: {scrape_err}")

#     # 3. Final Fallback: Direct BeautifulSoup Scraper
#     print("Executing BeautifulSoup fallback scraper...")
#     return fallback_web_scrape(url)

# replacing firecrawl with apify
import os
import re
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from apify_client import ApifyClient


class ExtractedJobDescription(BaseModel):
    job_title: str = Field(default="Job Position", description="The title of the job position")
    years_experience_required: float = Field(default=0.0, description="Minimum total years of professional experience required")
    required_skills: list[str] = Field(default_factory=list, description="Key technical skills mentioned")
    job_overview: str = Field(default="", description="Summary of key duties and responsibilities")


def compress_markdown_tokens(text: str) -> str:
    """Utility function to strip web boilerplate noise, URLs, and extra whitespaces to reduce tokens."""
    if not text:
        return ""
    
    # 1. Remove markdown images ![alt](url)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    
    # 2. Convert markdown links [link text](http://...) to plain text
    text = re.sub(r'\[(.*?)\]\(https?://\S+\)', r'\1', text)
    
    # 3. Remove standalone bare URLs and HTML tags
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    
    # 4. Remove cookie/tracking boilerplate
    text = re.sub(r'(?i)(accept cookies|privacy policy|terms of service|manage preferences|all rights reserved)', '', text)
    
    # 5. Collapse duplicate newlines and whitespace
    text = re.sub(r'\n\s*\n', '\n', text)
    
    return text.strip()


def transform_linkedin_url(url: str) -> str:
    """Extracts job ID from standard LinkedIn job URLs and builds an unauthenticated guest API endpoint."""
    # Matches numeric job ID from URLs like:
    # https://www.linkedin.com/jobs/view/4448853020/ or https://www.linkedin.com/jobs/collections/recommended/?currentJobId=4448853020
    match = re.search(r'(?:/jobs/view/|currentJobId=)(\d+)', url)
    
    if match:
        job_id = match.group(1)
        return f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"
    
    return url


def fallback_web_scrape(url: str) -> dict:
    """Enhanced fallback scraping method using browser-like headers."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Sec-Ch-Ua": '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"macOS"',
        "Upgrade-Insecure-Requests": "1"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=12)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            
            for element in soup(["script", "style", "nav", "footer", "header", "form", "aside", "svg"]):
                element.decompose()
                
            clean_text = soup.get_text(separator="\n", strip=True)
            compressed_text = compress_markdown_tokens(clean_text)
            
            if len(compressed_text) > 100:
                return {
                    "job_title": "Position (Scraped via Web Fallback)",
                    "years_experience_required": 0.0,
                    "required_skills": [],
                    "job_overview": compressed_text[:3500]
                }
    except Exception as e:
        print(f"Fallback BeautifulSoup scrape failed: {e}")
        
    return {}


def extract_jd_from_url(url: str) -> dict:
    # 1. Check if URL is from LinkedIn and convert to unauthenticated Guest API endpoint
    target_url = transform_linkedin_url(url)
    
    if "jobs-guest/jobs/api/jobPosting" in target_url:
        print(f"Bypassing LinkedIn auth wall via Guest API: {target_url}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }
        try:
            res = requests.get(target_url, headers=headers, timeout=10)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, "html.parser")
                
                # Remove boilerplate scripts and navigation tags
                for tag in soup(["script", "style", "button", "nav"]):
                    tag.decompose()
                
                job_title_el = soup.find("h2") or soup.find("h1")
                title_text = job_title_el.get_text(strip=True) if job_title_el else "LinkedIn Job Position"
                
                clean_text = soup.get_text(separator="\n", strip=True)
                compressed = compress_markdown_tokens(clean_text)
                
                if len(compressed) > 150:
                    return {
                        "job_title": title_text,
                        "years_experience_required": 0.0,
                        "required_skills": [],
                        "job_overview": compressed[:3500]
                    }
        except Exception as e:
            print(f"LinkedIn Guest API fetch failed: {e}")

    # 2. Try Apify Web Scraper for standard non-LinkedIn websites
    api_token = os.getenv("APIFY_API_TOKEN")
    if api_token:
        print("Executing Apify Web Scraper...")
        client = ApifyClient(api_token)
        
        run_input = {
            "runMode": "PRODUCTION",
            "startUrls": [{"url": target_url}],
            "pageFunction": """async function pageFunction(context) {
                const title = document.querySelector('h1')?.innerText || document.title || 'Job Position';
                const bodyText = document.body?.innerText || '';
                return {
                    title: title.trim(),
                    text: bodyText.trim()
                };
            }""",
            "proxyConfiguration": {"useApifyProxy": True},
            "waitUntil": ["domcontentloaded"]
        }

        try:
            run = client.actor("apify/web-scraper").call(run_input=run_input)
            
            dataset_id = run.get("defaultDatasetId") if isinstance(run, dict) else getattr(run, "default_dataset_id", None)
            
            if dataset_id:
                for item in client.dataset(dataset_id).iterate_items():
                    jd_text = item.get("text", "")
                    job_title = item.get("title") or "Job Position (Parsed from URL)"
                    
                    if jd_text:
                        compressed_md = compress_markdown_tokens(jd_text)
                        return {
                            "job_title": job_title,
                            "years_experience_required": 0.0,
                            "required_skills": [],
                            "job_overview": compressed_md[:3500]
                        }
        except Exception as e:
            print(f"Apify execution error: {e}")

    # 3. Final Fallback: Direct BeautifulSoup HTTP Request
    print("Executing HTTP fallback scraper...")
    return fallback_web_scrape(target_url)



