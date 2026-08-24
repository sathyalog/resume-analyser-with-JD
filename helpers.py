import re
from typing import Optional
# Helper function to extract GitHub username from text
def extract_github_handle(text: str) -> Optional[str]:
    # Matches patterns like github.com/username or github.com/username/
    pattern = r"github\.com/([a-zA-Z0-9_-]+)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


extracted_resume_text = ""
github_username = None