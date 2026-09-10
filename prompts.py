import os
import yaml

PROMPTS_DIR = "prompts"

def load_prompt(file_name: str, key: str) -> str:
    """Utility function to load prompt strings from YAML files inside /prompts."""
    file_path = os.path.join(PROMPTS_DIR, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompt file not found at {file_path}")
        
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    prompt_str = data.get(key)
    if not prompt_str:
        raise KeyError(f"Key '{key}' not found in {file_name}")
        
    return prompt_str
