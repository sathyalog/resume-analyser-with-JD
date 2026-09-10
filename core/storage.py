import os
import json
from typing import Dict, Any

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

FILES = {
    "skills": os.path.join(DATA_DIR, "skills.json"),
    "challenges": os.path.join(DATA_DIR, "challenges.json"),
    "projects": os.path.join(DATA_DIR, "projects.json"),
    "misc": os.path.join(DATA_DIR, "misc.json"),
    "responsibilities": os.path.join(DATA_DIR, "responsibilities.json")
}

def load_json_data(file_key: str) -> list:
    filepath = FILES.get(file_key)
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_json_data(file_key: str, new_entry: Dict[str, Any]):
    filepath = FILES.get(file_key)
    existing_data = load_json_data(file_key)
    existing_data.append(new_entry)
    with open(filepath, "w") as f:
        json.dump(existing_data, f, indent=4)

def sync_to_vector_db(index_obj, doc_id: str, text_content: str, metadata: dict):
    """Placeholder vector update: generates mock sparse/dense vectors or upserts text chunks into Pinecone."""
    if index_obj:
        # Pass text to your embedding model (e.g. OpenAI/Cohere/HuggingFace)
        # index_obj.upsert(vectors=[(doc_id, vector_embedding, metadata)])
        pass
