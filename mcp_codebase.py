import os
import glob
from typing import List, Dict, Any
from langchain_core.tools import tool

# Default local projects directory (Update this path to your local projects folder)
DEFAULT_PROJECTS_DIR = os.path.expanduser("~/Sathya_Drive/DEV Workspace")


def scan_local_projects(query: str, root_dir: str = DEFAULT_PROJECTS_DIR) -> str:
    """Scans up to 15 local repositories for matches in READMEs and source code files."""
    if not os.path.exists(root_dir):
        return f"Projects directory '{root_dir}' not found on local machine."

    query_terms = [q.strip().lower() for q in query.split() if len(q.strip()) > 2]
    if not query_terms:
        return "No valid search terms provided."

    results = []
    # Fetch up to 15 project directories
    project_dirs = [
        d for d in glob.glob(os.path.join(root_dir, "*")) if os.path.isdir(d)
    ][:15]

    for project_path in project_dirs:
        project_name = os.path.basename(project_path)

        # 1. Check README.md
        readme_path = os.path.join(project_path, "README.md")
        if os.path.exists(readme_path):
            try:
                with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
                    readme_content = f.read()
                    if any(term in readme_content.lower() for term in query_terms):
                        results.append(
                            f"📁 **PROJECT MATCH (README)**: `{project_name}`\n"
                            f"**Overview Snippet:** {readme_content[:400]}...\n"
                        )
            except Exception:
                pass

        # 2. Check Source Code files
        supported_exts = ["*.py", "*.ts", "*.js", "*.swift", "*.ipynb"]
        for ext in supported_exts:
            for file_path in glob.glob(
                os.path.join(project_path, f"**/{ext}"), recursive=True
            ):
                # Ignore dependencies and virtual environments
                if any(
                    ignore_dir in file_path
                    for ignore_dir in [".venv", "node_modules", ".git", "build", "dist"]
                ):
                    continue

                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
                        for idx, line in enumerate(lines):
                            if any(term in line.lower() for term in query_terms):
                                start = max(0, idx - 8)
                                end = min(len(lines), idx + 8)
                                snippet = "".join(lines[start:end])
                                rel_path = os.path.relpath(file_path, project_path)

                                results.append(
                                    f"💻 **CODE MATCH**: `{project_name}` -> `{rel_path}` (Line {idx + 1})\n"
                                    f"```python\n{snippet}\n```"
                                )
                                break  # Cap to 1 match per file to avoid context bloat
                except Exception:
                    continue

    if not results:
        return f"No code or README matches found across local projects for query: '{query}'"

    return "\n\n".join(results[:3])  # Return top 3 strongest matches


@tool
def search_local_codebase_tool(search_query: str) -> str:
    """Searches local GitHub project repositories for exact code implementations and README definitions."""
    return scan_local_projects(search_query)
