import asyncio
import json
import os
import re
from typing import Any, Dict, List
from langchain_mcp_adapters.client import MultiServerMCPClient


def extract_skills_from_state(state: Dict[str, Any]) -> List[str]:
    """
    Tries multiple keys in state to retrieve skills.
    If no explicit list is found, it extracts common tech keywords from raw resume text.
    """
    # 1. Check explicit skill keys in state
    for key in ["resume_skills", "skills", "extracted_skills"]:
        val = state.get(key)
        if val:
            if isinstance(val, list):
                return [str(s).strip() for s in val if s]
            if isinstance(val, str):
                return [s.strip() for s in val.split(",") if s.strip()]

    # 2. Check nested dictionaries (e.g., state["resume_data"]["skills"])
    resume_data = state.get("resume_data") or state.get("resume_analysis") or {}
    if isinstance(resume_data, dict):
        skills = resume_data.get("skills") or resume_data.get("core_skills")
        if skills:
            if isinstance(skills, list):
                return [str(s).strip() for s in skills if s]
            if isinstance(skills, str):
                return [s.strip() for s in skills.split(",") if s.strip()]

    # 3. Fallback: Parse raw resume text using a standard tech keyword list
    raw_text = state.get("resume_text", "") or state.get("pdf_text", "")
    if isinstance(raw_text, str) and raw_text:
        common_tech = [
            "python", "react", "node", "typescript", "javascript", "langgraph", 
            "langchain", "mcp", "azure", "docker", "fastapi", "streamlit", 
            "pinecone", "chroma", "rag", "pydantic", "redux", "postgres", "sql"
        ]
        found = []
        text_lower = raw_text.lower()
        for tech in common_tech:
            if re.search(r"\b" + re.escape(tech) + r"\b", text_lower):
                found.append(tech)
        return found

    return []


def check_resume_github_skills_match(resume_skills: List[str], repo_items: List[Dict[str, Any]]) -> str:
    """
    Compares core skills extracted from the resume against GitHub repo languages,
    topics, descriptions, and repository names.
    """
    if not resume_skills or not repo_items:
        return "> ⚠️ **Resume GitHub Matching Score:** Unable to evaluate skill match due to missing resume skills in state.\n"

    # Aggregate text from GitHub repositories
    github_corpus = []
    for repo in repo_items:
        if repo.get("language"):
            github_corpus.append(str(repo["language"]).lower())
        
        topics = repo.get("topics", [])
        for topic in topics:
            github_corpus.append(str(topic).lower())

        name = repo.get("name", "").lower().replace("-", " ").replace("_", " ")
        desc = (repo.get("description") or "").lower().replace("-", " ").replace("_", " ")
        github_corpus.append(name)
        github_corpus.append(desc)

    full_github_text = " ".join(github_corpus)

    # Search for matched skills in the corpus
    matched_skills = []
    for skill in resume_skills:
        skill_clean = skill.strip().lower()
        if not skill_clean:
            continue
        
        # Word boundary search to prevent false positive partial matches
        pattern = r"\b" + re.escape(skill_clean) + r"\b"
        if re.search(pattern, full_github_text):
            matched_skills.append(skill.strip())

    if matched_skills:
        skills_str = ", ".join(list(set(matched_skills)))
        return (
            f"> 💡 **Resume GitHub Matching Score:** **Yes**, Candidate skills are verified in GitHub public repositories.\n"
            f"> **Matched Skills Found:** `{skills_str}`\n"
        )
    else:
        return (
            f"> 💡 **Resume GitHub Matching Score:** **No**, primary resume skills were not directly detected in GitHub repository titles/descriptions.\n"
        )


def format_github_response(repos_raw: Any, commits_raw: Any, resume_skills: List[str] = None) -> str:
    output_markdown = []
    items = []

    # 1. Parse Repositories
    if repos_raw:
        try:
            parsed_repos = (
                json.loads(repos_raw)
                if isinstance(repos_raw, str)
                else repos_raw
            )
            if (
                isinstance(parsed_repos, list)
                and len(parsed_repos) > 0
                and isinstance(parsed_repos[0], dict)
                and "text" in parsed_repos[0]
            ):
                parsed_repos = json.loads(parsed_repos[0]["text"])

            if isinstance(parsed_repos, dict):
                items = parsed_repos.get("items", [])
                total_count = parsed_repos.get("total_count", len(items))
            elif isinstance(parsed_repos, list):
                items = parsed_repos
                total_count = len(items)

            items = sorted(
                items, key=lambda x: x.get("updated_at", ""), reverse=True
            )

            # Skill match evaluation callout
            skill_match_msg = check_resume_github_skills_match(resume_skills, items)
            output_markdown.append(skill_match_msg)

            output_markdown.append(
                f"### 📦 Public Repositories (Total: {total_count})\n"
            )

            if items:
                repo_table = [
                    "| Repository | Description | Primary Language | Stars | Last Updated |",
                    "| :--- | :--- | :--- | :--- | :--- |",
                ]
                for repo in items[:10]:
                    name = repo.get("name", "N/A")
                    html_url = repo.get("html_url", "#")
                    desc = (
                        repo.get("description") or "No description provided."
                    )
                    desc = desc.replace("\n", " ").replace("|", "-")
                    lang = repo.get("language") or "N/A"
                    stars = repo.get("stargazers_count", 0)
                    updated = repo.get("updated_at", "")[:10]

                    repo_table.append(
                        f"| [{name}]({html_url}) | {desc} | `{lang}` | {stars} | **{updated}** |"
                    )

                output_markdown.append("\n".join(repo_table))
            else:
                output_markdown.append("No public repositories found.")

        except Exception as e:
            output_markdown.append(f"**Repositories Parsing Error:** `{str(e)}`")

    output_markdown.append("\n---\n")

    # 2. Parse Recent Commits
    if commits_raw:
        try:
            parsed_commits = (
                json.loads(commits_raw)
                if isinstance(commits_raw, str)
                else commits_raw
            )
            if (
                isinstance(parsed_commits, list)
                and len(parsed_commits) > 0
                and isinstance(parsed_commits[0], dict)
                and "text" in parsed_commits[0]
            ):
                parsed_commits = json.loads(parsed_commits[0]["text"])

            output_markdown.append("### 📝 Recent Commits (Latest Activity)\n")

            if isinstance(parsed_commits, list) and len(parsed_commits) > 0:
                commit_table = [
                    "| Commit Message | Author | Date | SHA |",
                    "| :--- | :--- | :--- | :--- |",
                ]
                for c in parsed_commits[:5]:
                    sha = c.get("sha", "")[:7]
                    commit_data = c.get("commit", {})
                    author_data = commit_data.get("author", {})

                    message = commit_data.get("message", "No commit message")
                    message = message.split("\n")[0].replace("|", "-")[:60]
                    author = author_data.get("name", "Unknown")
                    date = author_data.get("date", "")[:10]

                    commit_table.append(
                        f"| {message} | {author} | **{date}** | `{sha}` |"
                    )

                output_markdown.append("\n".join(commit_table))
            else:
                output_markdown.append("No recent commits found.")

        except Exception as e:
            output_markdown.append(f"**Commits Parsing Error:** `{str(e)}`")

    return "\n".join(output_markdown)


# Asynchronous Node to execute MCP Client
async def FetchGitHubMCPData(state: Dict[str, Any]) -> Dict[str, Any]:
    handle = state.get("github_handle")

    if not handle:
        return {
            "github_mcp_output": "GitHub analysis skipped: No GitHub handle detected in state."
        }

    token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    if not token:
        return {
            "github_mcp_output": "Error: GITHUB_PERSONAL_ACCESS_TOKEN is missing in environment variables."
        }

    # Extract resume skills safely using multi-key search and text parsing fallback
    resume_skills = extract_skills_from_state(state)

    mcp_config = {
        "github": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": token},
            "transport": "stdio",
        }
    }

    try:
        client = MultiServerMCPClient(mcp_config)
        tools = await client.get_tools()

        search_repo_tool = next(
            (t for t in tools if t.name == "search_repositories"), None
        )
        list_commits_tool = next(
            (t for t in tools if t.name == "list_commits"), None
        )

        repos_res = ""
        commits_res = ""
        recent_repo = None

        if search_repo_tool:
            repos_res = await search_repo_tool.ainvoke(
                {
                    "query": f"user:{handle} pushed:>2026-01-01",
                    "sort": "updated",
                    "order": "desc",
                    "per_page": 100,
                }
            )

            try:
                parsed = (
                    json.loads(repos_res)
                    if isinstance(repos_res, str)
                    else repos_res
                )
                if (
                    isinstance(parsed, list)
                    and len(parsed) > 0
                    and "text" in parsed[0]
                ):
                    parsed = json.loads(parsed[0]["text"])

                items = (
                    parsed.get("items", [])
                    if isinstance(parsed, dict)
                    else parsed
                )

                if not items:
                    repos_res = await search_repo_tool.ainvoke(
                        {
                            "query": f"user:{handle}",
                            "sort": "updated",
                            "order": "desc",
                            "per_page": 100,
                        }
                    )
                    parsed = (
                        json.loads(repos_res)
                        if isinstance(repos_res, str)
                        else repos_res
                    )
                    if (
                        isinstance(parsed, list)
                        and len(parsed) > 0
                        and "text" in parsed[0]
                    ):
                        parsed = json.loads(parsed[0]["text"])
                    items = (
                        parsed.get("items", [])
                        if isinstance(parsed, dict)
                        else parsed
                    )

                if items:
                    items = sorted(
                        items,
                        key=lambda x: x.get("updated_at", ""),
                        reverse=True,
                    )
                    recent_repo = items[0].get("name")
            except Exception:
                pass

        if list_commits_tool and recent_repo:
            try:
                commits_res = await list_commits_tool.ainvoke(
                    {"owner": handle, "repo": recent_repo}
                )
            except Exception:
                pass

        formatted_tables = format_github_response(repos_res, commits_res, resume_skills)
        return {"github_mcp_output": formatted_tables}

    except Exception as e:
        return {"github_mcp_output": f"MCP Execution Error: {str(e)}"}


def run_github_mcp(state: Dict[str, Any]) -> Dict[str, Any]:
    return asyncio.run(FetchGitHubMCPData(state))
