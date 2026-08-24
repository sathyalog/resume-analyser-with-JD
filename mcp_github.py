import asyncio
import json
import os
from typing import Any, Dict
from langchain_mcp_adapters.client import MultiServerMCPClient


def format_github_response(repos_raw: Any, commits_raw: Any) -> str:
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

            # Force client-side sorting on updated_at date to surface 2026 repos first
            items = sorted(
                items, key=lambda x: x.get("updated_at", ""), reverse=True
            )

            output_markdown.append(
                f"### 📦 Public Repositories (Total: {total_count})\n"
            )

            if items:
                repo_table = [
                    "| Repository | Description | Primary Language | Stars | Last Updated |",
                    "| :--- | :--- | :--- | :--- | :--- |",
                ]
                for repo in items[:10]:  # Top 10 most recently updated
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

        # 1. Search Repositories — sort by updated descending
        if search_repo_tool:
            repos_res = await search_repo_tool.ainvoke(
                {
                    "query": f"user:{handle} pushed:>2026-01-01",  # Search specifically for active 2026 pushes
                    "sort": "updated",
                    "order": "desc",
                    "per_page": 100,
                }
            )

            # Fallback if no 2026 query matches yet in GitHub search index
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
                    # Query without push date filter if search index is still updating
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

        # 2. Fetch commits from the most active repository
        if list_commits_tool and recent_repo:
            try:
                commits_res = await list_commits_tool.ainvoke(
                    {"owner": handle, "repo": recent_repo}
                )
            except Exception:
                pass

        formatted_tables = format_github_response(repos_res, commits_res)
        return {"github_mcp_output": formatted_tables}

    except Exception as e:
        return {"github_mcp_output": f"MCP Execution Error: {str(e)}"}


def run_github_mcp(state: Dict[str, Any]) -> Dict[str, Any]:
    return asyncio.run(FetchGitHubMCPData(state))
