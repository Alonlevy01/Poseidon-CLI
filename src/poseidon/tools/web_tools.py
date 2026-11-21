"""Read-only web search (sanctioned APIs) and SSRF-safe URL fetch tools for Poseidon."""

from __future__ import annotations

import html
import ipaddress
import os
import re
import socket
from urllib.parse import urlparse

import httpx

from poseidon.tools.registry import ToolDefinition, ToolExecutionResult, ToolRegistry

USER_AGENT = "Poseidon-CLI/0.5 (https://github.com/Alonlevy01/Poseidon-CLI)"
MAX_FETCH_CHARS = 8000
_SCRIPT_STYLE_RE = re.compile(r"<(script|style|noscript|svg)[^>]*>.*?</\1>", re.I | re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def _validate_public_url(raw_url: str) -> str:
    """Ensure URL is http(s) and does not resolve to loopback, private, or link-local IPs."""
    url = raw_url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only public http:// or https:// URLs are allowed.")
    host = parsed.hostname.lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        raise ValueError(f"Blocked non-public host: {host}")
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve host '{host}': {exc}") from exc

    for info in infos:
        ip_str = str(info[4][0])
        ip = ipaddress.ip_address(ip_str)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise ValueError(f"Blocked non-public IP address ({ip_str}) for host '{host}'.")
    return url


def html_to_clean_text(raw_html: str) -> str:
    """Convert raw HTML into concise readable plain text without script/style noise."""
    cleaned = _SCRIPT_STYLE_RE.sub(" ", raw_html)
    cleaned = re.sub(r"</?(p|div|br|li|tr|h[1-6]|pre|code|section|article)[^>]*>", "\n", cleaned)
    cleaned = _TAG_RE.sub(" ", cleaned)
    cleaned = html.unescape(cleaned)
    lines = [_WS_RE.sub(" ", line).strip() for line in cleaned.splitlines()]
    text = "\n".join(line for line in lines if line)
    return _BLANK_LINES_RE.sub("\n\n", text).strip()


def run_web_search(
    query: str,
    *,
    source: str = "auto",
    max_results: int = 5,
    tavily_api_key: str = "",
    brave_api_key: str = "",
) -> list[dict[str, str]]:
    """Query Tavily/Brave if configured, else sanctioned developer APIs (SO, GitHub, PyPI, Wiki)."""
    q = query.strip()
    if not q:
        raise ValueError("Search query cannot be empty.")
    limit = max(1, min(int(max_results), 8))
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}

    t_key = tavily_api_key or os.environ.get("TAVILY_API_KEY", "")
    b_key = brave_api_key or os.environ.get("BRAVE_API_KEY", "")

    with httpx.Client(timeout=12.0, headers=headers, follow_redirects=True) as client:
        # 1. Optional Tavily Search API (if user configured TAVILY_API_KEY)
        if source == "auto" and t_key:
            resp = client.post(
                "https://api.tavily.com/search",
                json={"api_key": t_key, "query": q, "max_results": limit},
            )
            resp.raise_for_status()
            return [
                {
                    "source": "tavily",
                    "title": str(item.get("title") or ""),
                    "url": str(item.get("url") or ""),
                    "snippet": str(item.get("content") or "")[:280],
                }
                for item in resp.json().get("results", [])[:limit]
            ]

        # 2. Optional Brave Search API (if user configured BRAVE_API_KEY)
        if source == "auto" and b_key:
            resp = client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": q, "count": limit},
                headers={"X-Subscription-Token": b_key},
            )
            resp.raise_for_status()
            return [
                {
                    "source": "brave",
                    "title": str(item.get("title") or ""),
                    "url": str(item.get("url") or ""),
                    "snippet": str(item.get("description") or "")[:280],
                }
                for item in resp.json().get("web", {}).get("results", [])[:limit]
            ]

        results: list[dict[str, str]] = []
        chosen = source.lower().strip() or "auto"

        # 3. PyPI exact package lookup
        if chosen == "pypi":
            pkg = re.sub(r"[^a-zA-Z0-9._-]+", "", q.split()[0])
            resp = client.get(f"https://pypi.org/pypi/{pkg}/json")
            if resp.status_code == 200:
                info = resp.json().get("info", {})
                results.append(
                    {
                        "source": "pypi",
                        "title": f"{info.get('name', pkg)} {info.get('version', '')}".strip(),
                        "url": str(info.get("project_url") or f"https://pypi.org/project/{pkg}/"),
                        "snippet": str(info.get("summary") or "")[:280],
                    }
                )
            return results

        # 4. StackOverflow public API (sanctioned, zero API key)
        if chosen in {"auto", "stackoverflow"}:
            try:
                so_limit = limit if chosen == "stackoverflow" else max(2, limit // 2)
                resp = client.get(
                    "https://api.stackexchange.com/2.3/search/advanced",
                    params={
                        "q": q,
                        "site": "stackoverflow",
                        "pagesize": so_limit,
                        "sort": "relevance",
                    },
                )
                if resp.status_code == 200:
                    for item in resp.json().get("items", [])[:so_limit]:
                        tags = ", ".join(item.get("tags", [])[:4])
                        answered = "answered" if item.get("is_answered") else "open"
                        results.append(
                            {
                                "source": "stackoverflow",
                                "title": html.unescape(str(item.get("title") or "")),
                                "url": str(item.get("link") or ""),
                                "snippet": (
                                    f"score={item.get('score', 0)} | {answered} | tags: {tags}"
                                ),
                            }
                        )
            except httpx.HTTPError:
                pass

        # 5. GitHub Repositories public API
        if chosen in {"auto", "github"}:
            try:
                gh_limit = limit if chosen == "github" else max(2, limit // 2)
                resp = client.get(
                    "https://api.github.com/search/repositories",
                    params={"q": q, "per_page": gh_limit, "sort": "stars"},
                )
                if resp.status_code == 200:
                    for item in resp.json().get("items", [])[:gh_limit]:
                        desc = str(item.get("description") or "").strip()
                        stars = item.get("stargazers_count", 0)
                        results.append(
                            {
                                "source": "github",
                                "title": str(item.get("full_name") or ""),
                                "url": str(item.get("html_url") or ""),
                                "snippet": f"★ {stars} | {desc[:220]}",
                            }
                        )
            except httpx.HTTPError:
                pass

        # 6. Wikipedia public API
        if chosen in {"auto", "wikipedia"} and len(results) < limit:
            try:
                wiki_limit = limit if chosen == "wikipedia" else max(1, limit - len(results))
                resp = client.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "list": "search",
                        "srsearch": q,
                        "format": "json",
                        "srlimit": wiki_limit,
                    },
                )
                if resp.status_code == 200:
                    for item in resp.json().get("query", {}).get("search", [])[:wiki_limit]:
                        title = str(item.get("title") or "")
                        slug = title.replace(" ", "_")
                        snippet = html_to_clean_text(str(item.get("snippet") or ""))
                        results.append(
                            {
                                "source": "wikipedia",
                                "title": title,
                                "url": f"https://en.wikipedia.org/wiki/{slug}",
                                "snippet": snippet[:240],
                            }
                        )
            except httpx.HTTPError:
                pass

        return results[:limit]


def fetch_public_url(raw_url: str, *, max_chars: int = MAX_FETCH_CHARS) -> tuple[str, str]:
    """Fetch a public http(s) URL safely (SSRF-checked) and return `(final_url, clean_text)`."""
    checked = _validate_public_url(raw_url)
    with httpx.Client(
        timeout=12.0,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
    ) as client:
        resp = client.get(checked)
        final_url = str(resp.url)
        _validate_public_url(final_url)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "").lower()
        body = resp.text
        text = html_to_clean_text(body) if "html" in content_type else body.strip()
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n... [truncated at {max_chars} chars]"
        return final_url, text


def register_web_tools(
    registry: ToolRegistry,
    *,
    tavily_api_key: str = "",
    brave_api_key: str = "",
) -> None:
    """Register read-only `web_search` and `fetch_url` tools on the registry."""

    def handle_web_search(args: dict[str, object]) -> ToolExecutionResult:
        query = str(args.get("query") or "").strip()
        source = str(args.get("source") or "auto").strip()
        max_results = int(args.get("max_results") or 5)
        try:
            hits = run_web_search(
                query,
                source=source,
                max_results=max_results,
                tavily_api_key=tavily_api_key,
                brave_api_key=brave_api_key,
            )
        except (ValueError, httpx.HTTPError) as exc:
            return ToolExecutionResult(
                tool_name="web_search",
                ok=False,
                output=f"Web search failed: {exc}",
                summary="web search error",
            )
        if not hits:
            return ToolExecutionResult(
                tool_name="web_search",
                ok=True,
                output=f"No web results found for {query!r} (source={source}).",
                summary=f"0 web hits for {query!r}",
            )
        lines = [
            f"{i}. [{h['source']}] {h['title']}\n   URL: {h['url']}\n   {h['snippet']}"
            for i, h in enumerate(hits, start=1)
        ]
        return ToolExecutionResult(
            tool_name="web_search",
            ok=True,
            output="\n".join(lines),
            summary=f"{len(hits)} web hit(s) for {query!r}",
        )

    def handle_fetch_url(args: dict[str, object]) -> ToolExecutionResult:
        url = str(args.get("url") or "").strip()
        try:
            final_url, text = fetch_public_url(url)
        except (ValueError, httpx.HTTPError) as exc:
            return ToolExecutionResult(
                tool_name="fetch_url",
                ok=False,
                output=f"Fetch blocked or failed: {exc}",
                summary="fetch_url error",
            )
        return ToolExecutionResult(
            tool_name="fetch_url",
            ok=True,
            output=f"URL: {final_url}\n\n{text}",
            summary=f"fetched {len(text)} chars from {urlparse(final_url).hostname}",
        )

    registry.register(
        ToolDefinition(
            name="web_search",
            description=(
                "Search public developer & documentation APIs (StackOverflow, GitHub, PyPI, "
                "Wikipedia; or Tavily/Brave if API key configured) for up-to-date answers."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "source": {
                        "type": "string",
                        "enum": ["auto", "stackoverflow", "github", "pypi", "wikipedia"],
                        "description": "Optional source filter (default: auto).",
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 8,
                        "description": "Maximum results to return (default: 5).",
                    },
                },
                "required": ["query"],
            },
            handler=handle_web_search,
        )
    )
    registry.register(
        ToolDefinition(
            name="fetch_url",
            description=(
                "Fetch a public http(s) documentation webpage and return clean readable text "
                "(SSRF-protected against local/private networks)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Public http:// or https:// URL."}
                },
                "required": ["url"],
            },
            handler=handle_fetch_url,
        )
    )
