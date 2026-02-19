#!/usr/bin/env python3
"""
Essence Scholar Summarizer

Analyze academic papers using the Essence Scholar Cloud Run backend.
Uses Playwright for stealth content extraction. Requires a user-supplied
LLM API key (Google Gemini, OpenAI, or Anthropic Claude).

Accepts either a URL or a local PDF file.

Usage:
    python summarize.py <url_or_pdf> [--output <path>] [--model <model>]
    python summarize.py https://arxiv.org/abs/1706.03762
    python summarize.py paper.pdf --model gpt-4o-mini
    python summarize.py --setup-key
"""

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

BACKEND_URL = "https://ssrn-summarizer-backend-v1-6-1-pisqy7uvxq-uc.a.run.app"
ANALYZE_ENDPOINT = f"{BACKEND_URL}/analyze-stream"
AUTH_CACHE_FILE = Path(__file__).parent / ".auth_cache"

MODELS = {
    "gemini-2.5-flash": "gemini",
    "gemini-2.5-pro": "gemini",
    "gpt-4": "openai",
    "gpt-4o-mini": "openai",
    "claude-3-5-sonnet-latest": "claude",
    "claude-3-5-haiku-latest": "claude",
}
DEFAULT_MODEL = "gemini-2.5-flash"

# Rate limiting: track requests in a local file
RATE_LIMIT_FILE = Path(__file__).parent / ".rate_limit_log"
MIN_INTERVAL_SECONDS = 5
MAX_REQUESTS_PER_HOUR = 10


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

def check_rate_limit():
    """Enforce rate limits."""
    now = time.time()
    window = 3600

    timestamps = []
    if RATE_LIMIT_FILE.exists():
        try:
            timestamps = json.loads(RATE_LIMIT_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            timestamps = []

    timestamps = [ts for ts in timestamps if now - ts < window]

    if timestamps:
        elapsed = now - max(timestamps)
        if elapsed < MIN_INTERVAL_SECONDS:
            wait = MIN_INTERVAL_SECONDS - elapsed
            print(f"Rate limit: waiting {wait:.1f}s between requests...")
            time.sleep(wait)

    if len(timestamps) >= MAX_REQUESTS_PER_HOUR:
        oldest = min(timestamps)
        retry_after = window - (now - oldest)
        print(
            f"Rate limit reached ({MAX_REQUESTS_PER_HOUR} requests/hour). "
            f"Try again in {retry_after / 60:.0f} minute(s).",
            file=sys.stderr,
        )
        sys.exit(1)


def record_request():
    """Record a request timestamp for rate limiting."""
    now = time.time()
    window = 3600

    timestamps = []
    if RATE_LIMIT_FILE.exists():
        try:
            timestamps = json.loads(RATE_LIMIT_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            timestamps = []

    timestamps = [ts for ts in timestamps if now - ts < window]
    timestamps.append(now)
    RATE_LIMIT_FILE.write_text(json.dumps(timestamps))


# ---------------------------------------------------------------------------
# Proxy rotation
# ---------------------------------------------------------------------------

_PROXY_FORCE: bool | None = None


def _load_proxy_config() -> dict:
    config_path = Path.home() / ".scholar-proxies.json"
    try:
        return json.loads(config_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"enabled": False, "proxies": []}


def _get_proxy_url() -> str | None:
    config = _load_proxy_config()
    enabled = _PROXY_FORCE if _PROXY_FORCE is not None else config.get("enabled", False)
    if not enabled:
        return None
    proxies = config.get("proxies", [])
    if not proxies:
        return None
    return random.choice(proxies)


def _get_proxy() -> dict | None:
    url = _get_proxy_url()
    if not url:
        return None
    return {"http": url, "https": url}


# ---------------------------------------------------------------------------
# API key management
# ---------------------------------------------------------------------------

def _load_auth_cache() -> dict:
    if not AUTH_CACHE_FILE.exists():
        return {}
    try:
        return json.loads(AUTH_CACHE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_auth_cache(data: dict):
    data["cached_at"] = time.time()
    AUTH_CACHE_FILE.write_text(json.dumps(data, indent=2))


def setup_api_key() -> dict:
    """Prompt user for LLM API key(s) and store them."""
    cache = _load_auth_cache()

    print("Essence Scholar requires an LLM API key.")
    print("Supported providers: Google Gemini, OpenAI, Anthropic Claude")
    print()
    print("Models available:")
    print("  Gemini:    gemini-2.5-flash (default), gemini-2.5-pro")
    print("  OpenAI:    gpt-4, gpt-4o-mini")
    print("  Anthropic: claude-3-5-sonnet-latest, claude-3-5-haiku-latest")
    print()

    print("Enter API keys (press Enter to skip a provider):")
    gemini_key = input("  Gemini API key (https://aistudio.google.com/apikey): ").strip()
    openai_key = input("  OpenAI API key (https://platform.openai.com/api-keys): ").strip()
    claude_key = input("  Claude API key (https://console.anthropic.com): ").strip()

    if gemini_key:
        cache["gemini_key"] = gemini_key
    if openai_key:
        cache["openai_key"] = openai_key
    if claude_key:
        cache["claude_key"] = claude_key

    if not any([gemini_key, openai_key, claude_key]) and not any(
        cache.get(k) for k in ["gemini_key", "openai_key", "claude_key"]
    ):
        print("No API keys provided.", file=sys.stderr)
        sys.exit(1)

    _save_auth_cache(cache)
    print("API key(s) saved.")
    return cache


def get_api_keys(model: str) -> dict:
    """Get the required API key for the chosen model."""
    cache = _load_auth_cache()
    provider = MODELS.get(model)

    key_map = {
        "gemini": "gemini_key",
        "openai": "openai_key",
        "claude": "claude_key",
    }

    key_field = key_map.get(provider)
    if key_field and cache.get(key_field):
        return cache

    print(f"No API key found for {provider} (model: {model}).")
    return setup_api_key()


# ---------------------------------------------------------------------------
# Stealth Playwright browser
# ---------------------------------------------------------------------------

def _launch_browser(p):
    proxy_url = _get_proxy_url()
    browser = p.chromium.launch(
        headless=True,
        proxy={"server": proxy_url} if proxy_url else None,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-extensions",
        ],
    )

    context = browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        viewport={"width": 1280, "height": 720},
        locale="en-US",
        timezone_id="America/New_York",
        java_script_enabled=True,
    )

    page = context.new_page()

    page.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        Object.defineProperty(navigator, 'plugins', {
            get: () => [1, 2, 3, 4, 5],
        });
        Object.defineProperty(navigator, 'languages', {
            get: () => ['en-US', 'en'],
        });
        window.chrome = { runtime: {} };
    """)

    return browser, context, page


def _wait_for_cloudflare(page, timeout_seconds: int = 30):
    start = time.time()
    while time.time() - start < timeout_seconds:
        title = page.title().lower()
        body_text = page.text_content("body") or ""
        if "just a moment" in title or "checking your browser" in body_text.lower():
            print("Cloudflare challenge detected, waiting for it to resolve...")
            time.sleep(2)
            continue
        break


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------

def _rewrite_to_pdf_url(url: str) -> str | None:
    import re as _re
    m = _re.match(r"https?://arxiv\.org/abs/(.+?)(?:\?.*)?$", url)
    if m:
        return f"https://arxiv.org/pdf/{m.group(1)}"
    m = _re.match(r"https?://arxiv\.org/html/(.+?)(?:\?.*)?$", url)
    if m:
        return f"https://arxiv.org/pdf/{m.group(1)}"
    return None


def _download_pdf_text(url: str) -> tuple[str, str | None]:
    import tempfile
    from pdfminer.high_level import extract_text

    print(f"Downloading PDF: {url}")
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60, proxies=_get_proxy())
    if not resp.ok:
        print(f"Failed to download PDF (HTTP {resp.status_code})", file=sys.stderr)
        sys.exit(1)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(resp.content)
        tmp_path = f.name
    try:
        text = extract_text(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return text.strip(), None


def _extract_text_from_url(url: str) -> tuple[str, str | None]:
    pdf_url = _rewrite_to_pdf_url(url)
    if pdf_url:
        return _download_pdf_text(pdf_url)

    print(f"Extracting content from URL: {url}")
    with sync_playwright() as p:
        browser, context, page = _launch_browser(p)

        page.goto(url, wait_until="networkidle", timeout=60000)
        _wait_for_cloudflare(page)

        page_title = page.evaluate("() => document.title || null")
        text = page.evaluate("() => document.body.innerText")

        browser.close()

        if not text or len(text) < 50:
            print("Warning: extracted very little text from the page.", file=sys.stderr)

        return text, page_title


def _extract_text_from_pdf(pdf_path: str) -> str:
    from pdfminer.high_level import extract_text

    pdf_file = Path(pdf_path).resolve()
    if not pdf_file.exists():
        print(f"File not found: {pdf_file}", file=sys.stderr)
        sys.exit(1)
    if pdf_file.stat().st_size > 50 * 1024 * 1024:
        print("File too large (>50MB).", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting text from PDF: {pdf_file.name}")
    text = extract_text(str(pdf_file))

    if not text or len(text.strip()) < 50:
        print("Warning: extracted very little text from the PDF.", file=sys.stderr)

    return text.strip()


# ---------------------------------------------------------------------------
# Essence Scholar API call (SSE streaming)
# ---------------------------------------------------------------------------

def _parse_sse_stream(response) -> str:
    """Parse Server-Sent Events stream and concatenate text chunks."""
    full_text = []
    buffer = ""

    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
        buffer += chunk
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    if isinstance(data, dict):
                        # Handle status events
                        status = data.get("status", "")
                        if status == "error":
                            msg = data.get("message", "Unknown error")
                            print(f"\nServer error: {msg}", file=sys.stderr)
                            sys.exit(1)
                        if status == "starting":
                            print(f"  {data.get('message', '')}")
                            continue
                        if status == "progress":
                            step = data.get("step", "")
                            print(f"  [{step}] {data.get('message', '')}")
                            continue

                        # Extract text content from various possible fields
                        text = data.get("text", "") or data.get("content", "") or data.get("chunk", "")
                        if text:
                            full_text.append(text)
                            print(".", end="", flush=True)
                except json.JSONDecodeError:
                    full_text.append(data_str)

    print()  # newline after progress dots
    return "".join(full_text)


def summarize_content(
    content: str,
    url: str,
    model: str,
    api_keys: dict,
    scholar_url: str = "",
    research_interests: str = "",
    debug: bool = False,
) -> str:
    """Call the Essence Scholar analyze-stream endpoint."""
    provider = MODELS.get(model)

    payload = {
        "content": {"paperUrl": url, "text": content},
        "model": model,
        "user_scholar_url": scholar_url,
        "research_interests": research_interests,
    }

    # Add the appropriate API key
    if provider == "gemini" and api_keys.get("gemini_key"):
        payload["google_api_key"] = api_keys["gemini_key"]
    elif provider == "openai" and api_keys.get("openai_key"):
        payload["openai_api_key"] = api_keys["openai_key"]
    elif provider == "claude" and api_keys.get("claude_key"):
        payload["claude_api_key"] = api_keys["claude_key"]

    print(f"Requesting analysis from Essence Scholar (model: {model})...")

    if debug:
        safe_payload = {k: v for k, v in payload.items()}
        for key in ["google_api_key", "openai_api_key", "claude_api_key"]:
            if key in safe_payload:
                safe_payload[key] = safe_payload[key][:8] + "..."
        print(f"Payload: {json.dumps(safe_payload, indent=2)}")

    resp = requests.post(
        ANALYZE_ENDPOINT,
        json=payload,
        headers={
            "Content-Type": "application/json",
            "X-Extension-Version": "v1.7.0",
            "Accept": "text/event-stream",
        },
        timeout=180,
        stream=True,
        proxies=_get_proxy(),
    )

    if debug:
        print(f"Response status: {resp.status_code}")

    if resp.status_code == 429:
        print("Error: rate limited by Essence Scholar. Try again later.", file=sys.stderr)
        sys.exit(1)

    if not resp.ok:
        error_text = ""
        try:
            error_text = resp.text[:500]
        except Exception:
            pass
        print(f"API error (HTTP {resp.status_code}): {error_text}", file=sys.stderr)
        sys.exit(1)

    # Check if response is SSE or plain JSON
    content_type = resp.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        summary = _parse_sse_stream(resp)
    else:
        data = resp.json()
        if debug:
            print(json.dumps(data, indent=2, ensure_ascii=False))
        summary = data.get("summary", "") or data.get("analysis", "") or data.get("text", "")
        if isinstance(summary, dict):
            summary = json.dumps(summary, indent=2, ensure_ascii=False)

    if not summary:
        print("Error: no analysis received from server.", file=sys.stderr)
        sys.exit(1)

    return summary


# ---------------------------------------------------------------------------
# Markdown conversion
# ---------------------------------------------------------------------------

def to_markdown(summary: str, url: str, model: str, page_title: str | None = None) -> str:
    lines = []

    title = page_title or "Untitled Paper"
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**Source:** {url}")
    lines.append(f"**Model:** {model}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Analysis")
    lines.append("")
    lines.append(summary.strip())
    lines.append("")
    lines.append("---")
    lines.append(f"*Generated by Essence Scholar on {datetime.now().strftime('%Y-%m-%d')}*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    slug = text.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug[:80] if slug else "summary"


def _is_local_file(input_str: str) -> bool:
    if input_str.startswith(("http://", "https://")):
        return False
    return Path(input_str).expanduser().exists()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze academic papers using Essence Scholar"
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="URL of the paper or path to a local PDF",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (default: auto-generated from title)",
    )
    parser.add_argument(
        "--model", "-m",
        choices=list(MODELS.keys()),
        default=DEFAULT_MODEL,
        help=f"LLM model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--scholar-url",
        default="",
        help="Your Google Scholar profile URL (for personalized analysis)",
    )
    parser.add_argument(
        "--interests",
        default="",
        help="Your research interests (comma-separated)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Dump raw API request/response details",
    )
    parser.add_argument(
        "--setup-key",
        action="store_true",
        help="Set up or replace LLM API key(s)",
    )
    proxy_group = parser.add_mutually_exclusive_group()
    proxy_group.add_argument("--proxy", dest="use_proxy", action="store_true", default=None,
                             help="Force proxy usage (overrides ~/.scholar-proxies.json)")
    proxy_group.add_argument("--no-proxy", dest="use_proxy", action="store_false",
                             help="Disable proxy (overrides ~/.scholar-proxies.json)")
    args = parser.parse_args()

    global _PROXY_FORCE
    _PROXY_FORCE = args.use_proxy

    if args.setup_key:
        setup_api_key()
        if not args.input:
            return

    if not args.input:
        parser.error("Please provide a URL or PDF file path (or use --setup-key)")

    check_rate_limit()

    # Get API keys for the chosen model
    api_keys = get_api_keys(args.model)

    # Extract content
    page_title = None
    if _is_local_file(args.input):
        content = _extract_text_from_pdf(args.input)
        url = f"file://{Path(args.input).resolve()}"
    else:
        content, page_title = _extract_text_from_url(args.input)
        url = args.input

    # Call Essence Scholar API
    summary = summarize_content(
        content, url, args.model, api_keys,
        scholar_url=args.scholar_url,
        research_interests=args.interests,
        debug=args.debug,
    )
    record_request()

    # Convert to Markdown
    md = to_markdown(summary, url, args.model, page_title=page_title)

    # Write output
    if args.output:
        out_path = Path(args.output)
    else:
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        slug = slugify(page_title or "summary")
        out_path = output_dir / f"{slug}.md"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"Analysis saved to: {out_path}")


if __name__ == "__main__":
    main()
