# Essence Scholar CLI

CLI reimplementation of the [Essence Scholar](https://github.com/sasi2400/essence-scholar-extension-public) Chrome extension. Analyzes academic papers using a multi-model backend (Gemini, OpenAI, Claude).

## How it works

1. **Text extraction**: Uses Playwright (stealth mode) to load web pages and extract text. For arXiv URLs, rewrites to PDF URL and downloads directly. Local PDFs extracted via pdfminer.six.
2. **Backend**: Sends text to the Essence Scholar Cloud Run backend (`POST /analyze-stream`), which proxies the request to your chosen LLM provider using your own API key.
3. **Streaming**: Response arrives as Server-Sent Events (SSE), parsed and concatenated in real time.
4. **Output**: Saves a Markdown file with title, source URL, analysis, and date.

## Setup

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

### API key

You must provide your own LLM API key. The backend proxies requests to the provider you choose — you pay for your own usage.

```bash
python summarize.py --setup-key
```

Keys are stored in `.auth_cache` (git-ignored).

| Provider | Models | Get a key |
|----------|--------|-----------|
| Google Gemini | `gemini-2.5-flash` (default), `gemini-2.5-pro` | https://aistudio.google.com/apikey |
| OpenAI | `gpt-4`, `gpt-4o-mini` | https://platform.openai.com/api-keys |
| Anthropic | `claude-3-5-sonnet-latest`, `claude-3-5-haiku-latest` | https://console.anthropic.com |

## Usage

```bash
python summarize.py <url_or_pdf>
python summarize.py <url_or_pdf> --model gpt-4o-mini
python summarize.py <url_or_pdf> --model claude-3-5-sonnet-latest
python summarize.py <url_or_pdf> --scholar-url "https://scholar.google.com/citations?user=..."
python summarize.py <url_or_pdf> --interests "behavioral economics, game theory"
python summarize.py <url_or_pdf> --output analysis.md
python summarize.py --debug
python summarize.py --proxy / --no-proxy
```

### Personalization

Essence Scholar can tailor its analysis to your research profile:

- `--scholar-url`: Your Google Scholar profile URL
- `--interests`: Comma-separated research interests

## Known limitations

- **API key always required**: The extension labels Gemini models as "Free" but this refers to Google's free tier — the backend still requires you to supply your own Gemini API key. Get one at https://aistudio.google.com/apikey (free: 15 RPM, 1M tokens/day). OpenAI and Claude keys require paid accounts.
- **Backend dependency**: Relies on the Essence Scholar Cloud Run service (`ssrn-summarizer-backend-v1-6-1`). If the service goes offline or the URL changes, the CLI will break.
- **SSE parsing**: The streaming response format may change across backend versions.
- **Academic focus**: Designed for academic papers (SSRN, arXiv, journal articles). May produce suboptimal results on non-academic content.

## Proxy support

Reads proxy configuration from `~/.scholar-proxies.json`:

```json
{
  "enabled": true,
  "proxies": ["http://proxy1:8080", "http://proxy2:8080"]
}
```

Override with `--proxy` or `--no-proxy` flags.
