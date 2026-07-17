"""
Daily vacancy tracker.

Reads config.yaml (list of sites to check), fetches each one, extracts job
listings (either via CSS selectors or an LLM fallback), compares against
data/seen.json from the previous run, and writes an updated dashboard to
docs/index.html.
"""

import hashlib
import html
import json
import os
import sys
from datetime import datetime, timezone

import requests
import yaml
from bs4 import BeautifulSoup

CONFIG_FILE = "config.yaml"
DATA_FILE = "data/seen.json"
DASHBOARD_FILE = "docs/index.html"
USER_AGENT = "Mozilla/5.0 (compatible; VacancyTracker/1.0)"


def load_config():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_seen():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_seen(data):
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def make_id(title, url):
    return hashlib.sha256(f"{title}|{url}".encode("utf-8")).hexdigest()[:16]


def resolve_url(base_url, link):
    if not link:
        return None
    if link.startswith("http://") or link.startswith("https://"):
        return link
    from urllib.parse import urljoin
    return urljoin(base_url, link)


def fetch_html(url, use_browser=False):
    """Fetch a page's HTML. If use_browser is True, renders the page with a
    headless browser first -- needed for sites where listings are loaded by
    JavaScript after the initial page load, rather than present in the raw
    HTML response."""
    if use_browser:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(user_agent=USER_AGENT)
            page.goto(url, wait_until="load", timeout=45000)
            page.wait_for_timeout(4000)  # let JS finish populating listings
            content = page.content()
            browser.close()
        return content
    else:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
        resp.raise_for_status()
        return resp.text


def fetch_css(site):
    """Extract listings using CSS selectors defined in config.yaml."""
    page_html = fetch_html(site["url"], use_browser=site.get("render", False))
    soup = BeautifulSoup(page_html, "html.parser")

    sel = site["selectors"]
    listings = []
    for item in soup.select(sel["item"]):
        title_sel = sel["title"]
        link_sel = sel.get("link", title_sel)
        title_el = item if title_sel == "self" else item.select_one(title_sel)
        link_el = item if link_sel == "self" else item.select_one(link_sel)
        title = title_el.get_text(strip=True) if title_el else None
        href = link_el.get("href") if link_el else None
        url = resolve_url(site["url"], href)
        if title and url:
            listings.append({"title": title, "url": url})
    return listings


def fetch_llm(site):
    """Extract listings by asking an LLM to read the page text. Requires
    ANTHROPIC_API_KEY to be set as an environment variable / repo secret."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    page_html = fetch_html(site["url"], use_browser=site.get("render", False))
    soup = BeautifulSoup(page_html, "html.parser")

    # Build "link text | URL" pairs rather than stripping all HTML down to
    # plain text -- the model needs real URLs to work with, it can't
    # reliably invent or reconstruct them from visible text alone.
    links = []
    for a in soup.find_all("a", href=True):
        text = a.get_text(" ", strip=True)
        href = a["href"]
        if text and href and not href.startswith("#") and not href.lower().startswith("javascript:"):
            links.append(f"{text} | {href}")
    links_text = "\n".join(links)[:15000]

    prompt = (
        "Below is a list of link text and URL pairs extracted from a careers "
        "page, one per line, in the format 'link text | url'. Identify which "
        "lines represent actual job vacancy postings (ignore navigation, "
        "social media, cookie/privacy, login, and any other unrelated links). "
        "Return ONLY a JSON array (no markdown, no commentary) of objects "
        "with 'title' and 'url' fields, using the exact URL given on that "
        "line (do not modify or invent URLs).\n\n" + links_text
    )

    api_resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-sonnet-5",
            "max_tokens": 8000,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    api_resp.raise_for_status()
    content_blocks = api_resp.json().get("content", [])
    text = next((b["text"] for b in content_blocks if b.get("type") == "text"), None)
    if text is None:
        raise RuntimeError("No text content returned by the API")
    text = text.strip()

    # The model is asked to return only a JSON array, but sometimes adds
    # commentary or markdown fences around it anyway. Pull out just the
    # array portion rather than assuming the whole response is clean JSON.
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise RuntimeError(f"No JSON array found in model response: {text[:200]!r}")
    json_str = text[start:end + 1]
    raw_listings = json.loads(json_str)
    listings = []
    for item in raw_listings:
        title = item.get("title")
        url = resolve_url(site["url"], item.get("url"))
        if title and url:
            listings.append({"title": title, "url": url})
    return listings


def fetch_site(site):
    """Fetch one site. Supports either a single 'url' or a list of 'urls'
    (for sites that paginate listings across multiple pages) -- results
    from every page are merged together under this site's name."""
    urls = site.get("urls") or [site["url"]]
    all_listings = []
    for url in urls:
        page_site = dict(site)
        page_site["url"] = url
        if site["type"] == "css":
            all_listings.extend(fetch_css(page_site))
        elif site["type"] == "llm":
            all_listings.extend(fetch_llm(page_site))
        else:
            raise ValueError(f"Unknown site type: {site['type']}")
    return all_listings


def generate_dashboard(results, generated_at):
    rows = []
    for r in results:
        status_badge = (
            f'<span class="err">error: {html.escape(r["error"])}</span>'
            if r.get("error")
            else f'{len(r["listings"])} listing(s)'
        )
        listing_rows = ""
        for listing in r.get("listings", []):
            is_new = listing["id"] in r.get("new_ids", [])
            badge = '<span class="new">NEW</span> ' if is_new else ""
            listing_rows += (
                f'<li>{badge}<a href="{html.escape(listing["url"])}" '
                f'target="_blank" rel="noopener">{html.escape(listing["title"])}</a></li>\n'
            )
        rows.append(
            f'<section class="site">'
            f'<h2>{html.escape(r["name"])} <small>({status_badge})</small></h2>'
            f'<ul>{listing_rows or "<li><em>No listings found</em></li>"}</ul>'
            f"</section>"
        )

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Vacancy tracker</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body {{ font-family: -apple-system, Arial, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; color: #222; }}
h1 {{ font-size: 20px; }}
.site {{ margin-bottom: 28px; }}
.site h2 {{ font-size: 16px; border-bottom: 1px solid #ddd; padding-bottom: 6px; }}
.site small {{ color: #777; font-weight: normal; }}
ul {{ padding-left: 20px; }}
li {{ margin-bottom: 6px; }}
.new {{ background: #fdecea; color: #b3261e; font-size: 11px; font-weight: bold; padding: 2px 6px; border-radius: 4px; margin-right: 6px; }}
.err {{ color: #b3261e; }}
footer {{ color: #999; font-size: 12px; margin-top: 40px; }}
</style>
</head>
<body>
<h1>Vacancy tracker</h1>
<footer>Last checked: {html.escape(generated_at)}</footer>
{"".join(rows)}
<footer>Generated automatically. NEW badges clear on the next run once a listing has been seen once.</footer>
</body>
</html>
"""
    os.makedirs(os.path.dirname(DASHBOARD_FILE), exist_ok=True)
    with open(DASHBOARD_FILE, "w", encoding="utf-8") as f:
        f.write(html_doc)


def main():
    config = load_config()
    seen = load_seen()
    results = []

    for site in config["sites"]:
        name = site["name"]
        print(f"Checking {name}...")
        try:
            listings = fetch_site(site)
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            results.append({"name": name, "error": str(exc), "listings": [], "new_ids": []})
            continue

        site_seen = seen.get(name, {})
        current_ids = {}
        new_ids = []
        for listing in listings:
            uid = make_id(listing["title"], listing["url"])
            listing["id"] = uid
            current_ids[uid] = {"title": listing["title"], "url": listing["url"]}
            if uid not in site_seen:
                new_ids.append(uid)

        seen[name] = current_ids
        results.append({"name": name, "listings": listings, "new_ids": new_ids})
        print(f"  {len(listings)} listing(s), {len(new_ids)} new")

    save_seen(seen)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    generate_dashboard(results, generated_at)
    print("Done.")


if __name__ == "__main__":
    main()
