#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import time
import hashlib
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from urllib.robotparser import RobotFileParser
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

DEFAULT_CONFIG = {
    "BLACKLIST_PATH_REGEX": [
        r"^sso/.*",
        r"^qisserver/.*",
        r"^kontaktformular/.*",
        # r"^admin/.*",
        # r"^wp-admin/.*",
    ],
    "MAX_PAGES": 2500,
    "TIMEOUT": 20,
    "USER_AGENT": "SimpleTextCrawler/1.0 (+https://example.org; contact=you@example.org)",
    "OUTPUT_DIR": "out",
    "START_URL": "https://www.uni-hildesheim.de/",
    "WHITELIST_PATH_REGEX":  [
        r"^$",
    ],
    "WORKERS": 8
}

config = DEFAULT_CONFIG.copy()

def load_config():
    # Open config (if exists) and update defaults
    if os.path.exists("config.json"):
        with open("config.json", "r", encoding="utf-8") as f:
            user_config = json.load(f)
            config.update(user_config)
    print_config(config)

def compile_patterns(patterns, name: str):
    compiled = []
    for p in patterns:
        try:
            compiled.append(re.compile(p))
        except re.error as e:
            print(f"Invalid Regex in {name}: {p!r} -> {e}")
    return compiled

def serialize_config(config: dict):
    output = {}
    for key, value in config.items():
        if isinstance(value, list):
            output[key] = [
                v.pattern if isinstance(v, re.Pattern) else v
                for v in value
            ]
        elif isinstance(value, re.Pattern):
            output[key] = value.pattern
        else:
            output[key] = value
    return output


def print_config(config: dict):
    RED = "\033[91m"
    RESET = "\033[0m"
    printable = serialize_config(config)
    print("\nActive Configuration:\n")
    for key, value in printable.items():
        print(f"{key}: {RED}{value}{RESET}")

def path_for_match(url: str) -> str:
    p = urlparse(url)
    return (p.path or "/").lstrip("/")  # "sso/..." statt "/sso/..."

def is_whitelisted(url: str, compiled_whitelist: list[re.Pattern]) -> bool:
    # Wenn keine Whitelist-Regeln angegeben sind: alles zulassen
    if not compiled_whitelist:
        return True
    path = path_for_match(url)
    return any(rx.search(path) for rx in compiled_whitelist)

def is_blacklisted(url: str, compiled_blacklist: list[re.Pattern]) -> bool:
    path = path_for_match(url)
    return any(rx.search(path) for rx in compiled_blacklist)

def is_allowed_url(
    url: str,
    start_netloc: str,
    rp: RobotFileParser,
    user_agent: str,
    whitelist_rx: list[re.Pattern],
    blacklist_rx: list[re.Pattern],
) -> bool:
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        return False
    if not same_site(start_netloc, p.netloc):
        return False

    # robots.txt
    try:
        if not rp.can_fetch(user_agent, url):
            return False
    except Exception:
        pass  # im Zweifel erlauben (wie bei dir)

    # whitelist / blacklist
    if not is_whitelisted(url, whitelist_rx):
        return False
    if is_blacklisted(url, blacklist_rx):
        return False

    return True

def norm_url(base: str, href: str) -> str | None:
    """Resolve + remove fragments + basic cleanup."""
    if not href:
        return None
    href = href.strip()
    # Skip obvious non-http(s)
    if href.startswith(("mailto:", "tel:", "javascript:", "data:")):
        return None

    absolute = urljoin(base, href)
    absolute, _frag = urldefrag(absolute)  # remove #fragment
    return absolute


def same_site(start_netloc: str, candidate_netloc: str) -> bool:
    """
    Allow same domain + subdomains of start host.
    Example: start=uni-hildesheim.de allows foo.uni-hildesheim.de
    """
    start_host = start_netloc.split(":")[0].lower()
    cand_host = candidate_netloc.split(":")[0].lower()
    return cand_host == start_host or cand_host.endswith("." + start_host)


def safe_filename(url: str, max_len: int = 180) -> str:
    """
    Make a filesystem-safe, length-limited filename.
    Keeps a short hash for uniqueness and truncates the slug part.
    """
    p = urlparse(url)
    netloc = (p.netloc or "site").lower()

    path = p.path.strip("/") or "index"
    slug = re.sub(r"[^a-zA-Z0-9\-_/]+", "-", path).strip("-")
    slug = slug.replace("/", "__")
    if not slug:
        slug = "index"

    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    ext = ".md"

    base = f"{netloc}__{slug}__{h}{ext}"

    # If still too long, truncate slug to fit.
    # Reserve space for: netloc + separators + hash + ext
    reserved = len(f"{netloc}__") + len(f"__{h}") + len(ext)
    max_slug_len = max(20, max_len - reserved)

    if len(base) > max_len:
        slug = slug[:max_slug_len].rstrip("-_")
        base = f"{netloc}__{slug}__{h}{ext}"

    return base


def build_robot_parser(start_url: str) -> RobotFileParser:
    p = urlparse(start_url)
    robots_url = f"{p.scheme}://{p.netloc}/robots.txt"
    rp = RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
    except Exception:
        # If robots.txt cannot be fetched, "simple" behavior: treat as allowed.
        # You could change this to "disallow all" if you prefer conservative.
        pass
    return rp


def extract_main_markdown(html: str, page_url: str) -> tuple[str, str]:
    """
    Return (head_title, markdown_content).
    Keeps headings, lists, emphasis, strong, links etc (via markdownify).
    """
    soup = BeautifulSoup(html, "html.parser")
    
    # remove unwanted blocks
    selectors = [
        "div#breadcrumbs.container",
        "div#cookie-consent",
        "div.sitemap-menu",
        "aside.sidebar",
        "aside#secondary.widget-area",
        "a#totop.hidden",
        "div.navigation",
        "header",
        "li.nav-item",
        "ul.submenu-dropdown",
        "nav"
    ]

    # remove noise
    for tag in soup(["script", "style", "noscript", "template", "svg"]):
        tag.decompose()

    main = soup.find("main")
    if main is None:
        main = soup.body or soup
        
    for el in main.select(",".join(selectors)):
        el.decompose()

    # EXCLUDE all footers
    for ft in main.find_all("footer"):
        ft.decompose()
        
    for h in main.select("h1,h2,h3,h4,h5,h6"):
        # get_text strips all tags; if there's no visible text, drop the heading
        if not h.get_text(" ", strip=True):
            h.decompose()

    head_title = ""
    if main.title and main.title.string:
        head_title = main.title.string.strip()

    # Convert HTML->Markdown.
    # markdownify keeps <strong>/<em>, lists, headings etc.
    content_md = md(
        str(main),
        heading_style="ATX",
        bullets="-",
        strip=["img", "video", "audio", "iframe", "object", "embed"],
    )

    # Clean up markdown a bit
    content_md = re.sub(r"\n{3,}", "\n\n", content_md).strip()

    return head_title, content_md


# --- threading helpers ---
_thread_local = threading.local()

def get_session() -> requests.Session:
    """One requests.Session per thread."""
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        sess.headers.update({"User-Agent": config["USER_AGENT"]})
        _thread_local.session = sess
    return sess

def crawl(start_url: str, workers: int = 8):
    # Compile regex patterns
    blacklist_rx = compile_patterns(config["BLACKLIST_PATH_REGEX"], "BLACKLIST_PATH_REGEX")
    whitelist_rx = compile_patterns(config["WHITELIST_PATH_REGEX"], "WHITELIST_PATH_REGEX")

    print(f"\nStart crawling with {config['WORKERS']} workers at {config['MAX_PAGES']} pages.")
    os.makedirs(config["OUTPUT_DIR"], exist_ok=True)

    rp = build_robot_parser(start_url)
    try:
        crawl_delay = rp.crawl_delay(config["USER_AGENT"])
    except Exception:
        crawl_delay = None
    if crawl_delay is None:
        crawl_delay = 1.0

    start = urlparse(start_url)
    start_netloc = start.netloc

    q: "queue.Queue[str]" = queue.Queue()
    q.put(start_url)

    seen: set[str] = set()
    in_progress: set[str] = set()
    seen_lock = threading.Lock()

    pages = 0
    pages_lock = threading.Lock()

    # simple global per-domain politeness: one request at a time + delay
    rate_lock = threading.Lock()
    next_allowed_time = 0.0

    def rate_limit():
        nonlocal next_allowed_time
        with rate_lock:
            now = time.monotonic()
            if now < next_allowed_time:
                time.sleep(next_allowed_time - now)
            next_allowed_time = time.monotonic() + float(crawl_delay)

    def should_take(url: str) -> bool:
        if not is_allowed_url(url, start_netloc, rp, config["USER_AGENT"], whitelist_rx, blacklist_rx):
            return False

        with seen_lock:
            if url in seen or url in in_progress:
                return False
            in_progress.add(url)
            return True

    def mark_done(url: str):
        with seen_lock:
            in_progress.discard(url)
            seen.add(url)

    def worker():
        nonlocal pages
        while True:
            # stop condition: reached max pages (0 = unlimited)
            with pages_lock:
                if config["MAX_PAGES"] and pages >= config["MAX_PAGES"]:
                    return

            try:
                url = q.get(timeout=0.5)
            except queue.Empty:
                # no work available right now
                return

            if not should_take(url):
                q.task_done()
                continue

            try:
                rate_limit()

                session = get_session()
                try:
                    resp = session.get(url, timeout=config["TIMEOUT"], allow_redirects=True)
                except requests.RequestException:
                    mark_done(url)
                    q.task_done()
                    continue

                # Redirect checks
                final_url = resp.url
                if is_blacklisted(final_url, blacklist_rx):
                    mark_done(url)
                    q.task_done()
                    continue
                if not same_site(start_netloc, urlparse(final_url).netloc):
                    mark_done(url)
                    q.task_done()
                    continue

                if resp.status_code >= 400:
                    mark_done(url)
                    q.task_done()
                    continue
                ctype = resp.headers.get("Content-Type", "")
                if "text/html" not in ctype:
                    mark_done(url)
                    q.task_done()
                    continue

                html = resp.text
                head_title, body_md = extract_main_markdown(html, final_url)

                # Write markdown file (use final url for naming)
                fn = safe_filename(final_url)
                out_path = os.path.join(config["OUTPUT_DIR"], fn)

                parsed = urlparse(final_url)
                md_doc = []
                md_doc.append(body_md)
                md_doc.append("\n---\n")
                md_doc.append(f"*Quelle:* {final_url}\n")
                md_doc.append("")

                with open(out_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(md_doc))

                with pages_lock:
                    pages += 1
                    current_pages = pages

                # Extract & enqueue links
                soup = BeautifulSoup(html, "html.parser")
                for a in soup.find_all("a", href=True):
                    nxt = norm_url(final_url, a.get("href"))
                    if not nxt:
                        continue

                    if not is_allowed_url(nxt, start_netloc, rp, config["USER_AGENT"], whitelist_rx, blacklist_rx):
                        continue

                    with seen_lock:
                        if nxt in seen or nxt in in_progress:
                            continue
                    q.put(nxt)

                # Progress (queue.qsize() is approximate in threads)
                if config["MAX_PAGES"] > 0:
                    print_progress(current_pages, config["MAX_PAGES"], q.qsize(), final_url)
                else:
                    print_progress_unlimited(current_pages, q.qsize(), final_url)

                mark_done(url)

            finally:
                # If anything threw before mark_done, clean up claim
                with seen_lock:
                    in_progress.discard(url)
                    # don't blindly add to seen here; only if you want "attempted" semantics

                q.task_done()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker) for _ in range(workers)]
        for f in futures:
            f.result()

    print(f"\nDone. Saved {pages} pages to '{config['OUTPUT_DIR']}/' (seen={len(seen)}).")

def print_progress(done: int, total: int, queue_len: int, current_url: str, bar_width: int = 30):
    total = max(1, total)
    done = min(done, total)
    pct = (done / total) * 100.0
    filled = int((done / total) * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)
    msg = f"\r[{bar}] {done}/{total} ({pct:5.1f}%) | queue={queue_len:4d} | {current_url[:120]:120}"
    print("\r" + msg, end="", flush=True) 
    # print(msg, end="", flush=True)
    if done == total:
       print()  # newline

def print_progress_unlimited(done: int, queue_len: int, current_url: str):
    msg = f"\r{done} pages | queue={queue_len:4d} | {current_url[:120]:120}"
    print("\r" + msg, end="", flush=True)

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Simple crawler that respects robots.txt and saves page text as Markdown.")
    load_config()

    crawl(config["START_URL"], workers=config.get("WORKERS", 8))
