#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
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

BLACKLIST_PATH_REGEX = [
    r"^sso/.*",
    r"^qisserver/.*",
    r"^kontaktformular/.*",
    # r"^admin/.*",
    # r"^wp-admin/.*",
]

USER_AGENT = "SimpleTextCrawler/1.0 (+https://example.org; contact=you@example.org)"
TIMEOUT = 20
MAX_PAGES = 2500
OUTPUT_DIR = "docs_md"

def compile_blacklist(patterns):
    return [re.compile(p) for p in patterns]

def is_blacklisted(url: str, compiled) -> bool:
    p = urlparse(url)
    path = (p.path or "/").lstrip("/")  # match like "sso/..."
    return any(rx.search(path) for rx in compiled)

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
        "header#uni-head"
    ]

    # remove noise
    for tag in soup(["script", "style", "noscript", "template", "svg"]):
        tag.decompose()
        
    for el in soup.select(",".join(selectors)):
        el.decompose()

    # EXCLUDE all headers
    #for ft in soup.find_all("header"):
    #    ft.decompose()

    # EXCLUDE all footers
    for ft in soup.find_all("footer"):
        ft.decompose()
        
    for h in soup.select("h1,h2,h3,h4,h5,h6"):
        # get_text strips all tags; if there's no visible text, drop the heading
        if not h.get_text(" ", strip=True):
            h.decompose()

    head_title = ""
    if soup.title and soup.title.string:
        head_title = soup.title.string.strip()
        
    main = soup.body or soup

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
        sess.headers.update({"User-Agent": USER_AGENT})
        _thread_local.session = sess
    return sess

def crawl(start_url: str, workers: int = 8):
    blacklist_rx = compile_blacklist(BLACKLIST_PATH_REGEX)
    print(f"\nStart crawling with {workers} workers at {MAX_PAGES} pages.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rp = build_robot_parser(start_url)
    try:
        crawl_delay = rp.crawl_delay(USER_AGENT)
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
        """Atomically 'claim' url for processing."""
        if is_blacklisted(url, blacklist_rx):
            return False

        p = urlparse(url)
        if p.scheme not in ("http", "https"):
            return False
        if not same_site(start_netloc, p.netloc):
            return False

        try:
            allowed = rp.can_fetch(USER_AGENT, url)
        except Exception:
            allowed = True
        if not allowed:
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
            # stop condition: reached max pages
            with pages_lock:
                if pages >= MAX_PAGES:
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
                    resp = session.get(url, timeout=TIMEOUT, allow_redirects=True)
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
                out_path = os.path.join(OUTPUT_DIR, fn)

                parsed = urlparse(final_url)
                md_doc = []
                md_doc.append(f"# {head_title or parsed.path or final_url}\n")
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
                    p2 = urlparse(nxt)
                    if p2.scheme not in ("http", "https"):
                        continue
                    if not same_site(start_netloc, p2.netloc):
                        continue
                    if is_blacklisted(nxt, blacklist_rx):
                        continue

                    # Important: don't enqueue endlessly; also reduce duplicates early
                    with seen_lock:
                        if nxt in seen or nxt in in_progress:
                            continue
                    q.put(nxt)

                # Progress (queue.qsize() is approximate in threads)
                print_progress(current_pages, MAX_PAGES, q.qsize(), final_url)

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

    print(f"\nDone. Saved {pages} pages to '{OUTPUT_DIR}/' (seen={len(seen)}).")

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


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Simple crawler that respects robots.txt and saves page text as Markdown.")
    ap.add_argument("url", help="Start URL, e.g. https://example.com")
    ap.add_argument("--out", default=OUTPUT_DIR, help="Output directory (default: crawl_md)")
    ap.add_argument("--max-pages", type=int, default=MAX_PAGES, help="Max pages to crawl (default: 500)")
    ap.add_argument("--workers", type=int, default=8, help="Number of worker threads (default: 8)")
    args = ap.parse_args()

    OUTPUT_DIR = args.out
    MAX_PAGES = args.max_pages

    crawl(args.url, workers=args.workers)
