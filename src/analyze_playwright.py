"""
analyze_playwright.py

- Playwright ile web sayfası açar, tüm network response'ları dinler.
- Her site için detaylı sütunlar üretir: tiplere göre byte ve istek sayıları,
  transfer vs resource boyutu fallback'ları, cacheable bytes, third-party, TTFB, largest resource, vs.
- CSV (UTF-8-SIG), JSON, Excel çıktı.
- Kullanım: python analyze_playwright.py
"""

import asyncio
import csv
import json
import os
import time
import math
from urllib.parse import urlparse

import httpx  # used for HEAD fallback for media/content-length
import openpyxl
from openpyxl.utils import get_column_letter
from playwright.async_api import async_playwright

# ---------- AYARLAR ----------
INPUT_FILE = "university.txt"   # bir satırda 1 url
OUTPUT_CSV = "results_playwrightnew3.csv"
OUTPUT_JSON = "results_playwrightnew3.json"
OUTPUT_XLSX = "results_playwrightnew3.xlsx"

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
NAV_TIMEOUT = 90 * 1000   # ms
WAIT_AFTER_LOAD = 2.0     # sec to wait after networkidle
FETCH_BODY_FALLBACK = True  # Eğer content-length yoksa response.body() al (ağır)
HEAD_FALLBACK_FOR_MEDIA = True
SLEEP_BETWEEN_SITES = 2.0  # saniye

# CO2 ayarları
ENERGY_KWH_PER_GB = 0.1
CARBON_INTENSITY = 450.0  # gCO2/kWh

# ---------- YARDIMCILER ----------
def calc_co2(total_bytes):
    gb = total_bytes / (1024 ** 3)
    return gb * ENERGY_KWH_PER_GB * CARBON_INTENSITY

def is_third_party(resource_url, page_url):
    try:
        r = urlparse(resource_url).hostname or ""
        p = urlparse(page_url).hostname or ""
        # treat subdomains of same eTLD+1 as same origin (basic)
        return (r != p) and (not r.endswith("."+p))
    except Exception:
        return False

def safe_int(v):
    try:
        return int(v)
    except:
        return 0

# ---------- TEMPLATE OF STATS ----------
def make_empty_stats():
    return {
        # bytes per type
        "html_bytes": 0,
        "css_bytes": 0,
        "js_bytes": 0,
        "image_bytes": 0,
        "font_bytes": 0,
        "video_bytes": 0,
        "audio_bytes": 0,
        "json_bytes": 0,
        "other_bytes": 0,
        # counts per type
        "total_requests": 0,
        "html_requests": 0,
        "css_requests": 0,
        "js_requests": 0,
        "image_requests": 0,
        "font_requests": 0,
        "video_requests": 0,
        "audio_requests": 0,
        "json_requests": 0,
        "other_requests": 0,
        # transfer vs resource
        "total_transfer_bytes": 0,   # transferSize approximated by Content-Length or body length
        "total_resource_bytes": 0,   # fallback same as transfer
        "largest_resource_bytes": 0,
        "largest_resource_url": None,
        "gzip_compressed_bytes": 0,
        # cacheable
        "cacheable_bytes": 0,
        # protocols & meta
        "uses_http2": False,
        "third_party_requests": 0,
        "third_party_bytes": 0,
        # request/response status & timing aggregates
        "ttfb_ms_avg": None,
        "ttfb_ms_min": None,
        "ttfb_ms_max": None,
        "lighthouse_performance": None,  # placeholder if you want to fill separately
        # derived
        "total_bytes": 0,  # same as total_transfer_bytes
    }

# ---------- MAIN SITE SCAN ----------
async def scan_site(playwright, page_url):
    browser = await playwright.chromium.launch(headless=True, args=["--no-sandbox", "--disable-gpu"])
    context = await browser.new_context(user_agent=USER_AGENT, bypass_csp=True)
    page = await context.new_page()
    stats = make_empty_stats()

    # maps to store request start times
    req_start = {}
    # list of ttfb values (ms)
    ttfb_list = []

    async def on_request(request):
        req_id = request._impl_obj._guid  # internal unique id (works with Playwright impl)
        req_start[req_id] = time.monotonic()

    async def on_response(response):
        try:
            req = response.request
            req_id = req._impl_obj._guid
        except Exception:
            req_id = None

        # determine resource type
        rtype = (req.resource_type or "").lower() if req else ""
        url = response.url
        status = response.status
        headers = {k.lower(): v for k, v in response.headers.items()}

        # TTFB: time between request sent and first response event
        ttfb_ms = None
        if req_id and req_id in req_start:
            ttfb_ms = (time.monotonic() - req_start[req_id]) * 1000.0
            ttfb_list.append(ttfb_ms)

        # size: try content-length header
        size = 0
        cl = headers.get("content-length")
        if cl:
            size = safe_int(cl)
        else:
            # fallback: body() - may be heavy
            if FETCH_BODY_FALLBACK:
                try:
                    body = await response.body()
                    size = len(body or b"")
                except Exception:
                    size = 0

        # compressed bytes estimate: if transfer-encoding chunked we can't know; use size as compressed
        compressed = size

        # cacheable?
        cache_control = headers.get("cache-control", "")
        if "max-age" in cache_control.lower() or "immutable" in cache_control.lower():
            stats["cacheable_bytes"] += size

        # protocol detection: try to use headers['alt-svc'] or : via response timing? limited; check server header hints
        proto = headers.get("x-protocol") or headers.get("alt-svc") or ""
        if "h2" in proto or "http2" in proto.lower():
            stats["uses_http2"] = True

        # third-party?
        if is_third_party(url, page_url):
            stats["third_party_requests"] += 1
            stats["third_party_bytes"] += size

        # increment counts and bytes by inferred type
        stats["total_requests"] += 1
        stats["total_transfer_bytes"] += size
        stats["total_resource_bytes"] += size
        stats["total_bytes"] = stats["total_transfer_bytes"]

        lower_mime = headers.get("content-type", "").lower()

        def add(type_bytes_key, type_requests_key):
            stats[type_bytes_key] += size
            stats[type_requests_key] += 1

        # Heuristic allocations:
        if "text/html" in lower_mime or url.endswith(".html") or rtype == "document":
            add("html_bytes", "html_requests")
        elif "text/css" in lower_mime or rtype == "stylesheet" or url.endswith(".css"):
            add("css_bytes", "css_requests")
        elif "javascript" in lower_mime or rtype == "script" or url.endswith(".js"):
            add("js_bytes", "js_requests")
        elif lower_mime.startswith("image/") or rtype == "image" or url.split("?")[0].lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".avif")):
            add("image_bytes", "image_requests")
        elif "font" in lower_mime or url.split("?")[0].lower().endswith((".woff", ".woff2", ".ttf", ".eot")):
            add("font_bytes", "font_requests")
        elif lower_mime.startswith("video/") or rtype == "media" or url.split("?")[0].lower().endswith((".mp4", ".webm", ".m3u8")):
            add("video_bytes", "video_requests")
        elif lower_mime.startswith("audio/") or url.split("?")[0].lower().endswith((".mp3", ".wav")):
            add("audio_bytes", "audio_requests")
        elif "application/json" in lower_mime or rtype in ("xhr", "fetch"):
            add("json_bytes", "json_requests")
        else:
            add("other_bytes", "other_requests")

        # largest resource
        if size > stats["largest_resource_bytes"]:
            stats["largest_resource_bytes"] = size
            stats["largest_resource_url"] = url

        # ttfb aggregations handled later

    page.on("request", on_request)
    page.on("response", on_response)

    # navigate
    try:
        start = time.monotonic()
        await page.goto(page_url, timeout=NAV_TIMEOUT, wait_until="networkidle")
        await asyncio.sleep(WAIT_AFTER_LOAD)
    except Exception as e:
        print(f"❌ Navigation/Playwright error for {page_url}: {e}")
    finally:
        # ensure close to flush events
        await asyncio.sleep(0.5)

    # aggregate TTFB stats
    if ttfb_list:
        stats["ttfb_ms_avg"] = sum(ttfb_list) / len(ttfb_list)
        stats["ttfb_ms_min"] = min(ttfb_list)
        stats["ttfb_ms_max"] = max(ttfb_list)

    # attempt to find media files in main HTML (additional HEAD fallback)
    extra_media_bytes = 0
    if HEAD_FALLBACK_FOR_MEDIA:
        try:
            # fetch page HTML quickly with httpx (lighter)
            async with httpx.AsyncClient(follow_redirects=True, headers={"User-Agent": USER_AGENT}, timeout=15.0) as client:
                r = await client.get(page_url)
                text = r.text
                # find urls that look like media (mp4, mp3, webm, m3u8)
                import re
                found = set(re.findall(r'(https?://[^\s\'"<>]+?\.(?:mp4|webm|m3u8|mp3|wav))', text, flags=re.IGNORECASE))
                # also search src/data-src attributes
                found_attr = set(re.findall(r'(?:src|data-src|data-video)=["\']([^"\']+)["\']', text, flags=re.IGNORECASE))
                for u in found_attr:
                    if any(u.lower().endswith(ext) for ext in (".mp4",".webm",".mp3",".wav",".m3u8")):
                        if u.startswith("http"):
                            found.add(u)
                        else:
                            # relative -> join
                            from urllib.parse import urljoin
                            found.add(urljoin(page_url, u))
                # HEAD each media to get content-length
                for m in found:
                    try:
                        head = await client.head(m, timeout=20.0)
                        cl = head.headers.get("content-length")
                        if cl:
                            extra_media_bytes += safe_int(cl)
                    except Exception:
                        # try range request GET first 1 byte as fallback to get content-length via response headers
                        try:
                            r2 = await client.get(m, headers={"Range":"bytes=0-0"}, timeout=20.0)
                            cl2 = r2.headers.get("content-range") or r2.headers.get("content-length")
                            if cl2 and "bytes" in (r2.headers.get("content-range") or ""):
                                # content-range like: bytes 0-0/12345
                                total = int((r2.headers.get("content-range") or "0").split("/")[-1])
                                extra_media_bytes += total
                            elif cl2:
                                extra_media_bytes += safe_int(cl2)
                        except Exception:
                            pass
        except Exception:
            pass

    # finalize totals & CO2
    stats["extra_media_bytes"] = extra_media_bytes
    stats["final_total_bytes"] = stats["total_bytes"] + extra_media_bytes
    stats["final_co2_grams"] = calc_co2(stats["final_total_bytes"])

    # close
    await context.close()
    await browser.close()

    return stats

# ---------- RUN FOR ALL SITES ----------
async def main():
    # read urls
    if not os.path.exists(INPUT_FILE):
        print(f"Input file {INPUT_FILE} bulunamadı. Bir satırda 1 URL olacak şekilde oluştur.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        urls = [u.strip() for u in f.readlines() if u.strip()]

    results = []
    async with async_playwright() as p:
        for i, url in enumerate(urls, 1):
            print(f"\n[{i}/{len(urls)}] Analiz ediliyor: {url}")
            stats = await scan_site(p, url)
            entry = {"url": url, **stats}
            results.append(entry)
            print(f"✅ Final CO2: {entry['final_co2_grams']:.4f} g (final_total_bytes {entry['final_total_bytes']})")
            await asyncio.sleep(SLEEP_BETWEEN_SITES)

    # write CSV (UTF-8 with BOM)
    fieldnames = list(results[0].keys()) if results else ["url"]
    # ensure deterministic column order
    ordered = [
        "url",
        "total_requests","html_requests","css_requests","js_requests","image_requests","font_requests",
        "video_requests","audio_requests","json_requests","other_requests",
        "total_transfer_bytes","total_resource_bytes","total_bytes",
        "html_bytes","css_bytes","js_bytes","image_bytes","font_bytes","video_bytes","audio_bytes","json_bytes","other_bytes",
        "largest_resource_bytes","largest_resource_url",
        "gzip_compressed_bytes","cacheable_bytes",
        "third_party_requests","third_party_bytes",
        "uses_http2",
        "ttfb_ms_avg","ttfb_ms_min","ttfb_ms_max",
        "extra_media_bytes","final_total_bytes","final_co2_grams",
    ]
    # final columns = intersection of ordered + any extra
    cols = [c for c in ordered if c in fieldnames] + [c for c in fieldnames if c not in ordered]

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in results:
            writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # write XLSX
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "results"
    ws.append(cols)
    for r in results:
        ws.append([r.get(c,"") for c in cols])
    # autosize a few columns
    for i, col in enumerate(cols, start=1):
        ws.column_dimensions[get_column_letter(i)].width = min(60, max(10, len(col)+2))
    wb.save(OUTPUT_XLSX)

    print(f"\n✅ İşlem tamamlandı. CSV: {OUTPUT_CSV}  JSON: {OUTPUT_JSON}  XLSX: {OUTPUT_XLSX}")

if __name__ == "__main__":
    asyncio.run(main())
