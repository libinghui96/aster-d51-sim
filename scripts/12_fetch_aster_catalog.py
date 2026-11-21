#!/usr/bin/env python3
"""
scripts/12_fetch_aster_catalog.py
================================================================================
Aster Hobby Gauge-1 全历史车型规格目录抓取、检索与官方规格说明书 PDF 自动归档工具
数据源: https://www.asterhobby.com/railwayhistory.html

功能:
  1. 解析 Aster Hobby 官方 Railway History 全表 (1975–2023 共 110+ 款真蒸汽机车)，
     结构化导出至 `configs/aster_railwayhistory_catalog.json`；
  2. 支持按关键词 (`--query D51` / `--query C62` / `--query "Big Boy"`) 检索同车族历代版本，
     辅助辨析锅炉代系 (如 1986/1992 D51 Mixed-Type vs 2007/2010 D51 van Riemsdijk Type "C")；
  3. 支持 `--download-pdfs data/raw_pdfs` 自动下载关联车型的官方规格说明书 PDF 作为先验设计参考。
================================================================================
"""
from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
import urllib.parse
import urllib.request

BASE_URL = "https://www.asterhobby.com/"
HISTORY_URL = "https://www.asterhobby.com/railwayhistory.html"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG_JSON = PROJECT_ROOT / "configs" / "aster_railwayhistory_catalog.json"


class _AsterHistoryTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_tr = False
        self.in_cell = False
        self.cur_href: Optional[str] = None
        self.cur_cell_text: List[str] = []
        self.cur_cell_hrefs: List[str] = []
        self.cur_row: List[Dict[str, Any]] = []
        self.rows: List[List[Dict[str, Any]]] = []

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        t = tag.lower()
        if t == "tr":
            self.in_tr = True
            self.cur_row = []
        elif self.in_tr and t in ("td", "th"):
            self.in_cell = True
            self.cur_cell_text = []
            self.cur_cell_hrefs = []
        elif self.in_cell and t == "a":
            attr_d = dict(attrs)
            href = (attr_d.get("href") or "").strip()
            if href:
                self.cur_href = href
                self.cur_cell_hrefs.append(href)

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.cur_cell_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t == "a":
            self.cur_href = None
        elif self.in_cell and t in ("td", "th"):
            self.in_cell = False
            txt = re.sub(r"\s+", " ", "".join(self.cur_cell_text)).strip()
            self.cur_row.append({"text": txt, "hrefs": list(self.cur_cell_hrefs)})
        elif self.in_tr and t == "tr":
            self.in_tr = False
            if any(c["text"] for c in self.cur_row):
                self.rows.append(self.cur_row)


def parse_railwayhistory_html(html_text: str) -> List[Dict[str, Any]]:
    parser = _AsterHistoryTableParser()
    parser.feed(html_text)
    entries: List[Dict[str, Any]] = []
    for row in parser.rows:
        if len(row) < 6:
            continue
        no_txt = row[0]["text"]
        if not re.fullmatch(r"\d+", no_txt):
            continue
        pdfs: List[str] = []
        for cell in row:
            for h in cell["hrefs"]:
                if ".pdf" in h.lower():
                    pdfs.append(urllib.parse.urljoin(BASE_URL, h))
        entries.append(
            {
                "catalog_no": int(no_txt),
                "year": row[1]["text"],
                "locomotive": row[2]["text"],
                "wheel_arrangement": row[3]["text"],
                "railway_country": row[4]["text"],
                "production_qty": row[5]["text"],
                "spec_pdf_urls": sorted(set(pdfs)),
            }
        )
    return entries


def fetch_text(url: str, timeout: float = 25.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "AsterLiveSteamTwin/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def download_pdf(url: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)).name
    dst = out_dir / fname
    if dst.exists() and dst.stat().st_size > 1024:
        return dst
    encoded_url = urllib.parse.urlsplit(url)
    safe_path = urllib.parse.quote(urllib.parse.unquote(encoded_url.path), safe="/()_-.")
    clean_url = urllib.parse.urlunsplit((encoded_url.scheme, encoded_url.netloc, safe_path, encoded_url.query, ""))
    req = urllib.request.Request(clean_url, headers={"User-Agent": "AsterLiveSteamTwin/1.0"})
    with urllib.request.urlopen(req, timeout=30.0) as resp:
        dst.write_bytes(resp.read())
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch, query, and archive Aster Hobby Gauge-1 locomotive specifications")
    ap.add_argument("--refresh", action="store_true", help="Re-fetch HTML from https://www.asterhobby.com/railwayhistory.html")
    ap.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_JSON, help="Path to aster_railwayhistory_catalog.json")
    ap.add_argument("--query", type=str, default="", help="Case-insensitive locomotive substring filter (e.g. D51, C62, Shay)")
    ap.add_argument("--download-pdfs", type=Path, default=None, help="Directory (e.g. data/raw_pdfs) to save matched spec PDFs")
    args = ap.parse_args()

    if args.refresh or not args.catalog.exists():
        html = fetch_text(HISTORY_URL)
        models = parse_railwayhistory_html(html)
        payload = {
            "source_url": HISTORY_URL,
            "total_models": len(models),
            "models": models,
        }
        args.catalog.parent.mkdir(parents=True, exist_ok=True)
        args.catalog.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[ASTER-CATALOG] Parsed {len(models)} historical models -> {args.catalog}")
    else:
        raw = json.loads(args.catalog.read_text(encoding="utf-8"))
        models = raw.get("models", raw if isinstance(raw, list) else [])

    q = args.query.strip().lower()
    matched = [
        m
        for m in models
        if not q
        or q in str(m.get("locomotive", m.get("model_name", ""))).lower()
        or q in str(m.get("railway_country", m.get("country", ""))).lower()
    ]
    print(f"[ASTER-CATALOG] Matched {len(matched)}/{len(models)} models" + (f" for query={args.query!r}:" if q else ":"))
    for m in matched[:25]:
        no = m.get("catalog_no", m.get("no", "?"))
        yr = m.get("year", "")
        name = m.get("locomotive", m.get("model_name", ""))
        wh = m.get("wheel_arrangement", m.get("wheels", ""))
        pdfs = m.get("spec_pdf_urls", [m["pdf_url"]] if m.get("pdf_url") else [])
        print(f"  #{no:>3} ({yr}) {name:<36} [{wh}] PDFs={len(pdfs)}")
        if args.download_pdfs is not None:
            for purl in pdfs:
                saved = download_pdf(purl, args.download_pdfs)
                print(f"        -> archived {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
