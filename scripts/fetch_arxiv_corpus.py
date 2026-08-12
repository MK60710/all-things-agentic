"""Pull a curated paper set from the arXiv API for the demo corpus.

Usage:
    python scripts/fetch_arxiv_corpus.py --query "LLM agent memory architecture" --count 18
"""

from __future__ import annotations

import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"
CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"


def search_arxiv(query: str, count: int) -> list[dict]:
    # Field-scoped (title/abstract) + category filter instead of a loose
    # `all:` full-text match, which pulls in off-topic false positives
    # (e.g. hardware "memory architecture" papers on an LLM-agent query).
    terms = " AND ".join(f'(ti:"{t}" OR abs:"{t}")' for t in query.split())
    search_query = f"({terms}) AND (cat:cs.CL OR cat:cs.AI OR cat:cs.LG)"
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": count,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    response = requests.get(ARXIV_API, params=params, timeout=30)
    response.raise_for_status()
    root = ET.fromstring(response.text)

    papers = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        arxiv_id = entry.find(f"{ATOM_NS}id").text.strip().split("/abs/")[-1]
        title = entry.find(f"{ATOM_NS}title").text.strip().replace("\n", " ")
        title = re.sub(r"\s+", " ", title)
        summary = entry.find(f"{ATOM_NS}summary").text.strip()
        pdf_url = None
        for link in entry.findall(f"{ATOM_NS}link"):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
        if pdf_url is None:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        papers.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": summary,
                "pdf_url": pdf_url,
            }
        )
    return papers


def download_papers(papers: list[dict], out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for paper in papers:
        safe_id = paper["arxiv_id"].replace("/", "_")
        local_path = out_dir / f"{safe_id}.pdf"
        if not local_path.exists():
            response = requests.get(paper["pdf_url"], timeout=60)
            response.raise_for_status()
            local_path.write_bytes(response.content)
            time.sleep(1)  # be polite to arXiv's servers
        try:
            portable_path = local_path.relative_to(CORPUS_DIR.parent)
        except ValueError:
            portable_path = local_path
        manifest.append({**paper, "local_path": portable_path.as_posix()})
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query", default="LLM agent memory architecture", help="Search topic"
    )
    parser.add_argument("--count", type=int, default=18)
    parser.add_argument("--out", default=str(CORPUS_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out)
    print(f"Searching arXiv for: {args.query!r} (top {args.count})")
    papers = search_arxiv(args.query, args.count)
    print(f"Found {len(papers)} papers. Downloading to {out_dir}...")
    manifest = download_papers(papers, out_dir)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Done. {len(manifest)} papers saved. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
