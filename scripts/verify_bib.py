#!/usr/bin/env python3
"""Resolve every arXiv citation in a .bib file against the arXiv API and fail on mismatch.

Citation verification is a build step, not a courtesy. Several arXiv IDs in this project were
surfaced by literature search and are index-only until confirmed. This script fetches each
entry's `eprint` from the arXiv API and checks that the stored title and first-author surname
plausibly match what arXiv returns. Any mismatch, unresolvable ID, or entry missing an `eprint`
(other than the few genuinely non-arXiv venues, whitelisted below) fails the build.

Usage:
    python3 scripts/verify_bib.py reports/refs.bib
    python3 scripts/verify_bib.py reports/refs.bib --offline   # structural checks only, no network

Exit code 0 iff every entry passes.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

ARXIV_API = "http://export.arxiv.org/api/query?"

# Entries legitimately without an arXiv eprint (proceedings-only). Keyed by bibtex key.
NON_ARXIV_WHITELIST = {
    "zhang2024cam",  # CaM — ICML 2024, PMLR v235; no arXiv eprint field on record.
}


def parse_bib(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    entries = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),", text):
        start = m.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        body = text[start : i - 1]
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\}|\"[^\"]*\"|[^,\n]+)", body):
            key = fm.group(1).lower()
            val = fm.group(2).strip().strip("{}\"").strip()
            fields[key] = val
        fields["_type"] = m.group(1).lower()
        fields["_key"] = m.group(2).strip()
        entries.append(fields)
    return entries


def norm(s: str) -> str:
    s = re.sub(r"[{}\\]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s.lower())
    return " ".join(s.split())


def first_author_surname(author_field: str) -> str:
    first = author_field.split(" and ")[0].strip()
    if "," in first:
        return norm(first.split(",")[0])
    return norm(first.split()[-1]) if first else ""


def fetch_arxiv(arxiv_id: str, retries: int = 5) -> dict | None:
    q = urllib.parse.urlencode({"id_list": arxiv_id})
    data = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                ARXIV_API + q, headers={"User-Agent": "kvbench-verifybib/1.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                wait = 3 * (2**attempt)  # 3, 6, 12, 24s backoff
                print(f"    429 rate-limited, backing off {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            print(f"    network error: {e}", file=sys.stderr)
            return None
        except Exception as e:  # noqa: BLE001
            print(f"    network error: {e}", file=sys.stderr)
            return None
    if data is None:
        return None
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(data)
    entry = root.find("a:entry", ns)
    if entry is None:
        return None
    title_el = entry.find("a:title", ns)
    if title_el is None or title_el.text is None:
        return None
    authors = [
        (a.find("a:name", ns).text or "")
        for a in entry.findall("a:author", ns)
        if a.find("a:name", ns) is not None
    ]
    return {"title": title_el.text, "authors": authors}


def check(entry: dict, offline: bool) -> list[str]:
    key = entry["_key"]
    problems = []
    eprint = entry.get("eprint")
    if not eprint:
        if key not in NON_ARXIV_WHITELIST:
            problems.append(f"{key}: no eprint and not in NON_ARXIV_WHITELIST")
        return problems
    if not re.fullmatch(r"\d{4}\.\d{4,5}", eprint):
        problems.append(f"{key}: eprint '{eprint}' is not a valid arXiv id")
        return problems
    if offline:
        return problems
    meta = fetch_arxiv(eprint)
    if meta is None:
        problems.append(f"{key}: arXiv id {eprint} did not resolve")
        return problems
    got_title = norm(meta["title"])
    want_title = norm(entry.get("title", ""))
    # token-overlap heuristic: >=60% of the shorter title's tokens present in the other
    wt, gt = set(want_title.split()), set(got_title.split())
    if wt and gt:
        overlap = len(wt & gt) / min(len(wt), len(gt))
        if overlap < 0.6:
            problems.append(
                f"{key}: title mismatch for {eprint}\n"
                f"      bib:   {entry.get('title','')[:80]}\n"
                f"      arXiv: {meta['title'][:80]}"
            )
    want_author = first_author_surname(entry.get("author", ""))
    got_authors = norm(" ".join(meta["authors"]))
    if want_author and want_author not in got_authors:
        problems.append(
            f"{key}: first author '{want_author}' not among arXiv authors for {eprint}"
        )
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bib")
    ap.add_argument("--offline", action="store_true", help="structural checks only, no network")
    args = ap.parse_args()

    entries = parse_bib(args.bib)
    print(f"verify_bib: {len(entries)} entries in {args.bib}"
          f"{' (offline)' if args.offline else ''}")

    all_problems = []
    for e in entries:
        probs = check(e, args.offline)
        status = "ok " if not probs else "FAIL"
        print(f"  [{status}] {e['_key']} ({e.get('eprint','no-eprint')})")
        all_problems.extend(probs)
        if not args.offline and e.get("eprint"):
            time.sleep(0.34)  # be polite to the arXiv API

    if all_problems:
        print("\nBIBLIOGRAPHY VERIFICATION FAILED:", file=sys.stderr)
        for p in all_problems:
            print("  - " + p, file=sys.stderr)
        return 1
    print("\nAll citations verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
