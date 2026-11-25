"""RegLens pipeline: articles -> findings (+ references, similarity, terms, evaluation).

Usage:
    python run_pipeline.py                 # uses data/articles.json
    python run_pipeline.py --mock          # uses data/mock_articles.json
    python run_pipeline.py --articles PATH
Every step is optional: a missing or broken module prints a warning and the run continues.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import traceback
from collections import Counter

DATA = "data"
SEV_RANK = {"high": 3, "medium": 2, "low": 1}


def _load_module(name):
    try:
        return importlib.import_module(name)
    except Exception as e:  # noqa: BLE001
        print(f"[pipeline] WARNING: module '{name}' not available ({type(e).__name__}: {e})")
        return None


def _step(label, fn, default):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        print(f"[pipeline] WARNING: {label} failed: {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)
        return default


def _ws(s):
    return re.sub(r"\s+", " ", s or "").strip()


def _verify_fallback(findings, articles):
    by_id = {a["id"]: a for a in articles}
    for f in findings:
        ok = True
        for ev in f.get("evidence", []):
            a = by_id.get(ev.get("article_id"))
            q = (ev.get("quote") or "").strip()
            if not a or not q or (q not in a["text"] and _ws(q) not in _ws(a["text"])):
                ok = False
        f["quotes_verified"] = ok
    return findings


def _valid(f):
    return isinstance(f, dict) and f.get("id") and f.get("type") and isinstance(f.get("article_ids"), list)


def _dump(name, obj):
    with open(os.path.join(DATA, name), "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)


def run(articles_path, config=None):
    config = config or {}
    with open(articles_path, encoding="utf-8") as fh:
        articles = json.load(fh)
    print(f"[pipeline] {len(articles)} articles from {articles_path}")

    rules = _load_module("detect_rules")
    sim = _load_module("detect_similarity")
    ai = _load_module("ai_assist")

    findings = []
    if rules:
        findings += _step("detect_rules.detect", lambda: rules.detect(articles, config) or [], [])
    if sim:
        findings += _step("detect_similarity.detect", lambda: sim.detect(articles, config) or [], [])

    bad = [f for f in findings if not _valid(f)]
    if bad:
        print(f"[pipeline] WARNING: dropped {len(bad)} findings that do not match the contract")
    findings = [f for f in findings if _valid(f)]

    # Deduplicate by id, keep the higher severity.
    best = {}
    for f in findings:
        f.setdefault("severity", "medium")
        f.setdefault("evidence", [])
        f.setdefault("extra", {})
        cur = best.get(f["id"])
        if cur is None or SEV_RANK.get(f["severity"], 0) > SEV_RANK.get(cur["severity"], 0):
            best[f["id"]] = f
    findings = list(best.values())

    if ai and hasattr(ai, "verify_quotes"):
        findings = _step("ai_assist.verify_quotes", lambda: ai.verify_quotes(findings, articles),
                         _verify_fallback(findings, articles))
    else:
        findings = _verify_fallback(findings, articles)

    findings.sort(key=lambda f: (-SEV_RANK.get(f.get("severity"), 0), f.get("type", ""), f["id"]))
    os.makedirs(DATA, exist_ok=True)
    _dump("findings.json", findings)

    if rules and hasattr(rules, "references"):
        refs = _step("detect_rules.references", lambda: rules.references(articles), None)
        if refs is not None:
            _dump("references.json", refs)
    if sim and hasattr(sim, "similar_pairs"):
        pairs = _step("detect_similarity.similar_pairs", lambda: sim.similar_pairs(articles), None)
        if pairs is not None:
            _dump("similarity.json", [list(p) for p in pairs])
    if sim and hasattr(sim, "term_index"):
        terms = _step("detect_similarity.term_index", lambda: sim.term_index(articles), None)
        if terms is not None:
            _dump("terms.json", terms)

    print(f"\n[pipeline] wrote {len(findings)} findings to {DATA}/findings.json")
    for k, v in sorted(Counter(f["type"] for f in findings).items()):
        print(f"  {k:<30} {v:>4}")
    sev = Counter(f["severity"] for f in findings)
    print(f"  severity: high {sev['high']} · medium {sev['medium']} · low {sev['low']}")
    unverified = sum(1 for f in findings if not f.get("quotes_verified"))
    if unverified:
        print(f"  WARNING: {unverified} findings have quotes not found in source text")

    labels = os.path.join(DATA, "expert_labels.csv")
    if os.path.exists(labels):
        ev = _load_module("evaluate")
        if ev:
            res = _step("evaluate", lambda: ev.evaluate(findings, labels), None)
            if res:
                _dump("evaluation.json", res)
                print(f"\n[pipeline] evaluation: found {res['found']} of {res['labels']} expert-labeled cases "
                      f"(recall {res['recall']:.0%})")
    return findings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="use data/mock_articles.json")
    ap.add_argument("--articles", default=os.path.join(DATA, "articles.json"))
    ap.add_argument("--config", help="optional JSON file with detector config (thresholds, icao_current, def_regulation)")
    args = ap.parse_args()
    path = os.path.join(DATA, "mock_articles.json") if args.mock else args.articles
    if not os.path.exists(path):
        sys.exit(f"[pipeline] {path} not found. Run: python ingest.py data/raw/*  (or use --mock)")
    config = {}
    if args.config:
        with open(args.config, encoding="utf-8") as fh:
            config = json.load(fh)
    elif os.path.exists("config.json"):
        with open("config.json", encoding="utf-8") as fh:
            config = json.load(fh)
    run(path, config)


if __name__ == "__main__":
    main()
