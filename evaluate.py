"""RegLens evaluation: how many expert-labeled cases did we find?

data/expert_labels.csv columns: article_a,article_b,type   (article_b and type may be empty)
Article cells accept "CAR-OPS|4", "CAR-OPS 4", "CAR-OPS Art. 4", "CAR-OPS §4", "CAR-OPS · Art. 4", "CAR-OPS المادة 4".
Type accepts a family word (conflict, duplicate, overlap, definition, reference, icao) or an exact finding type.

Usage: python evaluate.py [findings.json] [labels.csv]
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import defaultdict

AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

FAMILIES = {
    "conflict": {"conflicting_values"},
    "duplicate": {"duplicate_requirement", "partial_overlap"},
    "overlap": {"duplicate_requirement", "partial_overlap"},
    "definition": {"duplicate_definition", "divergent_definition", "definition_missing_from_def"},
    "reference": {"broken_reference", "unverifiable_reference"},
    "icao": {"icao_edition_issue"},
}
ALL_TYPES = set().union(*FAMILIES.values())


def _reg_key(reg):
    r = reg.upper().translate(AR_DIGITS)
    r = re.sub(r"^CAR[\s\-–_]*", "", r)
    return re.sub(r"[^A-Z0-9]", "", r)


def _art_key(no):
    n = str(no).upper().translate(AR_DIGITS).replace(" ", "")
    return n.strip(".()[]")


def parse_article_ref(cell):
    """Return (reg_key, art_key) or None."""
    cell = (cell or "").strip()
    if not cell:
        return None
    if "|" in cell:
        reg, no = cell.split("|", 1)
        return _reg_key(reg), _art_key(no)
    m = re.match(r"^(.*?)[\s·,]*(?:§|Art(?:icle)?\.?|المادة|مادة)\s*\(?\s*([^\s)]+)\)?\s*$", cell, re.I)
    if not m:
        m = re.match(r"^(.*\S)\s+\(?([0-9٠-٩][^\s)]*)\)?$", cell)
    if not m:
        return None
    return _reg_key(m.group(1)), _art_key(m.group(2))


def _type_set(t):
    t = (t or "").strip().lower()
    if not t:
        return None
    if t in ALL_TYPES:
        return {t}
    for fam, types in FAMILIES.items():
        if fam in t:
            return types
    return None  # unknown type word: accept any type


def _finding_keys(f):
    keys = set()
    for aid in f.get("article_ids", []):
        if "|" in aid:
            reg, no = aid.split("|", 1)
            keys.add((_reg_key(reg), _art_key(no)))
    return keys


def load_labels(labels_path):
    labels = []
    with open(labels_path, encoding="utf-8-sig") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=2):
            a = parse_article_ref(row.get("article_a"))
            b = parse_article_ref(row.get("article_b"))
            if not a:
                print(f"[evaluate] line {i}: cannot read article_a {row.get('article_a')!r}; skipped")
                continue
            labels.append({"row": row, "keys": {k for k in (a, b) if k}, "types": _type_set(row.get("type"))})
    return labels


def _reviews():
    p = os.path.join("data", "reviews.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return {}


def evaluate(findings: list[dict], labels_path: str) -> dict:
    labels = load_labels(labels_path)
    fkeys = [(f, _finding_keys(f)) for f in findings]
    per_type = defaultdict(lambda: {"labels": 0, "found": 0})
    missed, matched = [], []
    for lab in labels:
        fam = (lab["row"].get("type") or "any").strip().lower() or "any"
        per_type[fam]["labels"] += 1
        hit = next((f for f, ks in fkeys
                    if lab["keys"] <= ks and (lab["types"] is None or f.get("type") in lab["types"])), None)
        if hit:
            per_type[fam]["found"] += 1
            matched.append({"label": lab["row"], "finding_id": hit["id"]})
        else:
            missed.append(lab["row"])
    n = len(labels)
    found = n - len(missed)

    rev = _reviews()
    decided = [r for r in rev.values() if isinstance(r, dict) and r.get("decision") in ("accepted", "modified", "rejected")]
    acc = (sum(1 for r in decided if r["decision"] in ("accepted", "modified")) / len(decided)) if decided else None

    return {
        "labels": n,
        "found": found,
        "recall": (found / n) if n else 0.0,
        "per_type": {k: {**v, "recall": (v["found"] / v["labels"]) if v["labels"] else 0.0} for k, v in per_type.items()},
        "missed": missed,
        "matched": matched,
        "findings_total": len(findings),
        "reviewed": len(decided),
        "expert_acceptance_rate": acc,
    }


def main(argv):
    fpath = argv[0] if len(argv) > 0 else os.path.join("data", "findings.json")
    lpath = argv[1] if len(argv) > 1 else os.path.join("data", "expert_labels.csv")
    if not os.path.exists(lpath):
        sys.exit(f"[evaluate] {lpath} not found (columns: article_a,article_b,type)")
    with open(fpath, encoding="utf-8") as fh:
        findings = json.load(fh)
    res = evaluate(findings, lpath)
    with open(os.path.join("data", "evaluation.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(f"Found {res['found']} of {res['labels']} expert-labeled cases (recall {res['recall']:.0%})")
    for t, v in res["per_type"].items():
        print(f"  {t:<14} {v['found']}/{v['labels']}")
    for m in res["missed"]:
        print(f"  MISSED: {m.get('article_a')} | {m.get('article_b')} | {m.get('type')}")
    if res["expert_acceptance_rate"] is not None:
        print(f"Expert acceptance rate: {res['expert_acceptance_rate']:.0%} of {res['reviewed']} reviewed findings")
    print("Wrote data/evaluation.json")


if __name__ == "__main__":
    main(sys.argv[1:])
