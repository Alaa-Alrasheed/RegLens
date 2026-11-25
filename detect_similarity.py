"""P3: definition consistency and cross-regulation requirement similarity.

Usage:
    python detect_similarity.py data/mock_articles.json
    python detect_similarity.py data/mock_articles.json --test
    python detect_similarity.py data/articles.json --config config.json

Config keys: dup (default 0.85), overlap (0.55), def_regulation (auto: CAR-DEF
or the first regulation containing DEF). Scores are lexical review candidates,
not a legal determination. Matching normalization never changes source quotes.
Only scikit-learn and the standard library are used; term_index works without
scikit-learn. Article comparisons use sparse products in blocks of 128 rows.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sys
import unicodedata

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = cosine_similarity = None

Article = dict
Finding = dict
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩أإآٱىة", "0123456789اااايه")
_MARKS = re.compile(r"[\u064b-\u0652\u0670\u0640]")
_SENTENCES = re.compile(r"(?<=[.!?؟؛;])\s+|\n+")
_QUOTED = re.compile(
    r'''["“'«]([^"”'»\n]{2,60})["”'»]\s*'''
    r"(?:means|shall mean|refers to|is defined as)\s+(.+)", re.I,
)
_UNQUOTED = re.compile(
    r"^\s*(?:\(?[a-z0-9]{1,3}\)\s*)?"
    r"([A-Z][A-Za-z\- ]{2,50}?)\s+means\s+(.+)$",
)
_ARABIC = re.compile(
    r"^\s*(?:[-–•]|\(?[0-9٠-٩a-zأ-ي]{1,3}[\).])?\s*"
    r"([^:\n]{2,40}?)\s*:\s+(.{10,})$",
)


def _warn(message: str) -> None:
    print(f"[similarity] WARNING: {message}", file=sys.stderr)


def _norm(text: str) -> str:
    return _MARKS.sub("", unicodedata.normalize("NFKC", text).lower().translate(_DIGITS))


def _term_key(term: str) -> str:
    words = re.findall(r"[^\W_]+", _norm(term))
    for i, word in enumerate(words):
        if word.startswith("ال") and len(word) >= 5:
            words[i] = word[2:]
        elif len(word) >= 5 and word.isascii() and word.endswith("s"):
            words[i] = word[:-1]
    return " ".join(words)


@dataclass(frozen=True)
class _Definition:
    article: Article
    term: str
    key: str
    body: str
    quote: str


def _articles(articles: list[Article]) -> list[Article]:
    # Ignore empty article bodies without mutating the caller's dictionaries.
    return sorted(
        (a for a in articles if a.get("id") and a.get("regulation") and a.get("text", "").strip()),
        key=lambda a: a["id"],
    )


def _definitions(articles: list[Article]) -> list[_Definition]:
    definitions = []
    seen = set()
    for article in _articles(articles):
        for line in article["text"].splitlines():
            quote = line.strip()
            match = _QUOTED.search(quote)
            if match is None and article.get("is_definitions", False):
                match = _UNQUOTED.fullmatch(quote) or _ARABIC.fullmatch(quote)
            if match is None:
                continue
            term, body = (part.strip() for part in match.groups())
            key = _term_key(term)
            identity = (article["id"], key, quote)
            if key and identity not in seen:
                seen.add(identity)
                definitions.append(_Definition(article, term, key, body, quote))
    return definitions


def term_index(articles: list[Article]) -> dict:
    """Group definitions by normalized term; text preserves the original line."""
    result = defaultdict(list)
    for definition in _definitions(articles):
        result[definition.key].append({
            "article_id": definition.article["id"],
            "term": definition.term,
            "text": definition.quote,
        })
    return dict(sorted(result.items()))


def _available() -> bool:
    if TfidfVectorizer is None or cosine_similarity is None:
        _warn("scikit-learn is unavailable; install requirements.txt to enable similarity scoring.")
        return False
    return True


def _vectors(texts: list[str]):
    if not texts:
        return None, None
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1,
    )
    try:
        return vectorizer, vectorizer.fit_transform([_norm(text) for text in texts])
    except ValueError as exc:
        # Blank/very short normalized text can have no character n-grams.
        _warn(f"cannot build a TF-IDF vocabulary ({exc}); skipped these comparisons.")
        return None, None


def _score(value) -> float:
    value = float(value)
    # Sparse products can place identical texts a few ulps below one.
    return 1.0 if value >= 1.0 - 1e-12 else max(0.0, value)


def _pair_scores(articles: list[Article], min_score: float):
    """Yield only qualifying upper-triangle nonzero entries, without a dense N² matrix."""
    if len(articles) < 2:
        return
    _, matrix = _vectors([a["text"] for a in articles])
    if matrix is None:
        return
    transpose = matrix.T
    for start in range(0, len(articles), 128):
        product = (matrix[start:start + 128] @ transpose).tocoo()
        keep = (product.col > product.row + start) & (product.data >= min_score - 1e-12)
        for row, col, value in zip(product.row[keep], product.col[keep], product.data[keep]):
            left, right = int(row) + start, int(col)
            score = _score(value)
            if score >= min_score and articles[left]["regulation"] != articles[right]["regulation"]:
                yield left, right, score


def similar_pairs(articles: list[Article], min_score: float = 0.25) -> list[tuple[str, str, float]]:
    """Return up to 2,000 scored cross-regulation links, including definitions.

    min_score must be in (0, 1]. Unlike detect's requirement candidates, this
    impact index includes short articles and definitions.
    """
    if not math.isfinite(min_score) or not 0 < min_score <= 1:
        raise ValueError("min_score must be greater than zero and at most one")
    source = _articles(articles)
    if len(source) < 2 or not _available():
        return []
    # Keep bounded output memory even when a corpus contains many repeated texts.
    import heapq

    pairs = ((source[i]["id"], source[j]["id"], score)
             for i, j, score in _pair_scores(source, min_score))
    return heapq.nsmallest(2000, pairs, key=lambda pair: (-pair[2], pair[0], pair[1]))


def _label(article: Article) -> str:
    return f"{article['regulation']} Art.{article.get('article_no', article['id'].split('|')[-1])}"


def _finding(kind, articles, evidence, summary, recommendation, rationale, confidence, extra):
    ids = sorted({article["id"] for article in articles})
    return {
        "id": f"{kind}:{'+'.join(ids)}",
        "type": kind,
        "severity": "high" if kind == "divergent_definition" else "medium",
        "article_ids": ids,
        "evidence": evidence,
        "summary": summary,
        "recommendation": recommendation,
        "rationale": rationale,
        "detector": "similarity",
        "confidence": _score(confidence),
        "quotes_verified": False,  # The pipeline/ai_assist owns verification.
        "extra": extra,
    }


def _definition_findings(articles, definitions, config):
    if not definitions:
        return []
    regulations = sorted({a["regulation"] for a in articles})
    requested = config.get("def_regulation")
    if requested:
        baseline = next((reg for reg in regulations if reg.casefold() == str(requested).casefold()), None)
    else:
        baseline = next((reg for reg in regulations if reg.upper() == "CAR-DEF"), None)
        baseline = baseline or next((reg for reg in regulations if "DEF" in reg.upper()), None)
    if baseline is None:
        _warn("definitions regulation not loaded; CAR-DEF comparisons skipped (term index is still available).")
        return []
    _, matrix = _vectors([definition.body for definition in definitions])
    if matrix is None:
        return []
    common, local = defaultdict(list), defaultdict(list)
    for i, definition in enumerate(definitions):
        target = common if definition.article["regulation"] == baseline else local
        target[definition.key].append(i)
    findings = []
    for key, indices in sorted(local.items()):
        if key not in common:
            missing = [definitions[i] for i in indices]
            if len({definition.article["regulation"] for definition in missing}) >= 2:
                term = missing[0].term
                findings.append(_finding(
                    "definition_missing_from_def", [d.article for d in missing],
                    [{"article_id": d.article["id"], "quote": d.quote} for d in missing],
                    f'"{term}" is defined in multiple regulations but absent from {baseline}',
                    "move_to_common", f"Add one definition to {baseline} and reference it from each regulation.",
                    0.7, {"term": term, "similarity": None},
                ))
            continue
        for i in indices:
            definition = definitions[i]
            scores = cosine_similarity(matrix[i:i + 1], matrix[common[key]]).ravel()
            best = int(scores.argmax())
            reference = definitions[common[key][best]]
            score = _score(scores[best])
            duplicate = score >= 0.80
            summary = (
                f'"{definition.term}" in {_label(definition.article)} repeats {_label(reference.article)}'
                if duplicate else
                f'"{definition.term}" is defined differently in {_label(definition.article)} and {_label(reference.article)}'
            )
            findings.append(_finding(
                "duplicate_definition" if duplicate else "divergent_definition",
                [definition.article, reference.article],
                [{"article_id": d.article["id"], "quote": d.quote} for d in (definition, reference)],
                summary, "cross_reference" if duplicate else "harmonize_wording",
                f"Delete the local definition and refer to {baseline}." if duplicate else
                f"Align with {baseline}, or keep with a documented sector-specific justification.",
                score, {"term": definition.term, "similarity": score},
            ))
    return findings


def _containment(left: str, right: str) -> float:
    words = [re.findall(r"[^\W_]+", _norm(text)) for text in (left, right)]
    shorter, longer = sorted(words, key=len)
    unique = set(shorter)
    return len(unique.intersection(longer)) / len(unique) if unique else 0.0


def _evidence(left: Article, right: Article) -> list[dict]:
    sentences = [[s.strip() for s in _SENTENCES.split(a["text"]) if s.strip()] for a in (left, right)]
    vectorizer, matrix = _vectors(sentences[0] + sentences[1])
    chosen = [0, 0]
    if matrix is not None:
        full = vectorizer.transform([_norm(left["text"]), _norm(right["text"])])
        split = len(sentences[0])
        chosen[0] = int(cosine_similarity(matrix[:split], full[1:2]).argmax())
        chosen[1] = int(cosine_similarity(matrix[split:], full[0:1]).argmax())
    return [{"article_id": article["id"], "quote": parts[index]}
            for article, parts, index in zip((left, right), sentences, chosen)]


def _thresholds(config):
    try:
        dup, overlap = float(config.get("dup", 0.85)), float(config.get("overlap", 0.55))
        if not (math.isfinite(dup) and math.isfinite(overlap) and 0 < overlap <= dup <= 1):
            raise ValueError("expected 0 < overlap <= dup <= 1")
        return dup, overlap
    except (TypeError, ValueError) as exc:
        _warn(f"invalid thresholds ({exc}); using dup=0.85, overlap=0.55.")
        return 0.85, 0.55


def _requirement_findings(articles, config):
    candidates = [a for a in articles if not a.get("is_definitions", False) and len(a["text"].split()) >= 15]
    dup, overlap = _thresholds(config)
    pairs = list(_pair_scores(candidates, overlap))
    parents = list(range(len(candidates)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for i, j, score in pairs:
        if score >= dup:
            parents[root(j)] = root(i)
    clusters = defaultdict(list)
    for i, article in enumerate(candidates):
        clusters[root(i)].append(article)
    findings = []
    for i, j, score in pairs:
        left, right = candidates[i], candidates[j]
        containment = _containment(left["text"], right["text"])
        extra = {"score": score, "containment": containment}
        duplicate = score >= dup
        if duplicate:
            cluster = clusters[root(i)]
            common = len({a["regulation"] for a in cluster}) >= 3
            recommendation = "move_to_common" if common else "cross_reference"
            rationale = (
                "This duplicate cluster spans at least three regulations; state the shared requirement once and reference it."
                if common else
                "Same requirement in two regulations; keep it in one (expert chooses the lead) and replace the copy with a reference."
            )
            if common:
                extra["cluster_article_ids"] = sorted(a["id"] for a in cluster)
        elif containment >= 0.8:
            recommendation = "cross_reference"
            rationale = "One article is largely contained in the other; review whether a reference can replace the repeated requirement."
        else:
            recommendation = "harmonize_wording"
            rationale = "The articles share part of a requirement; align the shared wording while preserving any differences in scope."
        findings.append(_finding(
            "duplicate_requirement" if duplicate else "partial_overlap", [left, right],
            _evidence(left, right),
            f"{_label(left)} and {_label(right)} are {'near-identical' if duplicate else 'partially overlapping'} (score {score:.2f})",
            recommendation, rationale, score, extra,
        ))
    return findings


def _merge_findings(findings):
    # Several terms may share the same contract-mandated type + article IDs.
    # Keep every term and quote instead of letting pipeline ID dedup discard them.
    merged = {}
    for finding in findings:
        previous = merged.get(finding["id"])
        if previous is None:
            merged[finding["id"]] = finding
            continue
        for evidence in finding["evidence"]:
            if evidence not in previous["evidence"]:
                previous["evidence"].append(evidence)
        terms = previous["extra"].setdefault("terms", [previous["extra"]["term"]])
        details = previous["extra"].setdefault("definitions", [{
            "term": previous["extra"]["term"], "similarity": previous["extra"]["similarity"],
        }])
        term = finding["extra"]["term"]
        if term not in terms:
            terms.append(term)
            previous["summary"] += "; " + finding["summary"]
        details.append({"term": term, "similarity": finding["extra"]["similarity"]})
        previous["confidence"] = min(previous["confidence"], finding["confidence"])
    return sorted(merged.values(), key=lambda f: (f["severity"] != "high", f["type"], f["id"]))


def detect(articles: list[Article], config: dict | None = None) -> list[Finding]:
    """Return review candidates using the shared Finding schema; never edit articles."""
    source = _articles(articles)
    if not source or not _available():
        return []
    config = config or {}
    definitions = _definitions(source)
    return _merge_findings(
        _definition_findings(source, definitions, config) + _requirement_findings(source, config)
    )


def _self_test(mock_articles: list[Article]) -> bool:
    """Acceptance and regression checks live here because P3 owns this file only."""
    import copy
    import unittest

    def article(reg, text, *, definitions=False, no="1"):
        return {"id": f"{reg}|{no}", "regulation": reg, "article_no": no,
                "text": text, "is_definitions": definitions}

    class SimilarityTests(unittest.TestCase):
        def test_mock_acceptance_and_contract(self):
            original = copy.deepcopy(mock_articles)
            findings = detect(mock_articles)
            expected = [
                ({"duplicate_requirement"}, {"CAR-OPS|5", "CAR-AIR|5"}, None),
                ({"partial_overlap", "duplicate_requirement"}, {"CAR-OPS|4", "CAR-AIR|4"}, None),
                ({"divergent_definition"}, {"CAR-OPS|2", "CAR-DEF|2"}, "Operator"),
                ({"duplicate_definition"}, {"CAR-AIR|2", "CAR-DEF|2"}, "Occurrence"),
                ({"definition_missing_from_def"}, {"CAR-OPS|2", "CAR-AIR|2"}, "Dry lease"),
            ]
            for types, ids, term in expected:
                self.assertTrue(any(f["type"] in types and set(f["article_ids"]) == ids
                                    and (term is None or f["extra"].get("term") == term) for f in findings),
                                f"Missing {types}: {ids} ({term})")
            sources = {a["id"]: a["text"] for a in mock_articles}
            self.assertEqual(len({f["id"] for f in findings}), len(findings))
            for finding in findings:
                self.assertEqual(finding["id"], finding["type"] + ":" + "+".join(sorted(finding["article_ids"])))
                self.assertFalse(finding["quotes_verified"])
                self.assertTrue(0 <= finding["confidence"] <= 1)
                for evidence in finding["evidence"]:
                    self.assertIn(evidence["quote"], sources[evidence["article_id"]])
            self.assertEqual(mock_articles, original)
            self.assertEqual(findings, detect(list(reversed(mock_articles))))

        def test_definition_forms_and_normalization(self):
            source = [
                article("CAR-DEF", '  (a) Operators means a person operating an aircraft.\n'
                        '• المُشَغِّل: الشخص المسؤول عن تشغيل الطائرة بأمان.', definitions=True),
                article("CAR-X", 'For this section, «Operator» shall mean a person operating an aircraft.\n'
                        'Unquoted means this is not a definition article.\n'
                        'المشغل: الشخص المسؤول عن تشغيل الطائرة بأمان.'),
            ]
            index = term_index(source)
            self.assertEqual(len(index["operator"]), 2)
            self.assertEqual(len(index["مشغل"]), 1)
            self.assertNotIn("unquoted", index)
            self.assertEqual(_norm("أَإِآٱ ى ة ـ ١٢"), "اااا ي ه  12")
            self.assertEqual(_term_key('"Safety-records"'), "safety record")
            self.assertEqual(_term_key("الجو"), "الجو")
            self.assertEqual(index["مشغل"][0]["text"], source[0]["text"].splitlines()[1])

        def test_missing_baseline_and_configured_baseline(self):
            text = '"Operator" means a person operating an aircraft.'
            sources = [article("COMMON", text, definitions=True), article("CAR-X", text, definitions=True)]
            self.assertEqual(detect(sources), [])
            findings = detect(sources, {"def_regulation": "common"})
            self.assertEqual([f["type"] for f in findings], ["duplicate_definition"])
            self.assertEqual(findings[0]["recommendation"], "cross_reference")

        def test_multiple_terms_keep_unique_finding_and_all_quotes(self):
            text = ('"Operator" means a person operating an aircraft.\n'
                    '"Aircraft" means a machine supported in the atmosphere.')
            findings = detect([article("CAR-DEF", text, definitions=True),
                               article("CAR-X", text, definitions=True)])
            self.assertEqual(len(findings), 1)
            self.assertEqual(set(findings[0]["extra"]["terms"]), {"Operator", "Aircraft"})
            self.assertEqual(len(findings[0]["evidence"]), 4)

        def test_missing_term_requires_distinct_regulations(self):
            sources = [article("CAR-DEF", '"Other" means an unrelated defined term.', definitions=True)]
            text = '"Dry lease" means an aircraft operated under the lessee certificate.'
            sources += [article("CAR-X", text, definitions=True, no=str(i)) for i in (1, 2)]
            self.assertEqual(detect(sources), [])
            sources.append(article("CAR-Y", text, definitions=True))
            findings = detect(sources)
            self.assertEqual(len(findings), 1)
            self.assertEqual(len(findings[0]["article_ids"]), 3)

        def test_three_regulation_cluster_and_candidate_filters(self):
            text = ("The operator shall establish, implement and maintain a safety management system "
                    "appropriate to the size and complexity of all its aircraft operations.")
            sources = [article(reg, text) for reg in ("CAR-A", "CAR-B", "CAR-C")]
            findings = detect(sources)
            self.assertEqual(len(findings), 3)
            self.assertTrue(all(f["recommendation"] == "move_to_common" for f in findings))
            self.assertTrue(all(len(f["extra"]["cluster_article_ids"]) == 3 for f in findings))
            same_reg = [article("CAR-A", text), article("CAR-A", text, no="2")]
            self.assertEqual(detect(same_reg), [])
            self.assertEqual(similar_pairs(same_reg), [])
            short = [article("CAR-A", "A short identical requirement."), article("CAR-B", "A short identical requirement.")]
            self.assertEqual(detect(short), [])
            self.assertEqual(len(similar_pairs(short, 1.0)), 1)

        def test_overlap_containment_and_sentence_evidence(self):
            base = ("The operator shall retain all aircraft maintenance records for a period of two years "
                    "after the aircraft has been permanently withdrawn from service.")
            unrelated = "Bananas grow in tropical orchards far from the northern ice fields."
            sources = [article("CAR-A", unrelated + "\n" + base), article("CAR-B", base)]
            finding = detect(sources, {"dup": 1.0, "overlap": 0.2})[0]
            self.assertEqual(finding["type"], "partial_overlap")
            self.assertEqual(finding["recommendation"], "cross_reference")
            self.assertEqual(finding["extra"]["containment"], 1.0)
            self.assertTrue(all(ev["quote"] == base for ev in finding["evidence"]))

        def test_empty_input_and_pair_order(self):
            self.assertEqual(detect([]), [])
            self.assertEqual(term_index([]), {})
            self.assertEqual(similar_pairs([]), [])
            self.assertEqual(detect([article("CAR-A", "   ")]), [])
            pairs = similar_pairs(mock_articles)
            self.assertEqual(pairs, sorted(pairs, key=lambda pair: (-pair[2], pair[0], pair[1])))
            self.assertTrue(all(a < b for a, b, _ in pairs))
            self.assertEqual(pairs, similar_pairs(list(reversed(mock_articles))))

        def test_pair_cap_and_arabic_requirements(self):
            text = "يجب على المشغل الاحتفاظ بجميع سجلات صيانة الطائرات لمدة خمس سنوات بعد سحب الطائرة نهائيا من الخدمة."
            sources = [article(f"CAR-{i:03}", text) for i in range(65)]
            self.assertEqual(len(similar_pairs(sources)), 2000)
            finding = detect(sources[:2])[0]
            self.assertEqual(finding["type"], "duplicate_requirement")
            self.assertTrue(all(ev["quote"] == text for ev in finding["evidence"]))

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SimilarityTests)
    return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("articles", nargs="?", default="data/mock_articles.json")
    parser.add_argument("--config", help="JSON file with dup, overlap and/or def_regulation")
    parser.add_argument("--test", action="store_true", help="run the mock acceptance and regression checks")
    args = parser.parse_args(argv)
    # Windows terminals may not support Arabic or smart quotes in source evidence.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        articles = json.loads(Path(args.articles).read_text(encoding="utf-8-sig"))
        if not isinstance(articles, list) or not all(isinstance(a, dict) for a in articles):
            raise ValueError("articles JSON must contain a list of article objects")
    except (OSError, ValueError) as exc:
        _warn(f"cannot read input: {exc}")
        return 1 if args.test else 0
    config = {}
    if args.config:
        try:
            config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
            if not isinstance(config, dict):
                raise ValueError("config JSON must contain an object")
        except (OSError, ValueError) as exc:
            _warn(f"cannot read optional config ({exc}); using defaults.")
            config = {}
    findings = detect(articles, config)
    print(json.dumps(findings, ensure_ascii=False, indent=2))
    print("\nTOP 20 cross-regulation pairs (lexical similarity):")
    for left, right, score in similar_pairs(articles)[:20]:
        print(f"  {score:.4f}  {left} + {right}")
    if args.test:
        if not _available():
            return 1
        return 0 if _self_test(articles) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
