# RegLens shared contract

Do not change without the lead. Extra fields go in `extra`.

```text
=== REGLENS SHARED CONTRACT (do not change; extra fields go in "extra") ===
Project: RegLens, a tool for GCAA (UAE) that detects duplication, overlap and conflict across Civil Aviation Regulations (CARs) and recommends an action per case. It SUPPORTS the expert and never replaces them. It NEVER edits official text. Every result is traceable to regulation + article number.

Language: Python 3.10+. Libraries: pypdf, python-docx, scikit-learn, pandas, openpyxl, streamlit, anthropic (optional), openai (optional). Text is Arabic and/or English.

Repo layout:
  data/raw/                 original regulation files (pdf/docx/txt)
  data/articles.json        list[Article]       (written by ingest.py)
  data/findings.json        list[Finding]       (written by run_pipeline.py)
  data/reviews.json         {finding_id: Review} (written by app.py)
  data/references.json      list[Reference]     (written by run_pipeline.py via detect_rules.references)
  data/similarity.json      list[[id_a,id_b,score]] (written by run_pipeline.py via detect_similarity.similar_pairs)
  data/terms.json           {term_key: [{article_id, term, text}]} (written by run_pipeline.py via detect_similarity.term_index)
  data/expert_labels.csv    columns: article_a,article_b,type
  data/evaluation.json      written by evaluate.py
  data/mock_articles.json   fixture below (create it if missing)
  data/mock_findings.json   fixture below (create it if missing)
  ingest.py  detect_rules.py  detect_similarity.py  ai_assist.py  evaluate.py  run_pipeline.py  app.py  prompts/

Function signatures (exact):
  ingest.parse_documents(paths: list[str]) -> list[Article]
  detect_rules.detect(articles: list[Article], config: dict | None = None) -> list[Finding]
  detect_rules.references(articles: list[Article]) -> list[Reference]
  detect_similarity.detect(articles: list[Article], config: dict | None = None) -> list[Finding]
  detect_similarity.similar_pairs(articles: list[Article], min_score: float = 0.25) -> list[tuple[str, str, float]]
  detect_similarity.term_index(articles: list[Article]) -> dict
  ai_assist.verify_quotes(findings: list[Finding], articles: list[Article]) -> list[Finding]
  ai_assist.build_prompt(finding: Finding, articles: list[Article], kind: str = "unified_wording") -> str
  ai_assist.draft(finding: Finding, articles: list[Article], kind: str = "unified_wording") -> dict
  evaluate.evaluate(findings: list[Finding], labels_path: str) -> dict

Article (dict):
{
  "id": "CAR-OPS|4",              # "<regulation>|<article_no>", unique
  "regulation": "CAR-OPS",
  "article_no": "4",               # as printed; Arabic-Indic digits converted to 0-9; Arabic ordinal words kept (e.g. "الأولى")
  "title": "Retention of records",
  "text": "...",                   # body text, original wording (only NFKC-normalized). Quotes must be exact substrings of this.
  "lang": "en" | "ar",
  "source_file": "CAR-OPS.pdf",
  "page": 3 | null,
  "is_definitions": true | false
}
Display format for humans: "CAR-OPS · Art. 4".

Finding (dict):
{
  "id": "<type>:<sorted article ids joined by +>",   # deterministic, e.g. "conflicting_values:CAR-AIR|4+CAR-OPS|4"
  "type": one of TYPES,
  "severity": "high" | "medium" | "low",
  "article_ids": ["CAR-OPS|4", "CAR-AIR|4"],
  "evidence": [{"article_id": "CAR-OPS|4", "quote": "<exact sentence copied from article text>"}],
  "summary": "one line a human understands",
  "recommendation": one of RECOMMENDATIONS,
  "rationale": "one or two sentences: why this recommendation",
  "detector": "rules" | "similarity",
  "confidence": 0.0-1.0,
  "quotes_verified": true | false,   # set by ai_assist.verify_quotes
  "extra": {}                        # anything detector-specific
}
TYPES: conflicting_values, duplicate_requirement, partial_overlap, duplicate_definition, divergent_definition, definition_missing_from_def, broken_reference, unverifiable_reference, icao_edition_issue
RECOMMENDATIONS: move_to_common, cross_reference, harmonize_wording, keep_with_justification, correct_reference
Severity order for sorting: high > medium > low.

Reference (dict): {"from": "CAR-OPS|6", "to_regulation": "CAR-OPS", "to_article": "14", "resolved_id": "CAR-OPS|14" | null, "quote": "<sentence>"}

Review (dict, in reviews.json keyed by finding id):
{"decision": "pending"|"accepted"|"modified"|"rejected", "final_recommendation": one of RECOMMENDATIONS, "note": "", "reviewer": "", "ai_draft": {} | null, "timestamp": "ISO-8601"}

Text normalization used for MATCHING only (never for quotes):
  NFKC; lowercase; Arabic-Indic digits ٠-٩ -> 0-9; remove tashkeel U+064B-U+0652, U+0670 and tatweel U+0640; أ إ آ ٱ -> ا ; ى -> ي ; ة -> ه
  Sentence split: re.split(r'(?<=[.!?؟؛;])\s+|\n+', text)

Rules for every module:
  - Never crash on missing optional files or missing modules: catch, print a warning, continue.
  - Evidence quotes are copied from Article.text, never paraphrased.
  - The code must run with: python <module>.py data/mock_articles.json (detectors) or data/mock_findings.json (ai_assist).

FIXTURE data/mock_articles.json (fictional, for testing only):
[
 {"id":"CAR-DEF|2","regulation":"CAR-DEF","article_no":"2","title":"Definitions","text":"\"Operator\" means a person, organisation or enterprise engaged in or offering to engage in an aircraft operation.\n\"Occurrence\" means any safety-related event which endangers or could endanger an aircraft, its occupants or any other person.","lang":"en","source_file":"mock","page":null,"is_definitions":true},
 {"id":"CAR-OPS|2","regulation":"CAR-OPS","article_no":"2","title":"Definitions","text":"\"Operator\" means any person or company that operates an aircraft for commercial air transport.\n\"Dry lease\" means an agreement under which the aircraft is operated under the certificate of the lessee.","lang":"en","source_file":"mock","page":null,"is_definitions":true},
 {"id":"CAR-OPS|3","regulation":"CAR-OPS","article_no":"3","title":"Occurrence reporting","text":"The operator shall notify the Authority of any occurrence within 72 hours of the event becoming known.","lang":"en","source_file":"mock","page":null,"is_definitions":false},
 {"id":"CAR-OPS|4","regulation":"CAR-OPS","article_no":"4","title":"Retention of records","text":"The operator shall retain all aircraft maintenance records for a period of 2 years after the aircraft has been permanently withdrawn from service.","lang":"en","source_file":"mock","page":null,"is_definitions":false},
 {"id":"CAR-OPS|5","regulation":"CAR-OPS","article_no":"5","title":"Safety management system","text":"The operator shall establish, implement and maintain a safety management system appropriate to the size and complexity of its operations. The system shall include a safety policy, hazard identification, risk assessment, safety assurance and safety promotion.","lang":"en","source_file":"mock","page":null,"is_definitions":false},
 {"id":"CAR-OPS|6","regulation":"CAR-OPS","article_no":"6","title":"Operations manual","text":"The operator shall maintain an operations manual in accordance with CAR-OPS Article 14 and ICAO Annex 6, Part I, 10th Edition.","lang":"en","source_file":"mock","page":null,"is_definitions":false},
 {"id":"CAR-AIR|2","regulation":"CAR-AIR","article_no":"2","title":"Definitions","text":"\"Occurrence\" means any safety-related event which endangers or could endanger an aircraft, its occupants or any other person.\n\"Dry lease\" means an arrangement in which an aircraft is leased without crew and operated under the lessee's certificate.","lang":"en","source_file":"mock","page":null,"is_definitions":true},
 {"id":"CAR-AIR|3","regulation":"CAR-AIR","article_no":"3","title":"Reporting","text":"The organisation shall report to the Authority any occurrence affecting airworthiness within 24 hours.","lang":"en","source_file":"mock","page":null,"is_definitions":false},
 {"id":"CAR-AIR|4","regulation":"CAR-AIR","article_no":"4","title":"Maintenance records","text":"Maintenance records shall be kept for a minimum of 3 years after the aircraft has been permanently withdrawn from service.","lang":"en","source_file":"mock","page":null,"is_definitions":false},
 {"id":"CAR-AIR|5","regulation":"CAR-AIR","article_no":"5","title":"Safety management","text":"The organisation shall establish, implement and maintain a safety management system appropriate to the size and complexity of the organisation. The system shall include a safety policy, hazard identification, risk assessment, safety assurance and safety promotion.","lang":"en","source_file":"mock","page":null,"is_definitions":false},
 {"id":"CAR-AIR|6","regulation":"CAR-AIR","article_no":"6","title":"Standards","text":"Continuing airworthiness shall be managed in accordance with ICAO Annex 6, Part I, 12th Edition and CAR-OPS Article 9.","lang":"en","source_file":"mock","page":null,"is_definitions":false},
 {"id":"CAR-AOC|2","regulation":"CAR-AOC","article_no":"2","title":"سجلات الصيانة","text":"يلتزم المشغل بالاحتفاظ بسجلات الصيانة لمدة (5) سنوات على الأقل.","lang":"ar","source_file":"mock","page":null,"is_definitions":false}
]

EXPECTED FINDINGS on the mock (all modules together):
  conflicting_values       CAR-OPS|4 + CAR-AIR|4 + CAR-AOC|2   (maintenance records 2 / 3 / 5 years)
  conflicting_values       CAR-OPS|3 + CAR-AIR|3               (occurrence reporting 72 h vs 24 h)
  broken_reference         CAR-OPS|6 -> CAR-OPS Article 14 (does not exist)
  broken_reference         CAR-AIR|6 -> CAR-OPS Article 9  (does not exist)
  icao_edition_issue       CAR-OPS|6 + CAR-AIR|6               (Annex 6: 10th vs 12th edition)
  duplicate_requirement    CAR-OPS|5 + CAR-AIR|5               (SMS)
  partial_overlap          CAR-OPS|4 + CAR-AIR|4               (acceptable if scored as duplicate)
  divergent_definition     "Operator"  CAR-OPS|2 vs CAR-DEF|2
  duplicate_definition     "Occurrence" CAR-AIR|2 vs CAR-DEF|2
  definition_missing_from_def "Dry lease" CAR-OPS|2 + CAR-AIR|2

FIXTURE data/mock_findings.json (fictional, for UI/AI testing only):
[
 {"id":"conflicting_values:CAR-AIR|4+CAR-AOC|2+CAR-OPS|4","type":"conflicting_values","severity":"high","article_ids":["CAR-OPS|4","CAR-AIR|4","CAR-AOC|2"],"evidence":[{"article_id":"CAR-OPS|4","quote":"The operator shall retain all aircraft maintenance records for a period of 2 years after the aircraft has been permanently withdrawn from service."},{"article_id":"CAR-AIR|4","quote":"Maintenance records shall be kept for a minimum of 3 years after the aircraft has been permanently withdrawn from service."},{"article_id":"CAR-AOC|2","quote":"يلتزم المشغل بالاحتفاظ بسجلات الصيانة لمدة (5) سنوات على الأقل."}],"summary":"Maintenance record retention: CAR-OPS Art.4 = 2 years; CAR-AIR Art.4 = 3 years; CAR-AOC Art.2 = 5 years","recommendation":"move_to_common","rationale":"Same obligation stated with three different periods in three regulations; the expert must choose one value and state it once.","detector":"rules","confidence":0.9,"quotes_verified":true,"extra":{"topic":"retention","object":"maintenance"}},
 {"id":"divergent_definition:CAR-DEF|2+CAR-OPS|2","type":"divergent_definition","severity":"high","article_ids":["CAR-OPS|2","CAR-DEF|2"],"evidence":[{"article_id":"CAR-OPS|2","quote":"\"Operator\" means any person or company that operates an aircraft for commercial air transport."},{"article_id":"CAR-DEF|2","quote":"\"Operator\" means a person, organisation or enterprise engaged in or offering to engage in an aircraft operation."}],"summary":"\"Operator\" is defined differently in CAR-OPS Art.2 and CAR-DEF Art.2","recommendation":"harmonize_wording","rationale":"CAR-DEF is the reference for definitions; align the CAR-OPS definition or justify a sector-specific meaning.","detector":"similarity","confidence":0.42,"quotes_verified":true,"extra":{"term":"Operator","similarity":0.42}},
 {"id":"broken_reference:CAR-OPS|6","type":"broken_reference","severity":"high","article_ids":["CAR-OPS|6"],"evidence":[{"article_id":"CAR-OPS|6","quote":"The operator shall maintain an operations manual in accordance with CAR-OPS Article 14 and ICAO Annex 6, Part I, 10th Edition."}],"summary":"CAR-OPS Art.6 refers to CAR-OPS Article 14, which does not exist","recommendation":"correct_reference","rationale":"The referenced article is not in the regulation; it may have been repealed, moved or renumbered.","detector":"rules","confidence":0.95,"quotes_verified":true,"extra":{"to_regulation":"CAR-OPS","to_article":"14"}}
]
=== END CONTRACT ===
```
