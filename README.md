# RegLens

GCAA "Innovation Takes Flight" · RegLens challenge (Safety sector).

RegLens reads the Civil Aviation Regulations (CARs), splits them into articles, and finds four kinds of problems: conflicting values, duplicate or divergent definitions, broken references, and duplicate or overlapping requirements. Each finding cites the regulation and article number, quotes the source text verbatim, and recommends an action. An expert reviews every finding. Official text is never modified.

## Run

```bash
pip install -r requirements.txt
python ingest.py data/raw/*          # regulations -> data/articles.json
python run_pipeline.py               # articles -> data/findings.json (+ references, similarity, terms)
python run_pipeline.py --mock        # same, on the fictional test data
python evaluate.py                   # recall vs data/expert_labels.csv
streamlit run app.py                 # expert review screen
```

## Who owns what

Each person commits only their own files. The shared formats and function signatures are in [CONTRACT.md](CONTRACT.md).

| Person | Files |
|---|---|
| P1 · Ingest | `ingest.py`, `run_pipeline.py`, `evaluate.py`, `requirements.txt` |
| P2 · Rules | `detect_rules.py` |
| P3 · Similarity | `detect_similarity.py` |
| P4 · App | `app.py` |
| P5 · AI | `ai_assist.py`, `prompts/` |

Test data: `data/mock_articles.json` and `data/mock_findings.json` (fictional, for testing only). Put the real regulation files in `data/raw/`.

Optional: `data/regulations.csv` with columns `file,regulation` overrides the regulation name taken from each file name.
