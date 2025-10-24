### 🧩 Ontology-Aware Story Consistency Agent

**How to run:**

```bash
pip install rdflib owlready2 spacy
python -m spacy download en_core_web_sm
python main.py
```

**What it does:**

* Generates 3 stories (Barista, Firefighters, Doctor)
* Extracts RDF triples
* Runs ontology checks
* Detects & fixes logical errors
* Rewrites inconsistent stories
* Evaluates before/after consistency

**Output:**

* All results saved in `outputs/`
* Final metrics in `outputs/evaluation_report.txt`
* Expect OCS to rise (≈ 0.5 → 1.0) and all violations resolved ✅
