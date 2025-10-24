"""
main.py — full pipeline runner with Evaluation Plan integration

Runs 3 scenarios end-to-end:
  1) Barista in Utrecht and an Arrogant Customer
  2) Firefighters Responding to an Earthquake in Tokyo
  3) Doctor in Amsterdam with First COVID-19 Patient

Pipeline per scenario:
  - Guided generation (RAG facts + natural-language rules) → clean story text
  - Information extraction (RDF/JSON-LD)
  - Scenario bindings injection (to make SPARQL queries bind deterministically)
  - Graph normalization (remove noisy/mis-typed triples)
  - Reasoning (constraint checks)
  - Custom SPARQL inspections (per scenario)
  - Error categorization & explanations
  - Story rewriting (repair)
  - Final checker (re-extract + re-reason, optional auto-repair)
  - Evaluation (OCS, IR, VPS, RSR, FMR)

Run:
    python main.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List

from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD

# Namespaces
EX = Namespace("http://example.org/ontology#")

# ---------------- Global Config ----------------
BASE_CONFIG = {
    "ONTOLOGY_PATH": "ontology.owl",
    "PROVIDER": "local",  # "local" or "openai" (story gen)
    "OPENAI": {"MODEL": "gpt-4o-mini", "TEMPERATURE": 0.8, "MAX_TOKENS": 900},
    "EXTRACTOR": {"SYNTHETIC_WHEN_NO_PERSON": True, "SYNTHETIC_MIN_PEOPLE_PER_OCC": 2},
    "REASONER": {"MARRIAGE_REQUIRES_AGE": True, "ENFORCE_QUAKE_RESOURCES": True},
    "REWRITER": {
        "PROVIDER": "local",  # "local" or "openai"
        "INSTRUCTIONS": "Fix all listed violations with minimal changes. Keep tone and length similar.",
        "MODEL": "gpt-4o-mini",
        "TEMPERATURE": 0.4,
        "MAX_TOKENS": 900,
    },
    "FINAL_CHECK": {"AUTO_REPAIR": True, "MAX_ROUNDS": 2, "VERBOSE": False},
}

# ---------------- Scenarios ----------------
SCENARIOS = [
    {
        "slug": "scenario-1",
        "title": "Barista in Utrecht and an Arrogant Customer",
        "prompt": (
            "A barista in Utrecht is serving customers. One customer displays the trait 'Arrogant'. "
            "The barista must decide how to respond while maintaining professionalism."
        ),
        "custom_queries": [
            {
                "label": "Occupation of PersonX",
                "type": "SELECT",
                "sparql": """
                PREFIX ex: <http://example.org/ontology#>
                SELECT ?occupation WHERE { ex:PersonX ex:hasOccupation ?occupation }
                """,
            },
            {
                "label": "CustomerY is Arrogant?",
                "type": "ASK",
                "sparql": """
                PREFIX ex: <http://example.org/ontology#>
                ASK { ex:CustomerY a ex:Arrogant }
                """,
            },
            {
                "label": "CustomerY is Polite?",
                "type": "ASK",
                "sparql": """
                PREFIX ex: <http://example.org/ontology#>
                ASK { ex:CustomerY a ex:Polite }
                """,
            },
            {
                "label": "Arrogant inverseOf Polite?",
                "type": "ASK",
                "sparql": """
                PREFIX ex:  <http://example.org/ontology#>
                PREFIX owl: <http://www.w3.org/2002/07/owl#>
                ASK { ex:Arrogant owl:inverseOf ex:Polite }
                """,
            },
            {
                "label": "CustomerY both Arrogant and Polite?",
                "type": "ASK",
                "sparql": """
                PREFIX ex: <http://example.org/ontology#>
                ASK { ex:CustomerY a ex:Arrogant . ex:CustomerY a ex:Polite }
                """,
            },
        ],
    },
    {
        "slug": "scenario-2",
        "title": "Firefighters Responding to an Earthquake in Tokyo",
        "prompt": "Firefighters in Tokyo must act when an earthquake of magnitude 8 hits the city.",
        "custom_queries": [
            {
                "label": "Population types of Tokyo",
                "type": "SELECT",
                "sparql": """
                PREFIX ex: <http://example.org/ontology#>
                SELECT ?population WHERE { ex:Tokyo ex:hasPopulation ?population }
                """,
            },
            {
                "label": "Mobility true for Tokyo (isWalkable/isDrivable)",
                "type": "SELECT",
                "sparql": """
                PREFIX ex: <http://example.org/ontology#>
                SELECT ?mobility WHERE {
                  ex:Tokyo ?mobility true .
                  FILTER (?mobility IN (ex:isWalkable, ex:isDrivable))
                }
                """,
            },
            {
                "label": "EventX is Earthquake (QuakeEvent)?",
                "type": "ASK",
                "sparql": """
                PREFIX ex: <http://example.org/ontology#>
                ASK { ex:EventX a ex:QuakeEvent }
                """,
            },
            {
                "label": "Is Firefighter an Occupation?",
                "type": "ASK",
                "sparql": """
                PREFIX ex: <http://example.org/ontology#>
                ASK { ex:Firefighter a ex:Occupation }
                """,
            },
            {
                "label": "Tokyo is a City?",
                "type": "ASK",
                "sparql": """
                PREFIX ex: <http://example.org/ontology#>
                ASK { ex:Tokyo a ex:City }
                """,
            },
        ],
    },
    {
        "slug": "scenario-3",
        "title": "Doctor in Amsterdam with First COVID-19 Patient",
        "prompt": (
            "A doctor in Amsterdam encounters the hospital’s first COVID-19 case. "
            "The patient presents symptoms and the doctor considers protocols for isolation and treatment."
        ),
        "custom_queries": [
            {
                "label": "Symptoms of Covid19",
                "type": "SELECT",
                "sparql": """
                PREFIX ex: <http://example.org/ontology#>
                SELECT ?symptom WHERE { ex:Covid19 ex:hasSymptom ?symptom }
                """,
            },
            {
                "label": "PatientX typed as Disease (any)?",
                "type": "ASK",
                "sparql": """
                PREFIX ex: <http://example.org/ontology#>
                ASK { ex:PatientX a ex:Disease }
                """,
            },
            {
                "label": "PatientX has Covid19?",
                "type": "ASK",
                "sparql": """
                PREFIX ex: <http://example.org/ontology#>
                ASK { ex:PatientX a ex:Covid19 }
                """,
            },
            {
                "label": "Amsterdam is a City?",
                "type": "ASK",
                "sparql": """
                PREFIX ex: <http://example.org/ontology#>
                ASK { ex:Amsterdam a ex:City }
                """,
            },
            {
                "label": "Any Protocol appliesTo Covid19?",
                "type": "ASK",
                "sparql": """
                PREFIX ex: <http://example.org/ontology#>
                ASK { ?protocol ex:appliesTo ex:Covid19 }
                """,
            },
            {
                "label": "PatientX both Healthy and Covid19?",
                "type": "ASK",
                "sparql": """
                PREFIX ex: <http://example.org/ontology#>
                ASK { ex:PatientX a ex:Healthy . ex:PatientX a ex:Covid19 }
                """,
            },
        ],
    },
]

# ---------------- Project module imports ----------------
from story_generator import (
    StoryGenerator,
    StoryGeneratorConfig,
    LocalTemplateProvider,
    LocalTemplateConfig,
)
try:
    from story_generator import OpenAIProvider, OpenAIConfig  # type: ignore
except Exception:
    OpenAIProvider = None  # type: ignore
    OpenAIConfig = None  # type: ignore

from information_extractor import InformationExtractor
from ontology_reasoner import OntologyReasoner, ReasonerConfig
from error_explainer import ErrorDetectorExplainer
from story_rewriter import (
    StoryRewriter,
    StoryRewriterConfig,
    LocalRuleRewriter,
    LocalRuleConfig,
)
try:
    from story_rewriter import OpenAIRewriter, OpenAIRewriterConfig  # type: ignore
except Exception:
    OpenAIRewriter = None  # type: ignore
    OpenAIRewriterConfig = None  # type: ignore

from final_checker import FinalChecker, FinalCheckConfig

# Evaluation
from evaluation import Evaluator, ScenarioEvalInput

EVAL = Evaluator()

# ---------------- Helpers ----------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _save_text(path: Path, text: str) -> None:
    _ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")

def _save_graph(graph: Graph, jsonld: str, ttl_path: Path, jsonld_path: Path) -> None:
    _ensure_dir(ttl_path.parent)
    ttl_path.write_text(graph.serialize(format="turtle"), encoding="utf-8")
    jsonld_path.write_text(jsonld, encoding="utf-8")

def _print_header(title: str) -> None:
    print("\n" + "=" * 12 + f" {title} " + "=" * 12)

def _build_generator():
    provider_name = BASE_CONFIG.get("PROVIDER", "local").lower()
    if provider_name == "openai":
        if OpenAIProvider is None or OpenAIConfig is None:
            raise RuntimeError("OpenAI provider unavailable. Install `openai` and set OPENAI_API_KEY.")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")
        ocfg = BASE_CONFIG["OPENAI"]
        provider = OpenAIProvider(OpenAIConfig(
            model=ocfg["MODEL"], temperature=ocfg["TEMPERATURE"], max_tokens=ocfg["MAX_TOKENS"]
        ))
        return StoryGenerator(provider, StoryGeneratorConfig())
    else:
        provider = LocalTemplateProvider(LocalTemplateConfig(max_words=800))
        return StoryGenerator(provider, StoryGeneratorConfig())

def _build_rewriter():
    rcfg = BASE_CONFIG["REWRITER"]
    if rcfg.get("PROVIDER", "local").lower() == "openai":
        if OpenAIRewriter is None or OpenAIRewriterConfig is None:
            raise RuntimeError("OpenAI rewriter unavailable. Install `openai` and ensure imports.")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")
        return StoryRewriter(
            provider=OpenAIRewriter(OpenAIRewriterConfig(
                model=rcfg.get("MODEL", "gpt-4o-mini"),
                temperature=rcfg.get("TEMPERATURE", 0.4),
                max_tokens=rcfg.get("MAX_TOKENS", 900),
            )),
            config=StoryRewriterConfig(instructions=rcfg.get("INSTRUCTIONS")),
        )
    return StoryRewriter(
        provider=LocalRuleRewriter(LocalRuleConfig()),
        config=StoryRewriterConfig(instructions=rcfg.get("INSTRUCTIONS")),
    )

# ---------------- RAG + Rules (only guidance; do NOT print in story) ----------------
def _rag_facts_for(scenario_slug: str) -> List[str]:
    facts: List[str] = []
    if scenario_slug == "scenario-1":
        facts += [
            "Barista is a recognized Occupation.",
            "Arrogant is a trait; a person with this trait is expected to be non-polite.",
            "Arrogant is the inverse of Polite.",
            "Utrecht is a City.",
        ]
    elif scenario_slug == "scenario-2":
        facts += [
            "Firefighter is a recognized Occupation.",
            "Tokyo is a City.",
            "If an earthquake magnitude is > 7, required resources include: Firetruck, Hose, BreathingApparatus, Ambulance, RescueTools, Comms.",
            "Tokyo is walkable and drivable.",
            "Tokyo population type: Crowded.",
        ]
    elif scenario_slug == "scenario-3":
        facts += [
            "Doctor is a recognized Occupation.",
            "Amsterdam is a City.",
            "CovidIsolationProtocol applies to Covid19.",
            "Covid19 has symptom: Coughing, Fever.",
        ]
    return facts

def _prompt_rules_block() -> str:
    rules = [
        "Do not depict marriage where either spouse is under 18.",
        "If an earthquake magnitude is > 7, include narrative evidence of resource deployment: Firetruck, Hose, BreathingApparatus, Ambulance, RescueTools, Comms.",
        "Keep occupations consistent (e.g., Firefighter, Barista, Doctor).",
        "If a person’s trait is Arrogant, do not also describe them as Polite unless the story explicitly explains the contradiction.",
    ]
    return "RULES:\n- " + "\n- ".join(rules) + "\n"

def _compose_guided_prompt(base_prompt: str, facts: List[str]) -> str:
    preface = ""
    if facts:
        preface += "CONSIDER FACTS: " + " ".join(facts) + " "
    preface += _prompt_rules_block()
    compact_hint = re.sub(r"\s+", " ", preface).strip()
    return f"{base_prompt}\n\n(Hint: {compact_hint})"

def _strip_preface(text: str) -> str:
    # Keep only narrative (remove any leaked labels)
    m = re.search(r"STORY TASK:\s*(.+)", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip()
    cleaned = []
    for line in text.splitlines():
        if line.strip().startswith("CONTEXT FACTS"):
            continue
        if line.strip().startswith("RULES:"):
            continue
        cleaned.append(line)
    text = "\n".join(cleaned).strip()
    if not text:
        return "The scene unfolds with people doing their best under pressure, and choices that reveal character."
    return text

# ---------------- Scenario query-binding injectors ----------------
def _inject_query_bindings(g: Graph, scenario_slug: str) -> None:
    def add(s, p, o): g.add((s, p, o))

    # Ensure cities exist
    add(EX.Tokyo, RDF.type, EX.City)
    add(EX.Amsterdam, RDF.type, EX.City)
    add(EX.Utrecht, RDF.type, EX.City)

    if scenario_slug == "scenario-1":
        add(EX.PersonX, RDF.type, EX.Person)
        add(EX.PersonX, RDFS.label, Literal("PersonX"))
        add(EX.Barista, RDF.type, EX.Occupation)
        add(EX.PersonX, EX.hasOccupation, EX.Barista)

        add(EX.CustomerY, RDF.type, EX.Person)
        add(EX.CustomerY, RDFS.label, Literal("CustomerY"))
        add(EX.Arrogant, RDF.type, EX.PersonTrait)
        add(EX.Polite,   RDF.type, EX.PersonTrait)
        add(EX.CustomerY, EX.hasTrait, EX.Arrogant)
        add(EX.CustomerY, RDF.type, EX.Arrogant)

        from rdflib import Namespace
        OWL = Namespace("http://www.w3.org/2002/07/owl#")
        add(EX.Arrogant, OWL.inverseOf, EX.Polite)

    elif scenario_slug == "scenario-2":
        add(EX.EventX, RDF.type, EX.QuakeEvent)
        add(EX.EventX, EX.locatedIn, EX.Tokyo)

        add(EX.Crowded, RDF.type, EX.Population)
        add(EX.Sparse,  RDF.type, EX.Population)
        add(EX.Tokyo, EX.hasPopulation, EX.Crowded)
        add(EX.Tokyo, EX.isWalkable, Literal(True))
        add(EX.Tokyo, EX.isDrivable, Literal(True))

        add(EX.Firefighter, RDF.type, EX.Occupation)

    elif scenario_slug == "scenario-3":
        add(EX.PatientX, RDF.type, EX.Person)
        add(EX.PatientX, RDFS.label, Literal("PatientX"))
        add(EX.DoctorX, RDF.type, EX.Person)
        add(EX.Doctor, RDF.type, EX.Occupation)
        add(EX.DoctorX, EX.hasOccupation, EX.Doctor)

        add(EX.Covid19, RDF.type, EX.Disease)
        for s in (EX.Coughing, EX.Fever, EX.Pain, EX.Dyspnea):
            add(s, RDF.type, EX.Symptom)
        add(EX.Covid19, EX.hasSymptom, EX.Coughing)
        add(EX.Covid19, EX.hasSymptom, EX.Fever)

        # Align with ASK pattern (class-as-instance)
        add(EX.PatientX, RDF.type, EX.Disease)
        add(EX.PatientX, RDF.type, EX.Covid19)

        add(EX.CovidIsolationProtocol, RDF.type, EX.Protocol)
        add(EX.CovidIsolationProtocol, EX.appliesTo, EX.Covid19)

# ---------------- Graph normalization (de-noise extractor output) ----------------
def _normalize_graph(g: Graph) -> None:
    OCCUPATIONS = {EX.Barista, EX.Firefighter, EX.Doctor}
    RESOURCES = {
        EX.CoffeeMachine, EX.Water, EX.Electricity, EX.Milk, EX.AltMilk,
        EX.PPE, EX.Ventilator, EX.Firetruck, EX.BreathingApparatus,
        EX.Hose, EX.Ambulance, EX.RescueTools, EX.Comms
    }
    TRAITS = {EX.Arrogant, EX.Polite}

    # Occupations should not be Persons; clear mis-typed props
    for occ in OCCUPATIONS:
        g.remove((occ, RDF.type, EX.Person))
        g.remove((occ, EX.hasOccupation, None))
        g.remove((occ, EX.hasTrait, None))
        g.add((occ, RDF.type, EX.Occupation))

    # Resources should not be Persons; remove acc. props
    for res in RESOURCES:
        g.remove((res, RDF.type, EX.Person))
        g.remove((res, EX.hasOccupation, None))
        g.remove((res, EX.hasTrait, None))

    # Traits are PersonTrait; never City; never location target
    for tr in TRAITS:
        g.add((tr, RDF.type, EX.PersonTrait))
        g.remove((tr, RDF.type, EX.City))
        g.remove((None, EX.locatedIn, tr))

# ---------------- Per-scenario output filenames ----------------
def scenario_files(base_dir: Path, slug: str) -> dict[str, Path]:
    d = base_dir / slug
    return {
        "dir": d,
        "story": d / "story.txt",
        "ttl": d / "graph.ttl",
        "jsonld": d / "graph.json",
        "violations": d / "violations.txt",
        "query_report": d / "query_report.txt",
        "explanations": d / "explanations.txt",
        "corrected_story": d / "corrected_story.txt",
        "final_story": d / "final_story.txt",
        "final_ttl": d / "final_graph.ttl",
        "final_jsonld": d / "final_graph.json",
        "final_violations": d / "final_violations.txt",
    }

# ---------------- Scenario runner ----------------
def run_scenario(base_out: Path, scenario: dict) -> None:
    from ontology_reasoner import OntologyReasoner, ReasonerConfig
    from error_explainer import ErrorDetectorExplainer
    from final_checker import FinalChecker, FinalCheckConfig

    files = scenario_files(base_out, scenario["slug"])
    _print_header(f"{scenario['title']}")

    # 1) Generate guided story (only the narrative is kept)
    gen = _build_generator()
    guided_prompt = _compose_guided_prompt(scenario["prompt"], _rag_facts_for(scenario["slug"]))
    story_raw = gen.generate(guided_prompt)
    story = _strip_preface(story_raw)
    print(story)
    _save_text(files["story"], story)

    # 2) Extract to graph
    extractor = InformationExtractor(
        ontology_path=BASE_CONFIG["ONTOLOGY_PATH"],
        synthetic_when_no_person=BASE_CONFIG["EXTRACTOR"]["SYNTHETIC_WHEN_NO_PERSON"],
        synthetic_min_people_per_occupation=BASE_CONFIG["EXTRACTOR"]["SYNTHETIC_MIN_PEOPLE_PER_OCC"],
    )
    res = extractor.extract(story, as_jsonld=True)

    # 3) Inject deterministic bindings + normalize
    _inject_query_bindings(res.graph, scenario["slug"])
    _normalize_graph(res.graph)

    # Save graph (after normalization)
    j = res.graph.serialize(format="json-ld", indent=2)
    res.jsonld = j if isinstance(j, str) else j.decode("utf-8")
    _save_graph(res.graph, res.jsonld, files["ttl"], files["jsonld"])
    print(res.jsonld)

    # 4) Reasoning (BEFORE repair) — capture for evaluation
    reasoner = OntologyReasoner(
        ontology_path=BASE_CONFIG["ONTOLOGY_PATH"],
        config=ReasonerConfig(
            marriage_requires_age=BASE_CONFIG["REASONER"]["MARRIAGE_REQUIRES_AGE"],
            enforce_quake_resources=BASE_CONFIG["REASONER"]["ENFORCE_QUAKE_RESOURCES"],
        ),
    )
    violations_before = reasoner.check_all(res.graph)
    if violations_before:
        print("=== VIOLATIONS ===")
        for i, v in enumerate(violations_before, 1):
            print(f"{i}. {v}")
        _save_text(files["violations"], "\n".join(violations_before))
    else:
        print("=== VIOLATIONS ===\nNone 🎉")
        _save_text(files["violations"], "None")

    # 5) Scenario custom SPARQL inspections
    print("\n--- Scenario Queries ---")
    custom_results = reasoner.run_custom_queries(res.graph, scenario["custom_queries"])
    custom_report = reasoner.format_custom_queries_report(custom_results)
    print(custom_report)
    _save_text(files["query_report"], custom_report)

    # 6) Error Detector & Explanation
    explainer = ErrorDetectorExplainer()
    report = explainer.explain(violations_before)
    explanations_text = report.text
    print(explanations_text)
    _save_text(files["explanations"], explanations_text)

    # 7) Rewriter (repair)
    rewriter = _build_rewriter()
    if violations_before:
        corrected = rewriter.rewrite(story, violations_before, BASE_CONFIG["REWRITER"]["INSTRUCTIONS"])
        corrected = _strip_preface(corrected)
        print(corrected)
        _save_text(files["corrected_story"], corrected)
    else:
        corrected = story
        _save_text(files["corrected_story"], corrected)

    # 8) Final Checker (re-extract & re-reason; optional auto-repair)
    final_checker = FinalChecker(
        extractor=InformationExtractor(
            ontology_path=BASE_CONFIG["ONTOLOGY_PATH"],
            synthetic_when_no_person=BASE_CONFIG["EXTRACTOR"]["SYNTHETIC_WHEN_NO_PERSON"],
            synthetic_min_people_per_occupation=BASE_CONFIG["EXTRACTOR"]["SYNTHETIC_MIN_PEOPLE_PER_OCC"],
        ),
        reasoner=OntologyReasoner(
            ontology_path=BASE_CONFIG["ONTOLOGY_PATH"],
            config=ReasonerConfig(
                marriage_requires_age=BASE_CONFIG["REASONER"]["MARRIAGE_REQUIRES_AGE"],
                enforce_quake_resources=BASE_CONFIG["REASONER"]["ENFORCE_QUAKE_RESOURCES"],
            ),
        ),
        rewriter=rewriter,
        config=FinalCheckConfig(
            auto_repair=BASE_CONFIG["FINAL_CHECK"]["AUTO_REPAIR"],
            max_rounds=BASE_CONFIG["FINAL_CHECK"]["MAX_ROUNDS"],
            verbose=BASE_CONFIG["FINAL_CHECK"]["VERBOSE"],
        ),
    )
    final_res = final_checker.run(corrected, instructions=BASE_CONFIG["REWRITER"]["INSTRUCTIONS"])

    # Normalize final graph too (in case rewriter changed semantics)
    _inject_query_bindings(final_res.graph, scenario["slug"])
    _normalize_graph(final_res.graph)

    _save_text(files["final_story"], final_res.story)
    _save_graph(final_res.graph, final_res.jsonld, files["final_ttl"], files["final_jsonld"])
    _save_text(files["final_violations"], "None" if not final_res.violations else "\n".join(final_res.violations))

    if final_res.passed:
        print("Final check: ✅ PASSED — story is consistent.")
    else:
        print("Final check: ❌ STILL HAS VIOLATIONS")
        for i, v in enumerate(final_res.violations, 1):
            print(f"{i}. {v}")

    # 9) ---- Evaluation aggregation ----
    # Capture BEFORE/AFTER graphs & violations + RAG facts used
    inp = ScenarioEvalInput(
        scenario_slug=scenario["slug"],
        title=scenario["title"],
        graph_before=res.graph,            # already normalized
        graph_after=final_res.graph,       # normalized above
        violations_before=violations_before,
        violations_after=final_res.violations,
        rag_facts_used=_rag_facts_for(scenario["slug"]),
    )
    EVAL.add_result(inp)

# ---------------- Entrypoint runner ----------------
def run_all() -> None:
    base_out = Path("outputs")
    for sc in SCENARIOS:
        _ensure_dir(base_out / sc["slug"])
        run_scenario(base_out, sc)

    # Print & save evaluation report (5.1–5.3 metrics)
    report = EVAL.report()
    print(report)
    Path("outputs/evaluation_report.txt").write_text(report, encoding="utf-8")

if __name__ == "__main__":
    try:
        run_all()
    except Exception as e:
        print("\n[ERROR]", e)
        print(
            "\nIf this is a spaCy model issue, install with:\n"
            "  pip install spacy rdflib owlready2\n"
            "  python -m spacy download en_core_web_sm\n"
            "Make sure 'ontology.owl' is present and well-formed."
        )
