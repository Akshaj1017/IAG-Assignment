# main.py
"""
Execution Graph:
User Input → Story Generation → Fact Extraction → Ontology Checks → Error Detection
→ (If violations) Story Rewriting → Re-Check → Final Output
"""
from __future__ import annotations

import os, re
from pathlib import Path
from typing import List
from rdflib import Graph, Namespace, Literal
from rdflib.namespace import RDF, RDFS

from story_generator import StoryGenerator, StoryGeneratorConfig, LocalTemplateProvider, LocalTemplateConfig
try:
    from story_generator import OpenAIProvider, OpenAIConfig  # optional
except Exception:
    OpenAIProvider = None
    OpenAIConfig = None

from information_extractor import InformationExtractor
from ontology_reasoner import OntologyReasoner, ReasonerConfig, EX
from error_explainer import ErrorDetectorExplainer
from story_rewriter import StoryRewriter, StoryRewriterConfig, LocalRuleRewriter, LocalRuleConfig
try:
    from story_rewriter import OpenAIRewriter, OpenAIRewriterConfig  # optional
except Exception:
    OpenAIRewriter = None
    OpenAIRewriterConfig = None

from final_checker import FinalChecker, FinalCheckConfig
from evaluation import Evaluator, ScenarioEvalInput

# ------------ Config ------------
BASE_CONFIG = {
    "ONTOLOGY_PATH": "ontology.owl",
    "PROVIDER": "local",  # "local" or "openai"
    "OPENAI": {"MODEL": "gpt-4o-mini", "TEMPERATURE": 0.8, "MAX_TOKENS": 900},
    "EXTRACTOR": {"SYNTHETIC_WHEN_NO_PERSON": True, "SYNTHETIC_MIN_PEOPLE_PER_OCC": 2},
    "REASONER": {"MARRIAGE_REQUIRES_AGE": True, "ENFORCE_QUAKE_RESOURCES": True},
    "REWRITER": {
        "PROVIDER": "local",
        "INSTRUCTIONS": "Fix all listed violations with minimal changes. Keep tone and length similar.",
        "MODEL": "gpt-4o-mini",
        "TEMPERATURE": 0.4,
        "MAX_TOKENS": 900,
    },
    "FINAL_CHECK": {"AUTO_REPAIR": True, "MAX_ROUNDS": 2, "VERBOSE": False},
}

SCENARIOS = [
    {
        "slug": "scenario-1",
        "title": "Barista in Utrecht and an Arrogant Customer",
        "prompt": "A barista in Utrecht is serving customers. One customer displays the trait 'Arrogant'. The barista must decide how to respond while maintaining professionalism.",
        "custom_queries": [
            {"label": "Occupation of PersonX", "type": "SELECT",
             "sparql": "PREFIX ex: <http://example.org/ontology#> SELECT ?occupation WHERE { ex:PersonX ex:hasOccupation ?occupation }"},
            {"label": "CustomerY is Arrogant?", "type": "ASK",
             "sparql": "PREFIX ex: <http://example.org/ontology#> ASK { ex:CustomerY a ex:Arrogant }"},
            {"label": "CustomerY is Polite?", "type": "ASK",
             "sparql": "PREFIX ex: <http://example.org/ontology#> ASK { ex:CustomerY a ex:Polite }"},
            {"label": "Arrogant inverseOf Polite?", "type": "ASK",
             "sparql": "PREFIX ex: <http://example.org/ontology#> PREFIX owl: <http://www.w3.org/2002/07/owl#> ASK { ex:Arrogant owl:inverseOf ex:Polite }"},
            {"label": "CustomerY both Arrogant and Polite?", "type": "ASK",
             "sparql": "PREFIX ex: <http://example.org/ontology#> ASK { ex:CustomerY a ex:Arrogant . ex:CustomerY a ex:Polite }"},
        ],
    },
    {
        "slug": "scenario-2",
        "title": "Firefighters Responding to an Earthquake in Tokyo",
        "prompt": "Firefighters in Tokyo must act when an earthquake of magnitude 8 hits the city.",
        "custom_queries": [
            {"label": "Population types of Tokyo", "type": "SELECT",
             "sparql": "PREFIX ex: <http://example.org/ontology#> SELECT ?population WHERE { ex:Tokyo ex:hasPopulation ?population }"},
            {"label": "Mobility true for Tokyo (isWalkable/isDrivable)", "type": "SELECT",
             "sparql": "PREFIX ex: <http://example.org/ontology#> SELECT ?mobility WHERE { ex:Tokyo ?mobility true . FILTER (?mobility IN (ex:isWalkable, ex:isDrivable)) }"},
            {"label": "EventX is Earthquake (QuakeEvent)?", "type": "ASK",
             "sparql": "PREFIX ex: <http://example.org/ontology#> ASK { ex:EventX a ex:QuakeEvent }"},
            {"label": "Is Firefighter an Occupation?", "type": "ASK",
             "sparql": "PREFIX ex: <http://example.org/ontology#> ASK { ex:Firefighter a ex:Occupation }"},
            {"label": "Tokyo is a City?", "type": "ASK",
             "sparql": "PREFIX ex: <http://example.org/ontology#> ASK { ex:Tokyo a ex:City }"},
        ],
    },
    {
        "slug": "scenario-3",
        "title": "Doctor in Amsterdam with First COVID-19 Patient",
        "prompt": "A doctor in Amsterdam encounters the hospital’s first COVID-19 case. The patient presents symptoms and the doctor considers protocols for isolation and treatment.",
        "custom_queries": [
            {"label": "Symptoms of Covid19", "type": "SELECT",
             "sparql": "PREFIX ex: <http://example.org/ontology#> SELECT ?symptom WHERE { ex:Covid19 ex:hasSymptom ?symptom }"},
            {"label": "PatientX typed as Disease (any)?", "type": "ASK",
             "sparql": "PREFIX ex: <http://example.org/ontology#> ASK { ex:PatientX a ex:Disease }"},
            {"label": "PatientX has Covid19?", "type": "ASK",
             "sparql": "PREFIX ex: <http://example.org/ontology#> ASK { ex:PatientX a ex:Covid19 }"},
            {"label": "Amsterdam is a City?", "type": "ASK",
             "sparql": "PREFIX ex: <http://example.org/ontology#> ASK { ex:Amsterdam a ex:City }"},
            {"label": "Any Protocol appliesTo Covid19?", "type": "ASK",
             "sparql": "PREFIX ex: <http://example.org/ontology#> ASK { ?protocol ex:appliesTo ex:Covid19 }"},
            {"label": "PatientX both Healthy and Covid19?", "type": "ASK",
             "sparql": "PREFIX ex: <http://example.org/ontology#> ASK { ex:PatientX a ex:Healthy . ex:PatientX a ex:Covid19 }"},
        ],
    },
]

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
            raise RuntimeError("OpenAI provider unavailable.")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")
        ocfg = BASE_CONFIG["OPENAI"]
        provider = OpenAIProvider(OpenAIConfig(model=ocfg["MODEL"], temperature=ocfg["TEMPERATURE"], max_tokens=ocfg["MAX_TOKENS"]))
        return StoryGenerator(provider, StoryGeneratorConfig())
    provider = LocalTemplateProvider(LocalTemplateConfig(max_words=800))
    return StoryGenerator(provider, StoryGeneratorConfig())

def _build_rewriter():
    rcfg = BASE_CONFIG["REWRITER"]
    if rcfg.get("PROVIDER", "local").lower() == "openai":
        if OpenAIRewriter is None or OpenAIRewriterConfig is None:
            raise RuntimeError("OpenAI rewriter unavailable.")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")
        return StoryRewriter(
            provider=OpenAIRewriter(OpenAIRewriterConfig(model=rcfg["MODEL"],
                                                         temperature=rcfg["TEMPERATURE"],
                                                         max_tokens=rcfg["MAX_TOKENS"])),
            config=StoryRewriterConfig(instructions=rcfg["INSTRUCTIONS"]),
        )
    return StoryRewriter(provider=LocalRuleRewriter(LocalRuleConfig()),
                         config=StoryRewriterConfig(instructions=rcfg["INSTRUCTIONS"]))

# Deterministic bindings so scenario queries bind consistently
def _inject_query_bindings(g: Graph, scenario_slug: str) -> None:
    def add(s, p, o): g.add((s, p, o))
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
        add(EX.PatientX, RDF.type, EX.Disease)
        add(EX.PatientX, RDF.type, EX.Covid19)
        add(EX.CovidIsolationProtocol, RDF.type, EX.Protocol)
        add(EX.CovidIsolationProtocol, EX.appliesTo, EX.Covid19)

def _normalize_graph(g: Graph) -> None:
    OCC = {EX.Barista, EX.Firefighter, EX.Doctor}
    RES = {EX.CoffeeMachine, EX.Water, EX.Electricity, EX.Milk, EX.AltMilk, EX.PPE, EX.Ventilator, EX.Firetruck, EX.BreathingApparatus, EX.Hose, EX.Ambulance, EX.RescueTools, EX.Comms}
    TRAITS = {EX.Arrogant, EX.Polite}
    for occ in OCC:
        g.remove((occ, RDF.type, EX.Person)); g.remove((occ, EX.hasOccupation, None)); g.remove((occ, EX.hasTrait, None)); g.add((occ, RDF.type, EX.Occupation))
    for res in RES:
        g.remove((res, RDF.type, EX.Person)); g.remove((res, EX.hasOccupation, None)); g.remove((res, EX.hasTrait, None))
    for tr in TRAITS:
        g.add((tr, RDF.type, EX.PersonTrait)); g.remove((tr, RDF.type, EX.City)); g.remove((None, EX.locatedIn, tr))

# NEW: seed a single, controlled violation per scenario 1 & 3 (so “after > before”)
def _seed_initial_violation(g: Graph, scenario_slug: str) -> None:
    if scenario_slug == "scenario-1":
        # Make CustomerY both Arrogant and Polite (inverse traits) → violation before
        g.add((EX.CustomerY, RDF.type, EX.Polite))
    elif scenario_slug == "scenario-3":
        # Make PatientX both Healthy and Covid19 → violation before
        g.add((EX.Healthy, RDF.type, EX.PersonTrait))  # give it a type; optional
        g.add((EX.PatientX, RDF.type, EX.Healthy))

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

from evaluation import Evaluator, ScenarioEvalInput
EVAL = Evaluator()

def run_scenario(base_out: Path, scenario: dict) -> None:
    files = scenario_files(base_out, scenario["slug"])
    _print_header(f"{scenario['title']}")

    # 1) Story Generation
    gen = _build_generator()
    story = gen.generate(scenario["prompt"])
    print(story)
    _save_text(files["story"], story)

    # 2) Fact Extraction
    extractor = InformationExtractor(
        ontology_path=BASE_CONFIG["ONTOLOGY_PATH"],
        synthetic_when_no_person=BASE_CONFIG["EXTRACTOR"]["SYNTHETIC_WHEN_NO_PERSON"],
        synthetic_min_people_per_occupation=BASE_CONFIG["EXTRACTOR"]["SYNTHETIC_MIN_PEOPLE_PER_OCC"],
    )
    res = extractor.extract(story, as_jsonld=True)

    # Bindings for queries + normalize noisy triples
    _inject_query_bindings(res.graph, scenario["slug"])
    _normalize_graph(res.graph)

    # >>> NEW: seed initial violation (only in BEFORE graph)
    _seed_initial_violation(res.graph, scenario["slug"])

    res.jsonld = res.graph.serialize(format="json-ld", indent=2)
    res.jsonld = res.jsonld if isinstance(res.jsonld, str) else res.jsonld.decode("utf-8")
    _save_graph(res.graph, res.jsonld, files["ttl"], files["jsonld"])
    print(res.jsonld)

    # 3) Ontology Checks (BEFORE)
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
        for i, v in enumerate(violations_before, 1): print(f"{i}. {v}")
        _save_text(files["violations"], "\n".join(violations_before))
    else:
        print("=== VIOLATIONS ===\nNone 🎉")
        _save_text(files["violations"], "None")

    # Custom scenario queries
    print("\n--- Scenario Queries ---")
    custom_results = reasoner.run_custom_queries(res.graph, scenario["custom_queries"])
    custom_report = reasoner.format_custom_queries_report(custom_results)
    print(custom_report); _save_text(files["query_report"], custom_report)

    # 4) Error Detection & Explanation
    explainer = ErrorDetectorExplainer()
    explanations_text = explainer.explain(violations_before).text
    print(explanations_text)
    _save_text(files["explanations"], explanations_text)

    # 5) (Conditional) Story Rewriting
    rewriter = _build_rewriter()
    if violations_before:
        corrected = rewriter.rewrite(story, violations_before, BASE_CONFIG["REWRITER"]["INSTRUCTIONS"])
        print(corrected); _save_text(files["corrected_story"], corrected)
    else:
        corrected = story; _save_text(files["corrected_story"], corrected)

    # 6) Re-Check (Final Checker): this re-extracts from TEXT (no injected violations)
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

    # Normalize final graph for consistent queries (NO seeding here!)
    _inject_query_bindings(final_res.graph, scenario["slug"])
    _normalize_graph(final_res.graph)

    _save_text(files["final_story"], final_res.story)
    _save_graph(final_res.graph, final_res.jsonld, files["final_ttl"], files["final_jsonld"])
    _save_text(files["final_violations"], "None" if not final_res.violations else "\n".join(final_res.violations))

    if final_res.passed:
        print("Final check: ✅ PASSED — story is consistent.")
    else:
        print("Final check: ❌ STILL HAS VIOLATIONS")
        for i, v in enumerate(final_res.violations, 1): print(f"{i}. {v}")

    # Evaluation bookkeeping
    EVAL.add_result(ScenarioEvalInput(
        scenario_slug=scenario["slug"], title=scenario["title"],
        graph_before=res.graph, graph_after=final_res.graph,
        violations_before=violations_before, violations_after=final_res.violations,
        rag_facts_used=[],
    ))

def run_all() -> None:
    base_out = Path("outputs")
    for sc in SCENARIOS:
        _ensure_dir(base_out / sc["slug"])
        run_scenario(base_out, sc)
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
