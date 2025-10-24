# ontology_reasoner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF

EX = Namespace("http://example.org/ontology#")

REQUIRED_QUAKE_RESOURCES = [
    EX.Firetruck, EX.Hose, EX.BreathingApparatus, EX.Ambulance, EX.RescueTools, EX.Comms,
]

@dataclass
class ReasonerConfig:
    marriage_requires_age: bool = True
    enforce_quake_resources: bool = True

class OntologyReasoner:
    """
    Execution Graph mapping: 'Ontology Checks' and 'Custom Queries'.
    """
    def __init__(self, ontology_path: Optional[str] = "ontology.owl", config: Optional[ReasonerConfig] = None):
        self.ontology_path = ontology_path
        self.config = config or ReasonerConfig()
        self.schema = Graph()
        if ontology_path:
            try:
                self.schema.parse(ontology_path)
            except Exception as e:
                print(f"[OntologyReasoner] Warning: could not parse ontology at {ontology_path}: {e}")

    def check_all(self, data: Graph) -> List[str]:
        v: List[str] = []
        v += self._check_person_age_bounds(data)
        v += self._check_marriage_legality(data)
        v += self._check_has_occupation_cardinality(data)
        v += self._check_marriedto_cardinality(data)
        v += self._check_quake_magnitude_bounds(data)
        v += self._check_symmetry(data)
        v += self._check_event_location_typing(data)
        v += self._check_trait_inverse_conflict(data)   # NEW
        v += self._check_health_vs_disease(data)        # NEW
        if self.config.enforce_quake_resources:
            v += self._check_quake_resource_requirements(data)
        return v

    # --- checks (SPARQL-based) ---
    def _check_person_age_bounds(self, g: Graph) -> List[str]:
        q = """
        PREFIX ex: <http://example.org/ontology#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        SELECT ?p ?age WHERE { ?p a ex:Person ; ex:hasAge ?age . FILTER (?age < 0 || ?age > 110) }
        """
        out = []
        for p, age in g.query(q):
            out.append(f"Age bounds violated: {self._short(p)} hasAge {age} (allowed 0..110).")
        return out

    def _check_marriage_legality(self, g: Graph) -> List[str]:
        q = "PREFIX ex: <http://example.org/ontology#> SELECT ?a ?b WHERE { ?a ex:marriedTo ?b . }"
        out: List[str] = []; seen = set()
        for a, b in g.query(q):
            if a == b: continue
            key = tuple(sorted((str(a), str(b))))
            if key in seen: continue
            seen.add(key)
            age_a = self._get_int(g, a, EX.hasAge)
            age_b = self._get_int(g, b, EX.hasAge)
            if age_a is None or age_b is None:
                if self.config.marriage_requires_age:
                    out.append(f"Marriage legality check: Missing age for {self._short(a)} or {self._short(b)}.")
                continue
            if age_a < 18 or age_b < 18:
                out.append(f"Marriage not legal: {self._short(a)} (age {age_a}) and {self._short(b)} (age {age_b}) must both be >= 18.")
        return out

    def _check_has_occupation_cardinality(self, g: Graph) -> List[str]:
        q = """
        PREFIX ex: <http://example.org/ontology#>
        SELECT ?p (COUNT(?o) AS ?cnt)
        WHERE { ?p a ex:Person ; ex:hasOccupation ?o . }
        GROUP BY ?p HAVING (COUNT(?o) > 1)
        """
        out = []
        for p, cnt in g.query(q):
            out.append(f"Occupation cardinality violated: {self._short(p)} has {int(cnt)} occupations (max 1).")
        return out

    def _check_marriedto_cardinality(self, g: Graph) -> List[str]:
        q = """
        PREFIX ex: <http://example.org/ontology#>
        SELECT ?p (COUNT(DISTINCT ?sp) AS ?cnt)
        WHERE { ?p a ex:Person ; ex:marriedTo ?sp . }
        GROUP BY ?p HAVING (COUNT(DISTINCT ?sp) > 1)
        """
        out = []
        for p, cnt in g.query(q):
            out.append(f"Spouse cardinality violated: {self._short(p)} has {int(cnt)} spouses (max 1).")
        return out

    def _check_quake_magnitude_bounds(self, g: Graph) -> List[str]:
        q = """
        PREFIX ex: <http://example.org/ontology#>
        SELECT ?e ?m WHERE { ?e a ex:QuakeEvent ; ex:hasMagnitude ?m . FILTER (?m < 0 || ?m > 10) }
        """
        out = []
        for e, m in g.query(q):
            out.append(f"Quake magnitude bounds violated: {self._short(e)} hasMagnitude {m} (allowed 0..10).")
        return out

    def _check_quake_resource_requirements(self, g: Graph) -> List[str]:
        q_hi = "PREFIX ex: <http://example.org/ontology#> SELECT ?e ?m WHERE { ?e a ex:QuakeEvent ; ex:hasMagnitude ?m . FILTER (?m > 7) }"
        out: List[str] = []
        for e, m in g.query(q_hi):
            missing: List[str] = []
            for res in REQUIRED_QUAKE_RESOURCES:
                ask_q = f"PREFIX ex: <http://example.org/ontology#> ASK {{ <{e}> ex:requiresResource <{res}> . }}"
                if not bool(g.query(ask_q)):
                    missing.append(self._short(res))
            if missing:
                out.append(f"Quake resources missing: {self._short(e)} (magnitude {m}) lacks {', '.join(missing)}.")
        return out

    def _check_symmetry(self, g: Graph) -> List[str]:
        out: List[str] = []
        for prop, label in [(EX.friendWith, "friendWith"), (EX.marriedTo, "marriedTo")]:
            q = f"PREFIX ex: <http://example.org/ontology#> SELECT ?a ?b WHERE {{ ?a <{prop}> ?b . FILTER NOT EXISTS {{ ?b <{prop}> ?a . }} }}"
            for a, b in g.query(q):
                out.append(f"Symmetry violated for {label}: {self._short(a)} {label} {self._short(b)} but not vice versa.")
        return out

    def _check_event_location_typing(self, g: Graph) -> List[str]:
        q = """
        PREFIX ex: <http://example.org/ontology#>
        SELECT ?s ?o WHERE { ?s ex:locatedIn ?o . FILTER NOT EXISTS { ?o a ex:City . } }
        """
        out: List[str] = []
        for s, o in g.query(q):
            out.append(f"Location typing: {self._short(s)} locatedIn {self._short(o)} but object is not typed as ex:City.")
        return out

    # NEW: inconsistency — same person both Arrogant and Polite
    def _check_trait_inverse_conflict(self, g: Graph) -> List[str]:
        q = """
        PREFIX ex: <http://example.org/ontology#>
        SELECT ?p WHERE { ?p a ex:Arrogant . ?p a ex:Polite . }
        """
        out: List[str] = []
        for (p,) in g.query(q):
            out.append(f"Trait conflict: {self._short(p)} cannot be both ex:Arrogant and ex:Polite.")
        return out

    # NEW: inconsistency — same person both Healthy and Covid19
    def _check_health_vs_disease(self, g: Graph) -> List[str]:
        q = """
        PREFIX ex: <http://example.org/ontology#>
        SELECT ?p WHERE { ?p a ex:Healthy . ?p a ex:Covid19 . }
        """
        out: List[str] = []
        for (p,) in g.query(q):
            out.append(f"Health conflict: {self._short(p)} cannot be both ex:Healthy and ex:Covid19.")
        return out

    # --- custom queries helper ---
    def run_query_select(self, data: Graph, query: str):
        rows = data.query(query)
        vars_ = [str(v) for v in rows.vars]
        out = []
        for r in rows:
            binding = {}
            for i, v in enumerate(vars_):
                binding[v] = str(r[i]) if r[i] is not None else None
            out.append(binding)
        return out

    def run_query_ask(self, data: Graph, query: str) -> bool:
        return bool(data.query(query))

    def run_custom_queries(self, data: Graph, queries: list[dict]) -> list[dict]:
        results = []
        for q in queries:
            qtype = q.get("type", "SELECT").upper()
            label = q.get("label", "(unnamed)")
            sparql = q["sparql"]
            try:
                res = self.run_query_ask(data, sparql) if qtype == "ASK" else self.run_query_select(data, sparql)
                results.append({"label": label, "type": qtype, "result": res})
            except Exception as e:
                results.append({"label": label, "type": qtype, "error": str(e)})
        return results

    def format_custom_queries_report(self, results: list[dict]) -> str:
        lines = ["=== CUSTOM QUERY REPORT ==="]
        for item in results:
            label = item.get("label", "(unnamed)")
            qtype = item.get("type", "?")
            if "error" in item:
                lines.append(f"- {label} [{qtype}]: ERROR -> {item['error']}")
                continue
            if qtype == "ASK":
                lines.append(f"- {label} [ASK]: {'yes' if item['result'] else 'no'}")
            else:
                rows = item["result"]
                if not rows:
                    lines.append(f"- {label} [SELECT]: (no results)")
                else:
                    lines.append(f"- {label} [SELECT]:")
                    for r in rows: lines.append(f"    - {r}")
        return "\n".join(lines) + "\n"

    # utils
    def _get_int(self, g: Graph, subj: URIRef, prop: URIRef) -> Optional[int]:
        for o in g.objects(subj, prop):
            try: return int(o.toPython())
            except Exception: continue
        return None

    def _short(self, uri: URIRef) -> str:
        s = str(uri)
        if s.startswith(str(EX)): return "ex:" + s.split("#", 1)[-1]
        return s
