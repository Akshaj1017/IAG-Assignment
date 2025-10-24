# error_explainer.py
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

Category = Literal["logical", "factual", "temporal", "resource"]

RULES: List[Tuple[re.Pattern, Category, str]] = [
    (re.compile(r"\bage bounds violated\b", re.I), "logical", "AGE_BOUNDS"),
    (re.compile(r"\bmarriage\b.*\bnot legal\b", re.I), "logical", "MARRIAGE_LEGALITY"),
    (re.compile(r"\bMarriage legality check: Missing age\b", re.I), "logical", "MARRIAGE_MISSING_AGE"),
    (re.compile(r"\bOccupation cardinality violated\b", re.I), "logical", "OCC_CARDINALITY"),
    (re.compile(r"\bSpouse cardinality violated\b", re.I), "logical", "SPOUSE_CARDINALITY"),
    (re.compile(r"\bSymmetry violated\b", re.I), "logical", "SYMMETRY"),
    (re.compile(r"\bLocation typing\b.*\bnot typed as ex:City\b", re.I), "logical", "LOCATION_TYPING"),
    (re.compile(r"\bQuake magnitude bounds violated\b", re.I), "factual", "MAG_BOUNDS"),
    (re.compile(r"\boccursBefore\b.*\bviolat", re.I), "temporal", "TEMPORAL_ORDER"),
    (re.compile(r"\btemporal\b.*\bviolat", re.I), "temporal", "TEMPORAL_GENERIC"),
    (re.compile(r"\bQuake resources missing\b", re.I), "resource", "QUAKE_KIT_MISSING"),
    (re.compile(r"\brequiresResource\b.*\bmissing\b", re.I), "resource", "GENERIC_RESOURCE"),
]

SUGGESTIONS: Dict[str, str] = {
    "AGE_BOUNDS": "Ensure ex:hasAge is an integer between 0 and 110.",
    "MARRIAGE_LEGALITY": "Set both partners' ages to >= 18, or remove the ex:marriedTo relation.",
    "MARRIAGE_MISSING_AGE": "Add ex:hasAge for both spouses (>= 18) or remove the marriage triple.",
    "OCC_CARDINALITY": "Restrict each ex:Person to a single ex:hasOccupation or remove extras.",
    "SPOUSE_CARDINALITY": "A person may have at most one ex:marriedTo partner. Remove additional spouse links.",
    "SYMMETRY": "For symmetric properties (e.g., ex:marriedTo), assert both directions (A → B and B → A).",
    "LOCATION_TYPING": "Type the object of ex:locatedIn as ex:City or use a property with the correct range.",
    "MAG_BOUNDS": "Set ex:hasMagnitude to a value in [0, 10].",
    "QUAKE_KIT_MISSING": "For ex:QuakeEvent with magnitude > 7, add required resources: Firetruck, Hose, BreathingApparatus, Ambulance, RescueTools, Comms.",
    "GENERIC_RESOURCE": "Add the required ex:requiresResource triples per the protocol or event rules.",
    "TEMPORAL_ORDER": "Repair cycles / inverted ordering in ex:occursBefore; align timestamps.",
    "TEMPORAL_GENERIC": "Fix time-related contradictions.",
}

CATEGORY_BLURBS: Dict[Category, str] = {
    "logical":  "Logical constraint issues (cardinality, legality, typing, symmetry).",
    "factual":  "Domain-bound value issues (e.g., magnitude out of range).",
    "temporal": "Ordering or time consistency problems.",
    "resource": "Missing or insufficient resource requirements.",
}

@dataclass
class Explanation:
    violation: str
    category: Category
    code: str
    suggestion: Optional[str] = None

@dataclass
class ExplanationReport:
    explanations: List[Explanation] = field(default_factory=list)

    def summary_by_category(self) -> Dict[Category, int]:
        counts = {k: 0 for k in CATEGORY_BLURBS.keys()}
        for e in self.explanations: counts[e.category] += 1
        return counts

    @property
    def text(self) -> str:
        if not self.explanations:
            return "No issues detected.\n"
        by_cat: Dict[Category, List[Explanation]] = {k: [] for k in CATEGORY_BLURBS.keys()}
        for e in self.explanations: by_cat[e.category].append(e)
        lines: List[str] = []
        for cat in ["logical", "factual", "temporal", "resource"]:
            if not by_cat[cat]: continue
            lines.append(f"## {cat.title()} — {CATEGORY_BLURBS[cat]}")
            for i, e in enumerate(by_cat[cat], 1):
                lines.append(f"{i}. {e.violation}")
                if e.suggestion:
                    lines.append(f"   ↳ Suggestion: {e.suggestion}")
            lines.append("")
        return "\n".join(lines)

    def as_json(self) -> Dict[str, object]:
        return {"summary": self.summary_by_category(),
                "items": [{"violation": e.violation, "category": e.category, "code": e.code,
                           "suggestion": e.suggestion} for e in self.explanations]}

class ErrorDetectorExplainer:
    """Execution Graph mapping: 'Error Detection' → human explanations."""
    def __init__(self, custom_rules: Optional[List[Tuple[re.Pattern, Category, str]]] = None):
        self.rules = custom_rules or RULES

    def classify(self, violation: str) -> tuple[Category, str]:
        v = violation.strip()
        for pat, cat, code in self.rules:
            if pat.search(v): return cat, code
        return "logical", "UNCLASSIFIED"

    def explain(self, violations: List[str]) -> ExplanationReport:
        items: List[Explanation] = []
        for v in violations:
            cat, code = self.classify(v)
            suggestion = SUGGESTIONS.get(code) or (SUGGESTIONS.get("GENERIC_RESOURCE") if cat == "resource" else None)
            items.append(Explanation(v, cat, code, suggestion))
        return ExplanationReport(explanations=items)
