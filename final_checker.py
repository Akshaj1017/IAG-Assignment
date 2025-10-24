# final_checker.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List
from rdflib import Graph

from information_extractor import InformationExtractor, ExtractionResult
from ontology_reasoner import OntologyReasoner
from story_rewriter import StoryRewriter

@dataclass
class FinalCheckConfig:
    auto_repair: bool = True
    max_rounds: int = 3
    verbose: bool = False

@dataclass
class FinalCheckResult:
    story: str
    graph: Graph
    jsonld: str
    violations: List[str]
    passed: bool

class FinalChecker:
    """
    Execution Graph mapping: 'Re-Check' (re-extract, re-reason, optional auto-repair loop).
    """
    def __init__(self, extractor: InformationExtractor, reasoner: OntologyReasoner,
                 rewriter: StoryRewriter, config: FinalCheckConfig | None = None):
        self.extractor = extractor
        self.reasoner = reasoner
        self.rewriter = rewriter
        self.config = config or FinalCheckConfig()

    def run(self, story_text: str, instructions: str = "") -> FinalCheckResult:
        current_story = story_text
        last_graph: Graph | None = None
        last_jsonld: str = ""
        violations: List[str] = []

        for round_idx in range(1, self.config.max_rounds + 1):
            res: ExtractionResult = self.extractor.extract(current_story, as_jsonld=True)
            last_graph = res.graph
            last_jsonld = res.jsonld

            violations = self.reasoner.check_all(res.graph)
            if self.config.verbose:
                print(f"[FinalChecker] Round {round_idx}: {len(violations)} violation(s).")

            if not violations:
                return FinalCheckResult(current_story, last_graph, last_jsonld, [], True)

            if self.config.auto_repair and round_idx < self.config.max_rounds:
                repaired = self.rewriter.rewrite(current_story, violations, instructions)
                if repaired.strip() == current_story.strip():
                    break
                current_story = repaired
                continue
            break

        return FinalCheckResult(current_story, last_graph or Graph(), last_jsonld, violations, False)
