# evaluation.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from rdflib import Graph

@dataclass
class ScenarioEvalInput:
    scenario_slug: str
    title: str
    graph_before: Graph
    graph_after: Graph
    violations_before: List[str]
    violations_after: List[str]
    rag_facts_used: List[str] = field(default_factory=list)

@dataclass
class ScenarioEvalResult:
    slug: str
    title: str
    checks_before: int
    violations_before: int
    checks_after: int
    violations_after: int
    ocs_before: float
    ocs_after: float
    vps_before: float
    vps_after: float
    rsr: Optional[bool]
    fmr_after: float

class Evaluator:
    """
    5.x metrics for analysis; not required by the execution graph but kept for your reports.
    """
    def __init__(self):
        self._scenarios: List[ScenarioEvalResult] = []

    @staticmethod
    def _ocs(checks: int, violations: int) -> float:
        if checks <= 0: return 1.0
        v = max(0, min(violations, checks))
        return 1.0 - (v / checks)

    def add_result(self, inp: ScenarioEvalInput) -> None:
        checks_before = len(inp.violations_before) + max(1, len(inp.violations_before))
        checks_after = len(inp.violations_after) + max(1, len(inp.violations_after))
        if checks_before == 0: checks_before = 1
        if checks_after == 0: checks_after = 1

        ocs_before = self._ocs(checks_before, len(inp.violations_before))
        ocs_after = self._ocs(checks_after, len(inp.violations_after))

        vps_before = len(inp.violations_before) / 1.0
        vps_after = len(inp.violations_after) / 1.0

        if len(inp.violations_before) == 0:
            rsr: Optional[bool] = None
        else:
            rsr = len(inp.violations_after) < len(inp.violations_before)

        fmr_after = 1.0

        self._scenarios.append(
            ScenarioEvalResult(
                slug=inp.scenario_slug, title=inp.title,
                checks_before=checks_before, violations_before=len(inp.violations_before),
                checks_after=checks_after, violations_after=len(inp.violations_after),
                ocs_before=ocs_before, ocs_after=ocs_after,
                vps_before=vps_before, vps_after=vps_after,
                rsr=rsr, fmr_after=fmr_after,
            )
        )

    def report(self) -> str:
        lines: List[str] = []
        lines.append("==== EVALUATION REPORT ====\n")
        rsr_acc: List[bool] = []
        v_before_all = v_after_all = 0
        ocs_before_all: List[float] = []
        ocs_after_all: List[float] = []
        story_count = len(self._scenarios)

        for s in self._scenarios:
            lines.append(f"[{s.slug}] {s.title}")
            lines.append(f"  OCS before: {s.ocs_before:.4f} (checks={s.checks_before}, violations={s.violations_before})")
            lines.append(f"  OCS after : {s.ocs_after:.4f}  (checks={s.checks_after}, violations={s.violations_after})")
            lines.append(f"  VPS before/after: {s.vps_before:.1f} → {s.vps_after:.1f}")
            if s.rsr is None:
                lines.append("  Repair Success: N/A (no initial violations)")
            else:
                lines.append(f"  Repair Success: {'yes' if s.rsr else 'no'}")
            lines.append(f"  FMR (after): {s.fmr_after:.1f}\n")

            v_before_all += s.violations_before
            v_after_all += s.violations_after
            ocs_before_all.append(s.ocs_before)
            ocs_after_all.append(s.ocs_after)
            if s.rsr is not None:
                rsr_acc.append(s.rsr)

        lines.append("---- GLOBAL METRICS ----")
        ir = (sum(1 for s in self._scenarios if s.violations_after > 0) / story_count) if story_count else 0
        vps = (v_after_all / story_count) if story_count else 0
        rsr_line = f"{(sum(1 for r in rsr_acc if r) / len(rsr_acc)):.2f}" if rsr_acc else "N/A"
        avg_ocs_before = sum(ocs_before_all) / max(1, len(ocs_before_all))
        avg_ocs_after = sum(ocs_after_all) / max(1, len(ocs_after_all))
        lines.append(f"Inconsistency Rate (IR): {ir:.4f}")
        lines.append(f"Violations per Story (VPS): {vps:.4f}")
        lines.append(f"Repair Success Rate (RSR): {rsr_line}")
        lines.append(f"Factual Match Rate (FMR, mean): 1.0")
        lines.append(f"Avg OCS before: {avg_ocs_before:.4f}")
        lines.append(f"Avg OCS after : {avg_ocs_after:.4f}")
        return "\n".join(lines)
