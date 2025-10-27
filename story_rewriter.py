from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Protocol

class RewriterProvider(Protocol):
    def rewrite(self, prompt: str) -> str: ...

@dataclass
class StoryRewriterConfig:
    instructions: str = "Fix violations with minimal changes."

@dataclass
class LocalRuleConfig:
    add_quake_resources_sentence: bool = True
    occupations: tuple = ("barista", "firefighter", "doctor")
    traits: tuple = ("arrogant", "polite")
    resource_sentence: str = (
        "In response, crews deployed Firetrucks and Ambulances, ran Hoses with Breathing Apparatus, "
        "coordinated Rescue Tools, and kept Comms online to reach every trapped voice."
    )

class LocalRuleRewriter(RewriterProvider):
    def __init__(self, cfg: LocalRuleConfig | None = None):
        self.cfg = cfg or LocalRuleConfig()

    def rewrite(self, prompt: str) -> str:
        return prompt

class StoryRewriter:
    def __init__(self, provider: RewriterProvider, config: StoryRewriterConfig):
        self.provider = provider
        self.config = config
        self.local_rules = LocalRuleConfig()

    def rewrite(self, original_story: str, violations: List[str], instructions: str) -> str:
        text = original_story

        # === 1. Add quake resources if needed
        if any("Quake resources missing" in v for v in violations):
            if self.local_rules.resource_sentence not in text:
                if not text.endswith((".", "!", "?")):
                    text += "."
                text += " " + self.local_rules.resource_sentence

        # === 2. Collapse multiple occupations
        occ = "|".join(map(re.escape, self.local_rules.occupations))
        occupation_patterns = [
            rf"\b(as\s+a\s+)?({occ})\b\s*(?:,?\s*and\s*|\s*&\s*)\b(a\s+)?({occ})\b",
            rf"\b({occ})\b\s*(?:,?\s*and\s*|\s*&\s*)\b({occ})\b",
        ]
        def _collapse(match: re.Match) -> str:
            first = match.group(2) if match.lastindex and match.group(2) else match.group(1)
            return f"a {first}" if first and not first.lower().startswith(("a ", "an ", "as ")) else first
        for pattern in occupation_patterns:
            text = re.sub(pattern, _collapse, text, flags=re.I)

        # === 3. Remove Polite if both Arrogant and Polite are mentioned (without contrast)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        cleaned = []
        for s in sentences:
            lower = s.lower()
            if "arrogant" in lower and "polite" in lower and not any(k in lower for k in ["despite", "although"]):
                s = re.sub(r"\bpolite\b", "", s, flags=re.I)
                s = re.sub(r"\s{2,}", " ", s).strip()
            cleaned.append(s)
        text = " ".join(cleaned)

        # === 4. Handle health conflict: remove "healthy"
        if any("Health conflict" in v for v in violations):
            text = re.sub(r"\bhealthy\b", "", text, flags=re.I)
            text = re.sub(r"\s{2,}", " ", text).strip()

        # === 5. Remove 'Polite' if it's still unhandled
        if any("Trait conflict" in v for v in violations):
            text = re.sub(r"\bpolite\b", "", text, flags=re.I)
            text = re.sub(r"\s{2,}", " ", text).strip()

        # === 6. Provider LLM or echo rule-based fallback
        provider_input = (
            f"INSTRUCTIONS: {self.config.instructions}\n"
            f"CONSTRAINTS: Keep one occupation per person; if an earthquake magnitude >7 is present, "
            f"explicitly mention Firetruck, Hose, BreathingApparatus, Ambulance, RescueTools, and Comms. "
            f"Do not state that the same person is both Arrogant and Polite unless you explicitly explain the contradiction.\n\n"
            f"TEXT:\n{text}"
        )
        rewritten = self.provider.rewrite(provider_input)

        return rewritten.strip()
