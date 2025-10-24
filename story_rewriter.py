# story_rewriter.py
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
        return prompt  # provider echoes; deterministic edits handled by facade

@dataclass
class OpenAIRewriterConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.4
    max_tokens: int = 900

class OpenAIRewriter(RewriterProvider):  # optional
    def __init__(self, cfg: OpenAIRewriterConfig):
        self.cfg = cfg
        import os
        from openai import OpenAI  # type: ignore
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def rewrite(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            messages=[
                {"role": "system", "content": "You repair stories to satisfy ontology rules; keep tone and length."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.cfg.max_tokens,
        )
        return resp.choices[0].message["content"].strip()

class StoryRewriter:
    """
    Execution Graph mapping: 'Story Rewriting' (only if violations exist).
    Performs deterministic edits around the provider to guarantee fixes.
    """
    def __init__(self, provider: RewriterProvider, config: StoryRewriterConfig):
        self.provider = provider
        self.config = config
        self.local_rules = LocalRuleConfig()

    def rewrite(self, original_story: str, violations: List[str], instructions: str) -> str:
        text = original_story

        # Deterministic fix: add quake resources sentence if needed
        needs_quake_resources = any("Quake resources missing" in v for v in violations)
        if needs_quake_resources and self.local_rules.add_quake_resources_sentence:
            if self.local_rules.resource_sentence not in text:
                if not text.endswith((".", "!", "?")):
                    text += "."
                text += " " + self.local_rules.resource_sentence

        # Collapse multi-occupation mentions
        occ = "|".join(map(re.escape, self.local_rules.occupations))
        pats = [
            rf"\b(as\s+a\s+)?({occ})\b\s*(?:,?\s*and\s*|\s*&\s*)\b(a\s+)?({occ})\b",
            rf"\b({occ})\b\s*(?:,?\s*and\s*|\s*&\s*)\b({occ})\b",
        ]
        def _collapse(m: re.Match) -> str:
            first = m.group(2) if m.lastindex and m.group(2) else m.group(1)
            if first is None: first = m.group(0)
            if not first.lower().startswith(("a ", "an ", "as ")):
                return f"a {first}"
            return first
        for p in pats:
            text = re.sub(p, _collapse, text, flags=re.I)

        # Avoid 'arrogant' and 'polite' together without contrast
        sents = re.split(r"(?<=[.!?])\s+", text)
        cleaned = []
        for s in sents:
            low = s.lower()
            if "arrogant" in low and "polite" in low and "despite" not in low and "although" not in low:
                s = re.sub(r"\bpolite\b", "", s, flags=re.I)
                s = re.sub(r"\s{2,}", " ", s).strip()
            cleaned.append(s)
        text = " ".join(cleaned)

        # Provider pass (LLM or local echo)
        provider_input = (
            f"INSTRUCTIONS: {self.config.instructions}\n"
            f"CONSTRAINTS: Keep one occupation per person; if an earthquake magnitude >7 is present, "
            f"explicitly mention Firetruck, Hose, BreathingApparatus, Ambulance, RescueTools, and Comms. "
            f"Do not state that the same person is both Arrogant and Polite unless you explicitly explain the contradiction.\n\n"
            f"TEXT:\n{text}"
        )
        text = self.provider.rewrite(provider_input)

        # Final cleanup again
        for p in pats:
            text = re.sub(p, _collapse, text, flags=re.I)
        sents = re.split(r"(?<=[.!?])\s+", text)
        cleaned = []
        for s in sents:
            low = s.lower()
            if "arrogant" in low and "polite" in low and "despite" not in low and "although" not in low:
                s = re.sub(r"\bpolite\b", "", s, flags=re.I)
                s = re.sub(r"\s{2,}", " ", s).strip()
            cleaned.append(s)
        return " ".join(cleaned).strip()
