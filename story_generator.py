# story_generator.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Optional, Dict, Any

class LLMProvider(Protocol):
    def generate(self, prompt: str, **kwargs) -> str: ...

@dataclass
class LocalTemplateConfig:
    max_words: int = 800
    temperature: float = 0.7  # API parity; unused locally

class LocalTemplateProvider:
    def __init__(self, config: Optional[LocalTemplateConfig] = None):
        self.config = config or LocalTemplateConfig()

    def generate(self, prompt: str, **kwargs) -> str:
        max_words = kwargs.get("max_words", self.config.max_words)
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        topic = prompt.strip().rstrip(".")
        paragraphs = [
            f"{topic}. The air was tense, and time felt brittle. On this day ({now}), chance and courage were about to collide.",
            "At first, there was only a shiver—barely a rumor beneath the city’s steady rhythm. Then the world lurched. "
            "Sirens bloomed, radios crackled, and ordinary people became a chain of steady hands.",
            "Among them stood a small band whose job was to run toward the heat—toward the noise—toward whatever others fled. "
            "They traded quick glances: a pact made of grit and trust. Their training thrummed like a second heartbeat.",
            "They moved by checklists and instincts. They mapped danger by sound, by dust, by the tilt of light through drifting haze. "
            "Names were taken, doors were pried, and promises were kept even when they shuddered on the edge of breaking.",
            "When the last call quieted, the city exhaled. Night took an inventory of courage and left a receipt in the form of stories. "
            "And though tomorrow would ask for more, tonight belonged to those who answered first—and to those who were found.",
        ]
        words = (" ".join(paragraphs)).split()
        if len(words) > max_words:
            words = words[:max_words] + ["…"]
        return " ".join(words)

@dataclass
class OpenAIConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.8
    max_tokens: int = 900
    system_prompt: str = (
        "You are a vivid, efficient story generator. "
        "Write tightly, with strong imagery and momentum. "
        "Avoid purple prose and keep the narrative focused."
    )

class OpenAIProvider:  # optional
    def __init__(self, config: Optional[OpenAIConfig] = None):
        self.config = config or OpenAIConfig()
        try:
            from openai import OpenAI  # type: ignore
            self._ClientClass = OpenAI
        except Exception as e:
            raise RuntimeError("OpenAIProvider requires `openai`. pip install openai") from e

    def generate(self, prompt: str, **kwargs) -> str:
        cfg = self.config
        model = kwargs.get("model", cfg.model)
        temperature = kwargs.get("temperature", cfg.temperature)
        max_tokens = kwargs.get("max_tokens", cfg.max_tokens)
        system_prompt = kwargs.get("system_prompt", cfg.system_prompt)

        client = self._ClientClass()
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()

@dataclass
class StoryGeneratorConfig:
    extras: Optional[Dict[str, Any]] = None

class StoryGenerator:
    def __init__(self, provider: LLMProvider, config: Optional[StoryGeneratorConfig] = None):
        self.provider = provider
        self.config = config or StoryGeneratorConfig()

    def generate(self, prompt: str) -> str:
        kwargs = self.config.extras or {}
        return self.provider.generate(prompt, **kwargs)
