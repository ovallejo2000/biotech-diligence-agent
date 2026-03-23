"""
Base module class for all diligence modules.
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Any


MASTER_SYSTEM_PROMPT = """You are a senior biotech venture capital investor at a top-tier life sciences fund (think Forbion, Atlas, Third Rock Ventures, OrbiMed).

Your job is rigorous investment diligence. You think probabilistically, not optimistically.
You prioritize: science quality → team → IP → market. In that order.

Rules you always follow:
- Be decisive. Avoid hedging without cause.
- Highlight what could KILL this deal, not just what looks good.
- Think in inflection points and probability-weighted outcomes.
- Distinguish between "exciting science" and "investable asset."
- Never confuse preclinical data with clinical validation.
- Market size claims are almost always inflated — cut them in half.
- Team is a gating factor. Execution kills more biotech companies than bad science.

You output structured JSON only. No prose outside of JSON values. No markdown code fences."""


class BaseModule(ABC):
    """Abstract base for all diligence modules."""

    MODULE_NAME: str = "base"
    MODULE_LABEL: str = "Base Module"

    def __init__(self, client: Any, model: str):
        self.client = client
        self.model = model

    @abstractmethod
    def run(self, context: dict) -> dict:
        """Execute the module. Returns a dict of results."""

    def _call(self, task_prompt: str, max_tokens: int = 2500) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=MASTER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": task_prompt}],
        )
        return response.content[0].text

    def _parse_json(self, text: str) -> dict:
        """Extract and parse JSON from Claude's response."""
        # Strip markdown code fences if present
        text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in text
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {"raw_output": text, "parse_error": True}

    def _build_context_block(self, context: dict) -> str:
        """Format the context dict into a prompt block."""
        company = context.get("company", "Unknown Company")
        inputs = context.get("inputs", "")
        prior_modules = {
            k: v for k, v in context.items()
            if k not in ("company", "inputs") and isinstance(v, dict)
        }

        lines = [f"COMPANY / ASSET: {company}"]

        if inputs:
            lines.append(f"\nRAW INPUTS PROVIDED:\n{inputs}")
        else:
            lines.append(
                "\nNO RAW INPUTS PROVIDED. Use your training knowledge about this company/asset. "
                "Be explicit when you are uncertain or when information may be outdated."
            )

        if prior_modules:
            lines.append("\nPRIOR MODULE OUTPUTS (use for synthesis):")
            for mod_name, mod_data in prior_modules.items():
                lines.append(f"\n[{mod_name.upper()}]:\n{json.dumps(mod_data, indent=2)}")

        return "\n".join(lines)


def _safe_get(d: dict, key: str, default: str = "") -> str:
    return d.get(key) or default
