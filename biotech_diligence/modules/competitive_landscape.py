"""
Module E: Competitive Landscape & Positioning
"""

from .base import BaseModule


COMPETITIVE_LANDSCAPE_PROMPT = """
TASK: COMPETITIVE LANDSCAPE & POSITIONING — Module E

Think at the mechanism level, not just the indication level. True competition comes from
alternatives that address the same unmet need, even through different biology.

Evaluate:

1. DIRECT COMPETITORS:
   - Assets in the same mechanism class targeting the same indication
   - Stage of development, key data readouts, company backing

2. MECHANISM-LEVEL COMPETITION:
   - Other mechanisms addressing the same biological problem
   - Any approved therapies that could set the bar too high?

3. DIFFERENTIATION ANALYSIS — For each dimension, score and explain:
   - Efficacy: Meaningfully better, equivalent, or worse than competitors?
   - Safety: Cleaner profile or similar/worse?
   - Dosing/Convenience: Patient-friendly or burdensome?
   - Modality advantage: Is there a fundamental advantage in the modality (e.g., oral vs injectable, long-acting, CNS penetrance)?
   - Cost of goods / pricing potential: Premium price justified?

4. COMPETITIVE MOAT:
   - What structural advantage does this company have?
   - Is differentiation durable (IP-protected, process-dependent, or easily copied)?

5. VERDICT:
   - Can this asset win in a competitive market?
   - Or will it be a "also-ran" that requires head-to-head superiority to survive?

{context_block}

Return ONLY this JSON structure:
{{
  "direct_competitors": [
    {{
      "company": "Company name",
      "asset": "Asset name",
      "mechanism": "Mechanism of action",
      "stage": "Development stage",
      "key_differentiator": "What makes them competitive or not"
    }}
  ],
  "mechanism_level_competition": [
    {{
      "mechanism": "Alternative mechanism",
      "representative_asset": "Key asset",
      "threat_level": "High | Medium | Low",
      "rationale": "Why this is or isn't a threat"
    }}
  ],
  "differentiation": {{
    "efficacy": {{"score": "Ahead | Equivalent | Behind | Unknown", "rationale": "1 sentence"}},
    "safety": {{"score": "Ahead | Equivalent | Behind | Unknown", "rationale": "1 sentence"}},
    "convenience": {{"score": "Ahead | Equivalent | Behind | Unknown", "rationale": "1 sentence"}},
    "modality": {{"score": "Ahead | Equivalent | Behind | Unknown", "rationale": "1 sentence"}},
    "pricing_power": {{"score": "High | Moderate | Low | Unknown", "rationale": "1 sentence"}}
  }},
  "competitive_moat": "Description of structural competitive advantage or lack thereof",
  "competitive_verdict": "Why this wins or loses vs alternatives",
  "biggest_competitive_threat": "The single most dangerous competitor or competitive dynamic"
}}
"""


class CompetitiveLandscapeModule(BaseModule):
    MODULE_NAME = "competitive_landscape"
    MODULE_LABEL = "E. Competitive Landscape & Positioning"

    def run(self, context: dict) -> dict:
        context_block = self._build_context_block(context)
        prompt = COMPETITIVE_LANDSCAPE_PROMPT.format(context_block=context_block)
        raw = self._call(prompt, max_tokens=2500)
        result = self._parse_json(raw)
        result["_module"] = self.MODULE_NAME
        return result
