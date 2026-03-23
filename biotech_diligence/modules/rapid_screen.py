"""
Module A: Rapid Screening Layer
Simulates the 20-minute VC filter — pass/fail before deep diligence.
"""

from .base import BaseModule


RAPID_SCREEN_PROMPT = """
TASK: RAPID SCREENING — Module A

You are doing the initial 20-minute VC filter. Most opportunities die here.

Evaluate the following company/asset across these four vectors:

1. SCIENCE QUALITY: Is the underlying science credible and differentiated?
   - Flag: "me-too" mechanisms, well-trodden biology with no twist, purely academic science with no translational plan.

2. PROOF OF CONCEPT: Is there credible POC?
   - Minimum bar: Human biomarker data, validated target in disease-relevant model, or clinical signal.
   - Flag: Purely in vitro, single animal model, unvalidated target.

3. VENTURE SCALE: Is this a venture-scale opportunity?
   - Would success produce a $500M–$3B+ exit?
   - Flag: Niche indications, chronic condition plays with no fast path to premium pricing, no scalable IP.

4. RED FLAGS: Anything existential?
   - Crowded IP landscape, regulatory nightmare, platform with no lead asset, founder with no drug dev experience, raise size incompatible with stage.

{context_block}

Return ONLY this JSON structure:
{{
  "verdict": "PASS | SOFT PASS | FAIL",
  "confidence": "Low | Medium | High",
  "science_quality": {{
    "score": "Strong | Moderate | Weak",
    "rationale": "1-2 sentences"
  }},
  "proof_of_concept": {{
    "score": "Strong | Moderate | Weak | None",
    "rationale": "1-2 sentences"
  }},
  "venture_scale": {{
    "score": "Yes | Potentially | No",
    "rationale": "1-2 sentences"
  }},
  "red_flags": ["flag1", "flag2"],
  "proceed_rationale": "1-2 sentence verdict explanation. Be direct.",
  "top_concern": "Single most important concern to resolve in deeper diligence"
}}
"""


class RapidScreenModule(BaseModule):
    MODULE_NAME = "rapid_screen"
    MODULE_LABEL = "A. Rapid Screening"

    def run(self, context: dict) -> dict:
        context_block = self._build_context_block(context)
        prompt = RAPID_SCREEN_PROMPT.format(context_block=context_block)
        raw = self._call(prompt, max_tokens=1200)
        result = self._parse_json(raw)
        result["_module"] = self.MODULE_NAME
        return result
