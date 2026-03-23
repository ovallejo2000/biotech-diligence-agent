"""
Module D: Development Pathway & Value Inflection Points
"""

from .base import BaseModule


DEVELOPMENT_PATHWAY_PROMPT = """
TASK: DEVELOPMENT PATHWAY & VALUE INFLECTION POINTS — Module D

VCs invest in value inflection, not products. Map the path from here to a fundable outcome.

Evaluate:

1. CURRENT STAGE: Where is the asset today?

2. NEXT 3 VALUE INFLECTION POINTS:
   For each milestone, identify:
   - What is the event?
   - What data/decision does it produce?
   - Estimated timeframe to reach it
   - Capital required to get there
   - Probability of achieving it (be realistic — most Phase 2s fail)
   - What does positive outcome mean for valuation?
   - What is the key risk at this step?

3. CAPITAL EFFICIENCY:
   - How much capital is needed to reach the next major value inflection?
   - Is this achievable in a single financing round?

4. CRITICAL PATH RISKS:
   - What could delay or derail development?
   - Manufacturing/CMC readiness
   - Patient recruitment feasibility
   - Regulatory agency alignment (is the pathway agreed with FDA/EMA?)

5. PARTNERSHIP LIKELIHOOD:
   - At what stage would a pharma partner be interested?
   - What data package would trigger a deal?

{context_block}

Return ONLY this JSON structure:
{{
  "current_stage": "Discovery | IND-Enabling | Phase 1 | Phase 2 | Phase 3 | NDA/BLA | Approved",
  "current_stage_detail": "1 sentence on where they are today",
  "inflection_points": [
    {{
      "milestone": "Name of milestone",
      "description": "What data/decision this produces",
      "timeframe": "e.g., 12-18 months",
      "capital_to_reach": "e.g., $30-50M",
      "probability_of_success": "e.g., 60-70%",
      "valuation_impact": "What positive outcome does to valuation",
      "key_risk": "Biggest risk at this step"
    }}
  ],
  "total_capital_to_value_event": "Total estimated capital to reach a meaningful partnership or exit trigger",
  "critical_path_risks": ["risk1", "risk2"],
  "partnership_trigger": "What data package would attract pharma interest",
  "development_confidence": "High | Medium | Low"
}}
"""


class DevelopmentPathwayModule(BaseModule):
    MODULE_NAME = "development_pathway"
    MODULE_LABEL = "D. Development Pathway & Inflection Points"

    def run(self, context: dict) -> dict:
        context_block = self._build_context_block(context)
        prompt = DEVELOPMENT_PATHWAY_PROMPT.format(context_block=context_block)
        raw = self._call(prompt, max_tokens=2000)
        result = self._parse_json(raw)
        result["_module"] = self.MODULE_NAME
        return result
