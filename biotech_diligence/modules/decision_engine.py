"""
Module K: Decision Engine — Final Verdict
"""

from .base import BaseModule


DECISION_ENGINE_PROMPT = """
TASK: INVESTMENT DECISION — Module K

You have reviewed all prior diligence modules. Now make a decision.
This is the IC moment. Be decisive. No hedging without cause. Every deal has risks —
the question is whether the risk/reward is worth it.

Verdicts:
- INVEST: Risk/reward is favorable, thesis is clear, team can execute. Recommend investment.
- WATCH: Too early or one key question must resolve before committing. Define exactly what would trigger a move to INVEST.
- PASS: Risk/reward is unfavorable, thesis is broken, or better opportunities exist. Be direct about why.

Your verdict must be supported by the weight of diligence evidence, not just the most compelling single module.

{context_block}

Return ONLY this JSON structure:
{{
  "verdict": "INVEST | WATCH | PASS",
  "confidence": "High | Medium | Low",
  "top_3_reasons": [
    "Reason 1 for the verdict (most important)",
    "Reason 2",
    "Reason 3"
  ],
  "top_3_risks": [
    "Risk 1 (most dangerous to the thesis)",
    "Risk 2",
    "Risk 3"
  ],
  "watch_triggers": [
    "If verdict is WATCH: what specific data/event would move to INVEST"
  ],
  "pass_rationale": "If PASS: the thesis-breaking reason in 1-2 sentences",
  "suggested_entry_conditions": [
    "Conditions on valuation, ownership, or milestone gating that would make this investable"
  ],
  "comparable_deals": [
    "Historical deal this resembles and why (for IC calibration)"
  ],
  "ic_one_liner": "The 1-sentence pitch or pass rationale you would say at IC"
}}
"""


class DecisionEngineModule(BaseModule):
    MODULE_NAME = "decision_engine"
    MODULE_LABEL = "K. Decision Engine"

    def run(self, context: dict) -> dict:
        # Decision engine uses ALL prior module outputs
        context_block = self._build_context_block(context)
        prompt = DECISION_ENGINE_PROMPT.format(context_block=context_block)
        raw = self._call(prompt, max_tokens=1500)
        result = self._parse_json(raw)
        result["_module"] = self.MODULE_NAME
        return result
