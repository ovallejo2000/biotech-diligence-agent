"""
Module J: Investment Framing (VC Thinking)
"""

from .base import BaseModule


INVESTMENT_FRAMING_PROMPT = """
TASK: INVESTMENT FRAMING — Module J

Think like an investor presenting to an IC (Investment Committee). Make the case for
and against. Then define what success looks like and who buys it.

Evaluate:

1. BULL CASE:
   - What needs to go right for this to be a great investment?
   - What does the best realistic outcome look like?
   - What is the return multiple in the bull case (2x, 5x, 10x+)?

2. BEAR CASE:
   - What is the most likely failure mode?
   - How much capital is at risk before the company fails or pivots?
   - What does the write-off scenario look like?

3. BASE CASE:
   - Most likely outcome given current evidence
   - Expected value of the investment

4. EXIT SCENARIOS:
   - M&A: Who acquires and at what stage / price?
     (Be specific: which pharma or large biotech, and why they would pay)
   - IPO: Is this an IPO-able story? When and at what market cap?
   - Partnership: Could a co-development deal with milestone payments be the value event?

5. STRATEGIC VALUE:
   - Does this asset have strategic value beyond its own data?
   - Platform potential (multiple programs)?
   - Competitive threat to acquirers (defensive M&A)?

6. RETURN ANALYSIS:
   - Entry valuation (if known or estimable)
   - Target return multiple and exit valuation
   - Is this a fund returner potential?

{context_block}

Return ONLY this JSON structure:
{{
  "bull_case": {{
    "scenario": "What the best realistic outcome looks like",
    "key_conditions": ["Condition 1 that must be true", "Condition 2"],
    "return_multiple": "e.g., 10-20x",
    "exit_value": "e.g., $2-4B acquisition"
  }},
  "bear_case": {{
    "most_likely_failure_mode": "The most probable way this fails",
    "capital_at_risk": "How much is invested before failure",
    "residual_value": "What remains (IP, platform, team)"
  }},
  "base_case": {{
    "scenario": "Most probable outcome",
    "return_multiple": "e.g., 3-5x",
    "exit_value": "e.g., $500-800M"
  }},
  "exit_scenarios": [
    {{
      "type": "M&A | IPO | Partnership | Write-off",
      "acquirer_or_path": "Specific company name or path description",
      "rationale": "Why this exit makes sense",
      "estimated_value": "e.g., $1-1.5B",
      "probability": "e.g., 30%",
      "timing": "e.g., 3-5 years"
    }}
  ],
  "strategic_value": {{
    "platform_potential": true,
    "defensive_ma_rationale": "Why a pharma would buy defensively",
    "unique_strategic_assets": ["asset1"]
  }},
  "fund_returner_potential": "Yes | Possible | Unlikely",
  "key_investment_insight": "The single most important insight for an IC presentation"
}}
"""


class InvestmentFramingModule(BaseModule):
    MODULE_NAME = "investment_framing"
    MODULE_LABEL = "J. Investment Framing"

    def run(self, context: dict) -> dict:
        context_block = self._build_context_block(context)
        prompt = INVESTMENT_FRAMING_PROMPT.format(context_block=context_block)
        raw = self._call(prompt, max_tokens=2500)
        result = self._parse_json(raw)
        result["_module"] = self.MODULE_NAME
        return result
