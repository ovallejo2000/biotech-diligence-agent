"""
Module I: Risk Decomposition & Probability of Success
"""

from .base import BaseModule


RISK_DECOMPOSITION_PROMPT = """
TASK: RISK DECOMPOSITION — Module I

Decompose all investment risk into five categories. Be honest about what you don't know.
Then synthesize into an overall probability of technical and commercial success (PTCS).

IMPORTANT: Use prior module outputs to inform your risk estimates. Don't score risks
in isolation — integrate the scientific, data, IP, team, and commercial picture.

The five risk categories:

1. SCIENTIFIC RISK: Will the drug work as the hypothesis predicts?
   - Target validation level, mechanism confidence, translational risk

2. CLINICAL RISK: Will it work in patients and generate approvable data?
   - Trial design risk, patient selection, biomarker strategy, endpoint risk

3. REGULATORY RISK: Will FDA/EMA approve it?
   - Pathway clarity, precedent, safety profile, REMS likelihood

4. COMMERCIAL RISK: Will it generate meaningful revenue if approved?
   - Market size reality, competitive dynamics, payer access, physician adoption

5. FINANCING RISK: Can the company fund itself to a value-creating event?
   - Capital efficiency, investor quality, dilution risk, market sentiment for sector

For each:
- Score: High / Medium / Low
- Key driver of the risk
- Mitigant (if any)

Then provide:
- OVERALL PTCS (probability of technical and commercial success): % range
- Key risk that determines the outcome

{context_block}

Return ONLY this JSON structure:
{{
  "risks": {{
    "scientific": {{
      "score": "High | Medium | Low",
      "key_driver": "Main source of scientific risk",
      "mitigant": "What reduces this risk",
      "notes": "1-2 sentences"
    }},
    "clinical": {{
      "score": "High | Medium | Low",
      "key_driver": "Main source of clinical risk",
      "mitigant": "What reduces this risk",
      "notes": "1-2 sentences"
    }},
    "regulatory": {{
      "score": "High | Medium | Low",
      "key_driver": "Main source of regulatory risk",
      "mitigant": "What reduces this risk",
      "notes": "1-2 sentences"
    }},
    "commercial": {{
      "score": "High | Medium | Low",
      "key_driver": "Main source of commercial risk",
      "mitigant": "What reduces this risk",
      "notes": "1-2 sentences"
    }},
    "financing": {{
      "score": "High | Medium | Low",
      "key_driver": "Main source of financing risk",
      "mitigant": "What reduces this risk",
      "notes": "1-2 sentences"
    }}
  }},
  "overall_ptcs": {{
    "range": "e.g., 15-25%",
    "benchmark": "Industry average Phase 2 PTCS is ~15%, rare disease is ~25%, Oncology is ~8%",
    "qualitative_assessment": "Above / In-line / Below industry benchmark for this stage and indication"
  }},
  "risk_that_determines_outcome": "The single risk that, if it resolves positively, changes the investment case most"
}}
"""


class RiskDecompositionModule(BaseModule):
    MODULE_NAME = "risk_decomposition"
    MODULE_LABEL = "I. Risk Decomposition"

    def run(self, context: dict) -> dict:
        context_block = self._build_context_block(context)
        prompt = RISK_DECOMPOSITION_PROMPT.format(context_block=context_block)
        raw = self._call(prompt, max_tokens=2000)
        result = self._parse_json(raw)
        result["_module"] = self.MODULE_NAME
        return result
