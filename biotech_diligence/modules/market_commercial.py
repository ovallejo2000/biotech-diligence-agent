"""
Module F: Market & Commercial Reality
"""

from .base import BaseModule


MARKET_COMMERCIAL_PROMPT = """
TASK: MARKET & COMMERCIAL REALITY — Module F

Be skeptical. Most biotech market size estimates are inflated by 3-10x.
Your job is to produce a realistic, bottom-up commercial picture.

Evaluate:

1. TARGET INDICATION:
   - Precise patient population (not broad disease category)
   - Epidemiology: prevalence, incidence, diagnosed fraction, treatable fraction
   - Disease severity and urgency (premium pricing justified?)

2. STANDARD OF CARE:
   - Current best treatment
   - Unmet need: Is there a genuine gap, or is this an improvement on adequate therapy?
   - Physician behavior: Will they change practice based on this data?

3. MARKET SIZING (bottom-up, realistic):
   - Total addressable patients (treated)
   - Expected penetration at peak (be conservative: rare disease might hit 30-50%, crowded space 5-15%)
   - Realistic price per patient per year (anchored to comparable approvals)
   - Peak annual revenue estimate (conservative case, not bull case)

4. ADOPTION BARRIERS:
   - Reimbursement risk (payer pushback, step therapy requirements)
   - Physician education curve
   - Competitive entrenchment (generic, biosimilar, or established brand)
   - Patient access/infrastructure (e.g., cell therapy logistics)

5. COMMERCIAL MODEL:
   - Company build (own salesforce) or partnership?
   - If partnership: what royalty/milestone structure is realistic?

{context_block}

Return ONLY this JSON structure:
{{
  "target_indication": "Precise indication name",
  "patient_population": {{
    "prevalence": "e.g., ~50,000 in US",
    "treatable_fraction": "e.g., 60% are diagnosed and eligible",
    "addressable_patients": "e.g., ~30,000 US / ~80,000 global"
  }},
  "standard_of_care": {{
    "current_best": "Current standard treatment",
    "unmet_need_level": "High | Moderate | Low",
    "unmet_need_description": "1-2 sentences on the gap"
  }},
  "market_sizing": {{
    "peak_penetration_estimate": "e.g., 15-20%",
    "price_per_patient_per_year": "e.g., $80,000-120,000",
    "peak_revenue_conservative": "e.g., $400-600M",
    "peak_revenue_bull": "e.g., $800M-1.2B",
    "sizing_confidence": "High | Medium | Low"
  }},
  "adoption_barriers": ["barrier1", "barrier2"],
  "commercial_model": "Own salesforce | Partnership | Hybrid",
  "commercial_risk_level": "High | Medium | Low",
  "key_commercial_insight": "The single most important commercial consideration"
}}
"""


class MarketCommercialModule(BaseModule):
    MODULE_NAME = "market_commercial"
    MODULE_LABEL = "F. Market & Commercial Reality"

    def run(self, context: dict) -> dict:
        context_block = self._build_context_block(context)
        prompt = MARKET_COMMERCIAL_PROMPT.format(context_block=context_block)
        raw = self._call(prompt, max_tokens=2000)
        result = self._parse_json(raw)
        result["_module"] = self.MODULE_NAME
        return result
