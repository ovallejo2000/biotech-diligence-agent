"""
Module C: Data & Evidence Quality
"""

from .base import BaseModule


DATA_EVIDENCE_PROMPT = """
TASK: DATA & EVIDENCE QUALITY ASSESSMENT — Module C

Assess the quality and strength of the data package. This is about rigor, not excitement.

Evaluate:

1. DATA PACKAGE OVERVIEW:
   - What data exists (preclinical, Phase 1, Phase 2, Phase 3)?
   - What is the most advanced piece of evidence?

2. TRIAL DESIGN QUALITY (if clinical data exists):
   - Endpoints: Primary endpoint appropriate? Validated surrogate or hard clinical outcome?
   - Controls: Placebo-controlled? Active comparator? Or single-arm (weaker)?
   - Powering: Was the trial adequately powered? Any risk of false positive due to small N?
   - Patient selection: Enriched/selected population (may not generalize) or broad?

3. SIGNAL QUALITY:
   - Effect size: Clinically meaningful or statistically significant but marginal?
   - Consistency: Reproducible across subgroups, centers, timepoints?
   - Durability: Is the benefit sustained or does it wane?

4. RED FLAGS IN DATA:
   - Cherry-picked endpoints or subgroups
   - Unusually clean data (too good to be true)
   - High dropout/discontinuation
   - Biomarker-only endpoints with no functional/clinical validation
   - Comparator arm that was stacked in their favor

5. OVERALL EVIDENCE CLASSIFICATION:
   - Compelling: Well-controlled, powered, clinically meaningful, reproducible
   - Emerging: Promising signals but early, small N, or surrogate endpoints only
   - Weak: Preclinical only, uncontrolled, or contradictory data

{context_block}

Return ONLY this JSON structure:
{{
  "most_advanced_data": "Preclinical | Phase 1 | Phase 2 | Phase 3 | Approved",
  "data_summary": "2-3 sentence summary of what data exists",
  "trial_design_assessment": {{
    "endpoint_quality": "Gold Standard | Acceptable | Weak | N/A",
    "control_quality": "RCT | Single-Arm | Historical Control | N/A",
    "powering": "Adequately Powered | Underpowered | Unknown | N/A",
    "patient_selection": "Broad | Enriched | N/A",
    "notes": "Key trial design observations"
  }},
  "signal_quality": {{
    "effect_size": "Large | Moderate | Small | Unknown",
    "consistency": "Consistent | Mixed | Unknown",
    "durability": "Durable | Short-term | Unknown",
    "clinical_meaningfulness": "Clearly Meaningful | Debatable | Unclear"
  }},
  "data_red_flags": ["flag1", "flag2"],
  "evidence_classification": "Compelling | Emerging | Weak",
  "key_data_gap": "The most important missing piece of evidence that would change the thesis"
}}
"""


class DataEvidenceModule(BaseModule):
    MODULE_NAME = "data_evidence"
    MODULE_LABEL = "C. Data & Evidence Quality"

    def run(self, context: dict) -> dict:
        context_block = self._build_context_block(context)
        prompt = DATA_EVIDENCE_PROMPT.format(context_block=context_block)
        raw = self._call(prompt, max_tokens=2000)
        result = self._parse_json(raw)
        result["_module"] = self.MODULE_NAME
        return result
