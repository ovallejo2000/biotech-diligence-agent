"""
Module H: IP & Regulatory Risk (CRITICAL)
"""

from .base import BaseModule


IP_REGULATORY_PROMPT = """
TASK: IP & REGULATORY RISK ASSESSMENT — Module H

IP and regulatory are binary risks. A single existential patent or a clinical hold
can kill a company regardless of science quality. Flag anything fatal.

Evaluate:

1. PATENT LANDSCAPE:
   - Composition of matter (CoM) patents: Strongest form — does the company own them? Expiry?
   - Method-of-use patents: Weaker but valuable — do they exist?
   - Formulation/process patents: Defense layer — are these in place?
   - Any blocking patents from third parties that create FTO issues?

2. FREEDOM TO OPERATE (FTO):
   - Are there known IP conflicts with GSK, Pfizer, Merck, etc. in this space?
   - Has an FTO analysis been conducted? What was the result?
   - Any active patent litigation or interference proceedings?

3. PATENT EXPIRY / EXCLUSIVITY:
   - When do key patents expire?
   - Can they layer on additional IP (new indications, formulations)?
   - Orphan drug / pediatric exclusivity extensions available?

4. REGULATORY PATHWAY:
   - FDA pathway: NDA (505b1, 505b2), BLA, 510k, PMA, Accelerated Approval?
   - Breakthrough Therapy, Fast Track, PRIME (EU) designations obtained?
   - Any precedent in this indication that de-risks the pathway?
   - Anticipated trial design agreed with FDA? (Type B/C meeting outcome?)

5. REGULATORY COMPLEXITY:
   - Single-indication / clean pathway vs multi-indication / complex
   - Risk of clinical hold, safety signals requiring additional studies
   - Post-market commitments likely (REMS, registry)?

6. EXISTENTIAL FLAGS:
   - Anything here that could kill the company outright?

{context_block}

Return ONLY this JSON structure:
{{
  "patent_position": {{
    "composition_of_matter": "Strong | Partial | Weak | Unknown",
    "method_of_use": "Exists | Pending | None | Unknown",
    "key_patents": ["patent or family description"],
    "earliest_expiry": "Year or range",
    "exclusivity_extensions": ["e.g., Orphan Drug through 2031"]
  }},
  "freedom_to_operate": {{
    "fto_status": "Clear | Risks Identified | Unknown | Litigation Ongoing",
    "key_fto_risks": ["risk1"],
    "fto_confidence": "High | Medium | Low"
  }},
  "regulatory_pathway": {{
    "primary_pathway": "e.g., BLA via Accelerated Approval",
    "designations": ["Fast Track", "Breakthrough Therapy"],
    "fda_alignment": "Confirmed via Type B meeting | Presumed | Unknown",
    "pathway_complexity": "Straightforward | Moderate | Complex",
    "precedent_exists": true
  }},
  "regulatory_risks": ["risk1", "risk2"],
  "existential_ip_regulatory_flags": ["CRITICAL: flag1"],
  "ip_regulatory_risk_level": "High | Medium | Low",
  "key_ip_insight": "Most important IP/regulatory observation for the investment decision"
}}
"""


class IPRegulatoryModule(BaseModule):
    MODULE_NAME = "ip_regulatory"
    MODULE_LABEL = "H. IP & Regulatory Risk"

    def run(self, context: dict) -> dict:
        context_block = self._build_context_block(context)
        prompt = IP_REGULATORY_PROMPT.format(context_block=context_block)
        raw = self._call(prompt, max_tokens=2000)
        result = self._parse_json(raw)
        result["_module"] = self.MODULE_NAME
        return result
