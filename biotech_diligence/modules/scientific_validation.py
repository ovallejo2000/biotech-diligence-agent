"""
Module B: Scientific & Mechanistic Validation
"""

from .base import BaseModule


SCIENTIFIC_PROMPT = """
TASK: SCIENTIFIC & MECHANISTIC VALIDATION — Module B

Go deep on the science. Your job is to identify whether the biological hypothesis is sound
and what must be true for the drug to work.

Evaluate:

1. BIOLOGICAL HYPOTHESIS: What is the mechanistic bet? Is it clearly stated?
   - Is the target biologically validated (human genetics, KO models, clinical data)?
   - Or is this a plausible hypothesis without human-level validation?

2. PATHWAY VALIDATION LEVEL:
   - Human validated (gold standard: genetic association, biomarker data from patients, clinical signal)
   - Preclinical only (animal models, in vitro)
   - Unvalidated hypothesis

3. ASSET CLASSIFICATION:
   - First-in-class (FIC): New mechanism, novel biology — higher risk/reward
   - Fast-follow (FF): Proven mechanism, execution bet — lower risk but crowded
   - Me-too: No differentiation thesis — rarely investable at early stage

4. KEY ASSUMPTIONS — what MUST be true for this to work:
   List the 3-5 critical assumptions. For each, state whether there is evidence supporting it.

5. SCIENTIFIC RISKS:
   - What could invalidate the hypothesis?
   - Any known target toxicity, poor selectivity, or on-target/off-target issues?

{context_block}

Return ONLY this JSON structure:
{{
  "biological_hypothesis": "Clear 2-3 sentence description of the mechanistic bet",
  "pathway_validation": {{
    "level": "Human Validated | Preclinical Only | Unvalidated Hypothesis",
    "evidence_summary": "Key supporting evidence in 2-3 sentences",
    "human_validation_exists": true
  }},
  "asset_classification": {{
    "type": "First-in-Class | Fast-Follow | Me-Too",
    "rationale": "1 sentence"
  }},
  "critical_assumptions": [
    {{
      "assumption": "What must be true",
      "supporting_evidence": "Evidence that supports or refutes this",
      "confidence": "High | Medium | Low | Unvalidated"
    }}
  ],
  "scientific_risks": ["risk1", "risk2"],
  "scientific_strength_score": "Compelling | Moderate | Weak",
  "key_insight": "The single most important scientific insight or concern for investors"
}}
"""


class ScientificValidationModule(BaseModule):
    MODULE_NAME = "scientific_validation"
    MODULE_LABEL = "B. Scientific & Mechanistic Validation"

    def run(self, context: dict) -> dict:
        context_block = self._build_context_block(context)
        prompt = SCIENTIFIC_PROMPT.format(context_block=context_block)
        raw = self._call(prompt, max_tokens=2000)
        result = self._parse_json(raw)
        result["_module"] = self.MODULE_NAME
        return result
