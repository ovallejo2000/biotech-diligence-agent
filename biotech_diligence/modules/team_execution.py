"""
Module G: Team & Execution Risk
"""

from .base import BaseModule


TEAM_EXECUTION_PROMPT = """
TASK: TEAM & EXECUTION RISK — Module G

In biotech, team quality is co-equal to science. A great team with okay science beats
a weak team with great science almost every time. Assess execution credibility ruthlessly.

Evaluate:

1. FOUNDING TEAM:
   - Scientific founders: are they thought leaders in the relevant field?
   - Clinical/development founders: do they have successful IND/NDA experience?
   - Business founders: have they built and exited a biotech company before?

2. MANAGEMENT TEAM (CEO, CMO, CSO, CFO):
   - CEO: Biotech-specific experience or pharma-derived? Have they raised Series A/B/C?
   - CMO: Clinical trial execution track record? FDA advisory experience?
   - CSO: Publication record, grant history, industry credibility?

3. BOARD AND ADVISORS:
   - Are marquee investors on the board (signal of institutional backing)?
   - Key opinion leaders (KOLs) as advisors — are they real advisors or just names?

4. EXECUTION TRACK RECORD:
   - Prior drugs advanced through development by this team?
   - Prior company outcomes (IPO, M&A, failure)?

5. CAPITAL RAISING ABILITY:
   - Current investors (signal quality — top-tier VC vs unknown angels)
   - Ability to attract institutional capital in next round
   - Any signal that team has struggled to raise?

6. KEY MAN RISK:
   - Is the company dependent on one or two individuals?
   - What happens if the lead scientist leaves?

{context_block}

Return ONLY this JSON structure:
{{
  "founding_team": {{
    "scientific_credibility": "Strong | Moderate | Weak",
    "clinical_dev_experience": "Strong | Moderate | Weak | None",
    "business_building_experience": "Strong | Moderate | Weak | None",
    "notes": "Key observations on founders"
  }},
  "management_team": {{
    "ceo_quality": "Strong | Adequate | Weak | Unknown",
    "cmo_quality": "Strong | Adequate | Weak | Unknown",
    "cso_quality": "Strong | Adequate | Weak | Unknown",
    "overall_team_depth": "Deep | Adequate | Thin"
  }},
  "board_and_investors": {{
    "investor_quality": "Tier 1 | Mid-tier | Unknown / Early-stage",
    "key_investors": ["investor1", "investor2"],
    "board_credibility": "Strong | Moderate | Weak"
  }},
  "execution_track_record": {{
    "prior_drug_successes": "Yes | Partial | No | Unknown",
    "prior_exits": "Yes | No | Unknown",
    "notable_achievements": ["achievement1"]
  }},
  "key_man_risk": "High | Medium | Low",
  "team_overall_score": "A | B | C | D",
  "team_verdict": "2-3 sentence assessment of whether this team can execute this program"
}}
"""


class TeamExecutionModule(BaseModule):
    MODULE_NAME = "team_execution"
    MODULE_LABEL = "G. Team & Execution Risk"

    def run(self, context: dict) -> dict:
        context_block = self._build_context_block(context)
        prompt = TEAM_EXECUTION_PROMPT.format(context_block=context_block)
        raw = self._call(prompt, max_tokens=2000)
        result = self._parse_json(raw)
        result["_module"] = self.MODULE_NAME
        return result
