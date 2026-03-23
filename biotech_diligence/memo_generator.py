"""
Memo Generator — assembles module outputs into a formatted investment memo.
"""

from datetime import datetime
from typing import Optional


VERDICT_EMOJI = {"INVEST": "✅", "WATCH": "⚠️", "PASS": "❌"}
# Rapid screen uses PASS=qualified (green), SOFT PASS=maybe (yellow), FAIL=disqualified (red)
SCREEN_EMOJI = {"PASS": "✅", "SOFT PASS": "⚠️", "FAIL": "❌"}
RISK_EMOJI = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
EVIDENCE_EMOJI = {"Compelling": "✅", "Emerging": "⚠️", "Weak": "❌"}


class MemoGenerator:
    """Converts structured module results into a formatted investment memo."""

    def generate(
        self,
        company: str,
        results: dict,
        run_id: Optional[str] = None,
        format: str = "markdown",
    ) -> str:
        if format == "markdown":
            return self._markdown_memo(company, results, run_id)
        elif format == "plain":
            return self._plain_memo(company, results, run_id)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _markdown_memo(self, company: str, results: dict, run_id: Optional[str]) -> str:
        date_str = datetime.utcnow().strftime("%B %d, %Y")
        lines = []

        # Header
        lines += [
            f"# INVESTMENT DILIGENCE MEMO",
            f"## {company.upper()}",
            f"**Date:** {date_str}  |  **Run ID:** {run_id or 'N/A'}  |  **Analyst:** AI Diligence Agent",
            "",
            "---",
            "",
        ]

        # Executive Summary
        lines += self._exec_summary_section(company, results)

        # Rapid Screen
        lines += self._rapid_screen_section(results.get("rapid_screen", {}))

        # Full Diligence
        lines += self._scientific_section(results.get("scientific_validation", {}))
        lines += self._data_section(results.get("data_evidence", {}))
        lines += self._development_section(results.get("development_pathway", {}))
        lines += self._competitive_section(results.get("competitive_landscape", {}))
        lines += self._market_section(results.get("market_commercial", {}))
        lines += self._team_section(results.get("team_execution", {}))
        lines += self._ip_section(results.get("ip_regulatory", {}))
        lines += self._risk_section(results.get("risk_decomposition", {}))
        lines += self._investment_framing_section(results.get("investment_framing", {}))

        # Final Verdict
        lines += self._verdict_section(results.get("decision_engine", {}))

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    def _exec_summary_section(self, company: str, results: dict) -> list[str]:
        decision = results.get("decision_engine", {})
        screen = results.get("rapid_screen", {})
        science = results.get("scientific_validation", {})
        market = results.get("market_commercial", {})
        risk = results.get("risk_decomposition", {})

        verdict = decision.get("verdict") or screen.get("verdict", "N/A")
        ic_line = decision.get("ic_one_liner", "")
        ptcs = risk.get("overall_ptcs", {}).get("range", "N/A")
        hypothesis = science.get("biological_hypothesis", "N/A")
        market_peak = market.get("market_sizing", {}).get("peak_revenue_conservative", "N/A")

        emoji = VERDICT_EMOJI.get(verdict, "")

        return [
            "## EXECUTIVE SUMMARY",
            "",
            f"**Verdict:** {emoji} **{verdict}**",
            "",
            f"> {ic_line}" if ic_line else "",
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
            f"| Company | {company} |",
            f"| Verdict | {verdict} |",
            f"| Overall PTCS | {ptcs} |",
            f"| Peak Revenue (Conservative) | {market_peak} |",
            "",
            f"**Thesis in one sentence:** {hypothesis[:200] if hypothesis else 'N/A'}",
            "",
            "---",
            "",
        ]

    def _rapid_screen_section(self, d: dict) -> list[str]:
        if not d:
            return []
        verdict = d.get("verdict", "N/A")
        emoji = SCREEN_EMOJI.get(verdict, "")
        lines = [
            "## A. RAPID SCREEN",
            "",
            f"**Result:** {emoji} **{verdict}** (Confidence: {d.get('confidence', 'N/A')})",
            "",
        ]
        # Score table
        sci = d.get("science_quality", {})
        poc = d.get("proof_of_concept", {})
        vs = d.get("venture_scale", {})
        lines += [
            "| Dimension | Score | Rationale |",
            "|-----------|-------|-----------|",
            f"| Science Quality | {sci.get('score', 'N/A')} | {sci.get('rationale', '')} |",
            f"| Proof of Concept | {poc.get('score', 'N/A')} | {poc.get('rationale', '')} |",
            f"| Venture Scale | {vs.get('score', 'N/A')} | {vs.get('rationale', '')} |",
            "",
        ]
        flags = d.get("red_flags", [])
        if flags:
            lines.append("**Red Flags:**")
            for f in flags:
                lines.append(f"- 🔴 {f}")
            lines.append("")
        if d.get("proceed_rationale"):
            lines += [f"**Rationale:** {d['proceed_rationale']}", ""]
        if d.get("top_concern"):
            lines += [f"**Top Concern for Deep Diligence:** {d['top_concern']}", ""]
        lines.append("---\n")
        return lines

    def _scientific_section(self, d: dict) -> list[str]:
        if not d:
            return []
        lines = [
            "## B. SCIENTIFIC & MECHANISTIC VALIDATION",
            "",
            f"**Scientific Strength:** {d.get('scientific_strength_score', 'N/A')}",
            "",
            f"**Biological Hypothesis:** {d.get('biological_hypothesis', 'N/A')}",
            "",
        ]
        pv = d.get("pathway_validation", {})
        lines += [
            f"**Pathway Validation Level:** {pv.get('level', 'N/A')}",
            f"> {pv.get('evidence_summary', '')}",
            "",
        ]
        ac = d.get("asset_classification", {})
        lines += [
            f"**Asset Type:** {ac.get('type', 'N/A')} — {ac.get('rationale', '')}",
            "",
            "**Critical Assumptions:**",
        ]
        for a in d.get("critical_assumptions", []):
            conf = a.get("confidence", "")
            conf_emoji = {"High": "✅", "Medium": "⚠️", "Low": "❌", "Unvalidated": "❓"}.get(conf, "")
            lines.append(f"- {conf_emoji} **{a.get('assumption', '')}**")
            lines.append(f"  - Evidence: {a.get('supporting_evidence', 'N/A')}")
        lines.append("")
        sci_risks = d.get("scientific_risks", [])
        if sci_risks:
            lines.append("**Scientific Risks:**")
            for r in sci_risks:
                lines.append(f"- {r}")
            lines.append("")
        if d.get("key_insight"):
            lines += [f"**Key Insight:** _{d['key_insight']}_", ""]
        lines.append("---\n")
        return lines

    def _data_section(self, d: dict) -> list[str]:
        if not d:
            return []
        ev = d.get("evidence_classification", "N/A")
        ev_emoji = EVIDENCE_EMOJI.get(ev, "")
        lines = [
            "## C. DATA & EVIDENCE QUALITY",
            "",
            f"**Evidence Classification:** {ev_emoji} **{ev}**",
            f"**Most Advanced Data:** {d.get('most_advanced_data', 'N/A')}",
            "",
            f"**Data Summary:** {d.get('data_summary', 'N/A')}",
            "",
        ]
        td = d.get("trial_design_assessment", {})
        sq = d.get("signal_quality", {})
        lines += [
            "**Trial Design:**",
            f"- Endpoint Quality: {td.get('endpoint_quality', 'N/A')}",
            f"- Control: {td.get('control_quality', 'N/A')}",
            f"- Powering: {td.get('powering', 'N/A')}",
            f"- Patient Selection: {td.get('patient_selection', 'N/A')}",
            f"- Notes: {td.get('notes', '')}",
            "",
            "**Signal Quality:**",
            f"- Effect Size: {sq.get('effect_size', 'N/A')}",
            f"- Consistency: {sq.get('consistency', 'N/A')}",
            f"- Durability: {sq.get('durability', 'N/A')}",
            f"- Clinical Meaningfulness: {sq.get('clinical_meaningfulness', 'N/A')}",
            "",
        ]
        flags = d.get("data_red_flags", [])
        if flags:
            lines.append("**Data Red Flags:**")
            for f in flags:
                lines.append(f"- 🔴 {f}")
            lines.append("")
        if d.get("key_data_gap"):
            lines += [f"**Key Data Gap:** {d['key_data_gap']}", ""]
        lines.append("---\n")
        return lines

    def _development_section(self, d: dict) -> list[str]:
        if not d:
            return []
        lines = [
            "## D. DEVELOPMENT PATHWAY & INFLECTION POINTS",
            "",
            f"**Current Stage:** {d.get('current_stage', 'N/A')}",
            f"> {d.get('current_stage_detail', '')}",
            "",
            "**Value Inflection Points:**",
            "",
        ]
        for i, ip in enumerate(d.get("inflection_points", []), 1):
            lines += [
                f"### Inflection {i}: {ip.get('milestone', 'N/A')}",
                f"- **Description:** {ip.get('description', '')}",
                f"- **Timeline:** {ip.get('timeframe', 'N/A')}",
                f"- **Capital Required:** {ip.get('capital_to_reach', 'N/A')}",
                f"- **Probability of Success:** {ip.get('probability_of_success', 'N/A')}",
                f"- **Valuation Impact:** {ip.get('valuation_impact', '')}",
                f"- **Key Risk:** 🔴 {ip.get('key_risk', '')}",
                "",
            ]
        lines += [
            f"**Total Capital to Value Event:** {d.get('total_capital_to_value_event', 'N/A')}",
            f"**Partnership Trigger:** {d.get('partnership_trigger', 'N/A')}",
            "",
        ]
        lines.append("---\n")
        return lines

    def _competitive_section(self, d: dict) -> list[str]:
        if not d:
            return []
        lines = [
            "## E. COMPETITIVE LANDSCAPE & POSITIONING",
            "",
            "**Direct Competitors:**",
            "",
            "| Company | Asset | Mechanism | Stage | Key Differentiator |",
            "|---------|-------|-----------|-------|-------------------|",
        ]
        for c in d.get("direct_competitors", []):
            lines.append(
                f"| {c.get('company','')} | {c.get('asset','')} | {c.get('mechanism','')} "
                f"| {c.get('stage','')} | {c.get('key_differentiator','')} |"
            )
        lines.append("")

        diff = d.get("differentiation", {})
        lines += [
            "**Differentiation vs Competition:**",
            "",
            "| Dimension | Score | Rationale |",
            "|-----------|-------|-----------|",
        ]
        for dim in ("efficacy", "safety", "convenience", "modality", "pricing_power"):
            dd = diff.get(dim, {})
            lines.append(f"| {dim.title()} | {dd.get('score', 'N/A')} | {dd.get('rationale', '')} |")
        lines.append("")

        if d.get("competitive_moat"):
            lines += [f"**Competitive Moat:** {d['competitive_moat']}", ""]
        if d.get("competitive_verdict"):
            lines += [f"**Verdict:** {d['competitive_verdict']}", ""]
        if d.get("biggest_competitive_threat"):
            lines += [f"**Biggest Threat:** 🔴 {d['biggest_competitive_threat']}", ""]
        lines.append("---\n")
        return lines

    def _market_section(self, d: dict) -> list[str]:
        if not d:
            return []
        pp = d.get("patient_population", {})
        ms = d.get("market_sizing", {})
        soc = d.get("standard_of_care", {})
        lines = [
            "## F. MARKET & COMMERCIAL REALITY",
            "",
            f"**Target Indication:** {d.get('target_indication', 'N/A')}",
            f"**Addressable Patients:** {pp.get('addressable_patients', 'N/A')} "
            f"({pp.get('treatable_fraction', '')})",
            "",
            f"**Standard of Care:** {soc.get('current_best', 'N/A')}",
            f"**Unmet Need:** {soc.get('unmet_need_level', 'N/A')} — {soc.get('unmet_need_description', '')}",
            "",
            "**Market Sizing:**",
            f"- Peak Penetration: {ms.get('peak_penetration_estimate', 'N/A')}",
            f"- Price/Patient/Year: {ms.get('price_per_patient_per_year', 'N/A')}",
            f"- Peak Revenue (Conservative): **{ms.get('peak_revenue_conservative', 'N/A')}**",
            f"- Peak Revenue (Bull): {ms.get('peak_revenue_bull', 'N/A')}",
            f"- Confidence: {ms.get('sizing_confidence', 'N/A')}",
            "",
        ]
        barriers = d.get("adoption_barriers", [])
        if barriers:
            lines.append("**Adoption Barriers:**")
            for b in barriers:
                lines.append(f"- {b}")
            lines.append("")
        lines += [
            f"**Commercial Risk:** {RISK_EMOJI.get(d.get('commercial_risk_level', ''), '')} {d.get('commercial_risk_level', 'N/A')}",
            f"**Key Insight:** _{d.get('key_commercial_insight', '')}_",
            "",
            "---\n",
        ]
        return lines

    def _team_section(self, d: dict) -> list[str]:
        if not d:
            return []
        ft = d.get("founding_team", {})
        mt = d.get("management_team", {})
        bi = d.get("board_and_investors", {})
        lines = [
            "## G. TEAM & EXECUTION RISK",
            "",
            f"**Overall Team Score:** {d.get('team_overall_score', 'N/A')}",
            f"**Key Man Risk:** {RISK_EMOJI.get(d.get('key_man_risk', ''), '')} {d.get('key_man_risk', 'N/A')}",
            "",
            "**Founding Team:**",
            f"- Scientific Credibility: {ft.get('scientific_credibility', 'N/A')}",
            f"- Clinical/Dev Experience: {ft.get('clinical_dev_experience', 'N/A')}",
            f"- Business Building: {ft.get('business_building_experience', 'N/A')}",
            f"- Notes: {ft.get('notes', '')}",
            "",
            "**Management:**",
            f"- CEO: {mt.get('ceo_quality', 'N/A')}",
            f"- CMO: {mt.get('cmo_quality', 'N/A')}",
            f"- CSO: {mt.get('cso_quality', 'N/A')}",
            f"- Team Depth: {mt.get('overall_team_depth', 'N/A')}",
            "",
            f"**Investors:** {bi.get('investor_quality', 'N/A')} — {', '.join(bi.get('key_investors', []))}",
            f"**Board Credibility:** {bi.get('board_credibility', 'N/A')}",
            "",
            f"**Verdict:** {d.get('team_verdict', 'N/A')}",
            "",
            "---\n",
        ]
        return lines

    def _ip_section(self, d: dict) -> list[str]:
        if not d:
            return []
        pp = d.get("patent_position", {})
        fto = d.get("freedom_to_operate", {})
        rp = d.get("regulatory_pathway", {})
        lines = [
            "## H. IP & REGULATORY RISK",
            "",
            f"**IP/Regulatory Risk Level:** {RISK_EMOJI.get(d.get('ip_regulatory_risk_level', ''), '')} {d.get('ip_regulatory_risk_level', 'N/A')}",
            "",
            "**Patent Position:**",
            f"- Composition of Matter: {pp.get('composition_of_matter', 'N/A')}",
            f"- Method of Use: {pp.get('method_of_use', 'N/A')}",
            f"- Earliest Expiry: {pp.get('earliest_expiry', 'N/A')}",
            f"- Exclusivity Extensions: {', '.join(pp.get('exclusivity_extensions', ['None identified']))}",
            "",
            f"**FTO Status:** {fto.get('fto_status', 'N/A')} (Confidence: {fto.get('fto_confidence', 'N/A')})",
        ]
        fto_risks = fto.get("key_fto_risks", [])
        if fto_risks:
            for r in fto_risks:
                lines.append(f"- 🔴 {r}")
        lines += [
            "",
            f"**Regulatory Pathway:** {rp.get('primary_pathway', 'N/A')}",
            f"- Designations: {', '.join(rp.get('designations', ['None confirmed']))}",
            f"- FDA Alignment: {rp.get('fda_alignment', 'N/A')}",
            f"- Pathway Complexity: {rp.get('pathway_complexity', 'N/A')}",
            "",
        ]
        existential = d.get("existential_ip_regulatory_flags", [])
        if existential:
            lines.append("**⚠️ EXISTENTIAL FLAGS:**")
            for f in existential:
                lines.append(f"- 🔴 {f}")
            lines.append("")
        if d.get("key_ip_insight"):
            lines += [f"**Key Insight:** _{d['key_ip_insight']}_", ""]
        lines.append("---\n")
        return lines

    def _risk_section(self, d: dict) -> list[str]:
        if not d:
            return []
        risks = d.get("risks", {})
        ptcs = d.get("overall_ptcs", {})
        lines = [
            "## I. RISK DECOMPOSITION",
            "",
            "| Risk Category | Score | Key Driver | Mitigant |",
            "|---------------|-------|------------|---------|",
        ]
        for cat in ("scientific", "clinical", "regulatory", "commercial", "financing"):
            r = risks.get(cat, {})
            score = r.get("score", "N/A")
            emoji = RISK_EMOJI.get(score, "")
            lines.append(
                f"| {cat.title()} | {emoji} {score} | {r.get('key_driver', '')} | {r.get('mitigant', '')} |"
            )
        lines += [
            "",
            f"**Overall PTCS:** **{ptcs.get('range', 'N/A')}** ({ptcs.get('qualitative_assessment', '')})",
            f"**Benchmark:** {ptcs.get('benchmark', '')}",
            "",
            f"**Risk That Determines Outcome:** {d.get('risk_that_determines_outcome', 'N/A')}",
            "",
            "---\n",
        ]
        return lines

    def _investment_framing_section(self, d: dict) -> list[str]:
        if not d:
            return []
        bull = d.get("bull_case", {})
        bear = d.get("bear_case", {})
        base = d.get("base_case", {})
        lines = [
            "## J. INVESTMENT FRAMING",
            "",
            f"**Fund Returner Potential:** {d.get('fund_returner_potential', 'N/A')}",
            "",
            f"### Bull Case ({bull.get('return_multiple', 'N/A')} | {bull.get('exit_value', 'N/A')})",
            f"{bull.get('scenario', 'N/A')}",
            "**Conditions:**",
        ]
        for c in bull.get("key_conditions", []):
            lines.append(f"- {c}")
        lines += [
            "",
            f"### Base Case ({base.get('return_multiple', 'N/A')} | {base.get('exit_value', 'N/A')})",
            f"{base.get('scenario', 'N/A')}",
            "",
            f"### Bear Case",
            f"{bear.get('most_likely_failure_mode', 'N/A')}",
            f"Capital at risk before failure: {bear.get('capital_at_risk', 'N/A')}",
            "",
            "**Exit Scenarios:**",
            "",
            "| Type | Acquirer/Path | Value | Probability | Timeline |",
            "|------|--------------|-------|-------------|---------|",
        ]
        for ex in d.get("exit_scenarios", []):
            lines.append(
                f"| {ex.get('type','')} | {ex.get('acquirer_or_path','')} "
                f"| {ex.get('estimated_value','')} | {ex.get('probability','')} | {ex.get('timing','')} |"
            )
        lines += [
            "",
            f"**Key Investment Insight:** _{d.get('key_investment_insight', '')}_",
            "",
            "---\n",
        ]
        return lines

    def _verdict_section(self, d: dict) -> list[str]:
        if not d:
            return []
        verdict = d.get("verdict", "N/A")
        emoji = VERDICT_EMOJI.get(verdict, "")
        lines = [
            "## K. FINAL VERDICT",
            "",
            f"# {emoji} {verdict}",
            f"**Confidence:** {d.get('confidence', 'N/A')}",
            "",
            f"> {d.get('ic_one_liner', '')}",
            "",
            "**Top 3 Reasons:**",
        ]
        for i, r in enumerate(d.get("top_3_reasons", []), 1):
            lines.append(f"{i}. {r}")
        lines += ["", "**Top 3 Risks:**"]
        for i, r in enumerate(d.get("top_3_risks", []), 1):
            lines.append(f"{i}. 🔴 {r}")

        watch = d.get("watch_triggers", [])
        if watch and verdict == "WATCH":
            lines += ["", "**To Move to INVEST, We Need:**"]
            for w in watch:
                lines.append(f"- {w}")

        pass_rat = d.get("pass_rationale", "")
        if pass_rat and verdict == "PASS":
            lines += ["", f"**Pass Rationale:** {pass_rat}"]

        conditions = d.get("suggested_entry_conditions", [])
        if conditions:
            lines += ["", "**Entry Conditions:**"]
            for c in conditions:
                lines.append(f"- {c}")

        comparables = d.get("comparable_deals", [])
        if comparables:
            lines += ["", "**Comparable Deals:**"]
            for c in comparables:
                lines.append(f"- {c}")

        lines += [
            "",
            "---",
            "",
            "*Generated by Biotech Diligence Agent | For internal use only*",
        ]
        return lines

    def _plain_memo(self, company: str, results: dict, run_id: Optional[str]) -> str:
        """Simplified plain text memo."""
        md = self._markdown_memo(company, results, run_id)
        # Strip markdown formatting minimally
        import re
        plain = re.sub(r"#{1,6} ", "", md)
        plain = re.sub(r"\*\*(.+?)\*\*", r"\1", plain)
        plain = re.sub(r"\*(.+?)\*", r"\1", plain)
        plain = re.sub(r"`(.+?)`", r"\1", plain)
        return plain
