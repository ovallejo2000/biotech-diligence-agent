#!/usr/bin/env python3
"""
Demo: Generates a realistic Karuna Therapeutics diligence memo
using mock data — no API key required. Shows the full output format.

Karuna Therapeutics (KarXT) was acquired by Bristol Myers Squibb for ~$14B in 2023.
This demo shows what the agent would have produced at Phase 3 stage, pre-acquisition.
"""

from biotech_diligence.memo_generator import MemoGenerator
from pathlib import Path

MOCK_RESULTS = {
    "rapid_screen": {
        "_module": "rapid_screen",
        "verdict": "PASS",
        "confidence": "High",
        "science_quality": {
            "score": "Strong",
            "rationale": "xanomeline-trospium targets M1/M4 muscarinic receptors with a novel peripheral-restricting mechanism. First muscarinic agonist in psychiatry with credible human validation from the Eli Lilly Phase 2 dataset (2000s).",
        },
        "proof_of_concept": {
            "score": "Strong",
            "rationale": "Positive Phase 3 EMERGENT-2 trial in schizophrenia (p<0.001 on PANSS total). Clinical signal is unambiguous.",
        },
        "venture_scale": {
            "score": "Yes",
            "rationale": "Schizophrenia affects ~3.5M patients in the US. Antipsychotic market is $7B+. A differentiated, better-tolerated agent could be a blockbuster.",
        },
        "red_flags": [
            "EPS/metabolic liability of current SoC creates high bar for tolerability differentiation — trospium adds complexity to the formulation.",
            "BMS/Pfizer muscarinic programs could accelerate in wake of KarXT success.",
        ],
        "proceed_rationale": "Clear Phase 3 proof of concept with a mechanism that addresses the #1 unmet need in schizophrenia — tolerability. This is a fundable, venture-scale opportunity with a credible path to approval.",
        "top_concern": "Does the co-administration of trospium to limit peripheral side effects create CMC and IP complexity that's defensible long-term?",
    },
    "scientific_validation": {
        "_module": "scientific_validation",
        "biological_hypothesis": "Selective activation of M1 and M4 muscarinic acetylcholine receptors (mAChRs) in the CNS reduces positive, negative, and cognitive symptoms of schizophrenia via modulation of dopamine and glutamate neurotransmission — without engaging peripheral muscarinic receptors (M2/M3) that cause cholinergic side effects.",
        "pathway_validation": {
            "level": "Human Validated",
            "evidence_summary": "Eli Lilly's Phase 2 data with xanomeline alone (vs placebo) in the early 2000s showed significant efficacy across PANSS domains. The human target validation predates Karuna. The trospium add serves as a pharmacokinetic blocker to restrict peripheral exposure.",
            "human_validation_exists": True,
        },
        "asset_classification": {
            "type": "First-in-Class",
            "rationale": "First M1/M4-preferring muscarinic agonist to reach Phase 3 with demonstrated tolerability through the peripheral restriction approach.",
        },
        "critical_assumptions": [
            {
                "assumption": "Trospium's peripheral restriction is sufficient to prevent dose-limiting cholinergic side effects while maintaining CNS xanomeline exposure",
                "supporting_evidence": "Phase 2 and Phase 3 data show GI AEs manageable vs xanomeline alone; discontinuation rate acceptable",
                "confidence": "High",
            },
            {
                "assumption": "M1/M4 agonism drives the efficacy signal (not off-target effects)",
                "supporting_evidence": "Mechanistically consistent with dopamine modulation; effect persists in controlled Phase 3 — confidence is high",
                "confidence": "High",
            },
            {
                "assumption": "Efficacy translates across broader schizophrenia population (not just Phase 3 enriched cohort)",
                "supporting_evidence": "Phase 3 EMERGENT-2 used broad inclusion criteria; secondary endpoints including negative symptoms showed trends",
                "confidence": "Medium",
            },
        ],
        "scientific_risks": [
            "Long-term safety signal of chronic muscarinic agonism not fully established",
            "Negative symptom and cognitive endpoints require larger, dedicated trials",
            "Drug-drug interactions via CYP and transporter pathways for the combination",
        ],
        "scientific_strength_score": "Compelling",
        "key_insight": "The human target validation from Lilly's abandoned xanomeline program is the critical de-risking asset — Karuna essentially acquired a validated mechanism and engineered a tolerable formulation around it.",
    },
    "data_evidence": {
        "_module": "data_evidence",
        "most_advanced_data": "Phase 3",
        "data_summary": "EMERGENT-1 (Phase 2) and EMERGENT-2 (Phase 3) both met primary endpoints with statistically significant and clinically meaningful reductions in PANSS total score vs placebo. EMERGENT-3 (second Phase 3) also positive. Three positive randomized trials.",
        "trial_design_assessment": {
            "endpoint_quality": "Gold Standard",
            "control_quality": "RCT",
            "powering": "Adequately Powered",
            "patient_selection": "Broad",
            "notes": "5-week in-patient design raises real-world generalizability questions but is FDA-standard for schizophrenia trials. PANSS is the accepted regulatory endpoint.",
        },
        "signal_quality": {
            "effect_size": "Moderate",
            "consistency": "Consistent",
            "durability": "Short-term",
            "clinical_meaningfulness": "Clearly Meaningful",
        },
        "data_red_flags": [
            "5-week in-patient trial does not address long-term durability of effect or outpatient compliance",
            "No direct head-to-head vs atypical antipsychotics — comparative efficacy unknown",
        ],
        "evidence_classification": "Compelling",
        "key_data_gap": "12-month open-label extension data for durability and long-term safety; head-to-head vs SoC for prescriber confidence.",
    },
    "development_pathway": {
        "_module": "development_pathway",
        "current_stage": "NDA/BLA",
        "current_stage_detail": "NDA submitted to FDA in 2023; PDUFA date set. Two positive Phase 3 trials completed. On track for 2024 approval.",
        "inflection_points": [
            {
                "milestone": "FDA Approval (KarXT in Schizophrenia)",
                "description": "First FDA approval of a muscarinic agonist for schizophrenia. Binary event with transformative valuation impact.",
                "timeframe": "6-9 months from NDA submission",
                "capital_to_reach": "$50-80M (commercialization build)",
                "probability_of_success": "75-85%",
                "valuation_impact": "3-5x valuation step-up at approval; acquisition trigger",
                "key_risk": "AdCom safety scrutiny on cholinergic AEs; label restrictions on patient population",
            },
            {
                "milestone": "Commercial Launch + First Revenue Quarter",
                "description": "Real-world uptake data, payer contracting, and prescriber adoption — first signal of commercial potential.",
                "timeframe": "12-18 months post-approval",
                "capital_to_reach": "$150-250M (commercial ramp)",
                "probability_of_success": "60-70%",
                "valuation_impact": "Determines peak revenue multiple; strong launch = acquisition at premium",
                "key_risk": "Formulary placement and payer step-edit requirements behind generics",
            },
            {
                "milestone": "Cognitive/Negative Symptom Phase 3 (KarXT-CN)",
                "description": "Dedicated trial in cognitive and negative symptoms of schizophrenia — the unmet need that drives premium pricing and label expansion.",
                "timeframe": "3-5 years",
                "capital_to_reach": "$200-400M",
                "probability_of_success": "40-55%",
                "valuation_impact": "Label expansion doubles addressable opportunity",
                "key_risk": "Cognitive endpoints are notoriously difficult — high regulatory bar",
            },
        ],
        "total_capital_to_value_event": "$250-400M to launch; M&A likely before full commercial buildout",
        "critical_path_risks": [
            "AdCom / FDA label negotiation on cholinergic side effect language",
            "Commercial team build quality — rare disease vs psychiatry commercial infrastructure",
            "Generic competition timeline from branded antipsychotics",
        ],
        "partnership_trigger": "Post-Phase 3 package with NDA filing — full data package triggers strategic M&A or major co-promotion deal",
        "development_confidence": "High",
    },
    "competitive_landscape": {
        "_module": "competitive_landscape",
        "direct_competitors": [
            {
                "company": "AbbVie",
                "asset": "Emraclidine (M4 PAM)",
                "mechanism": "M4 positive allosteric modulator",
                "stage": "Phase 2",
                "key_differentiator": "More selective M4 mechanism; once-daily oral; significant lag behind KarXT",
            },
            {
                "company": "Cerevel (now AbbVie)",
                "asset": "Tavapadon",
                "mechanism": "D1/D5 partial agonist",
                "stage": "Phase 3 (Parkinson's)",
                "key_differentiator": "Different indication; Parkinson's focus not schizophrenia",
            },
            {
                "company": "Intra-Cellular Therapies",
                "asset": "Lumateperone (Caplyta)",
                "mechanism": "Multimodal serotonin/dopamine",
                "stage": "Approved",
                "key_differentiator": "Already on market; tolerability advantage over older agents; will compete for same prescribers",
            },
        ],
        "mechanism_level_competition": [
            {
                "mechanism": "Atypical antipsychotics (D2/5HT2A)",
                "representative_asset": "Risperidone (generic), Aripiprazole (generic), Abilify Maintena",
                "threat_level": "High",
                "rationale": "Deeply entrenched SoC; generic pricing creates enormous barrier to formulary positioning",
            },
        ],
        "differentiation": {
            "efficacy": {"score": "Equivalent", "rationale": "PANSS reduction comparable to atypicals; no head-to-head data yet"},
            "safety": {"score": "Ahead", "rationale": "No EPS, no metabolic syndrome, no tardive dyskinesia risk — the defining differentiation"},
            "convenience": {"score": "Behind", "rationale": "Twice-daily oral with food; some GI side effects (nausea, constipation) during titration"},
            "modality": {"score": "Ahead", "rationale": "Novel mechanism is structurally differentiated; no cross-resistance or class-effect liability"},
            "pricing_power": {"score": "High", "rationale": "First-in-class with clean safety profile justifies $30-50k/year brand pricing"},
        },
        "competitive_moat": "First-mover in muscarinic agonism for schizophrenia with three positive Phase 3 trials. CoM patents on the xanomeline-trospium combination. KOL relationships and clinical investigator network built over 5+ years.",
        "competitive_verdict": "KarXT wins on safety profile — the dominant unmet need in schizophrenia. Efficacy parity with atypicals is sufficient if tolerability advantage is durable. The risk is payer step-therapy requiring generic atypical failure first.",
        "biggest_competitive_threat": "Payer formulary gatekeeping — not clinical competition. Generic antipsychotics will require step-therapy in most formularies, limiting de novo prescribing.",
    },
    "market_commercial": {
        "_module": "market_commercial",
        "target_indication": "Schizophrenia (acute exacerbations and maintenance)",
        "patient_population": {
            "prevalence": "~3.5M diagnosed schizophrenia patients in the US",
            "treatable_fraction": "~70% are on pharmacotherapy; ~2.4M active treated patients",
            "addressable_patients": "~2.4M US / ~12M global (treated schizophrenia)",
        },
        "standard_of_care": {
            "current_best": "Atypical antipsychotics (aripiprazole, risperidone, quetiapine, lurasidone) — majority now generic",
            "unmet_need_level": "High",
            "unmet_need_description": "~40% of patients discontinue first-line antipsychotics within 18 months due to EPS, metabolic effects, or sedation. Negative symptoms and cognitive impairment remain unaddressed by all approved therapies.",
        },
        "market_sizing": {
            "peak_penetration_estimate": "3-6% of treated schizophrenia market (realistic for non-generic branded drug in psychiatry)",
            "price_per_patient_per_year": "$30,000-50,000 (benchmarked to Caplyta ~$30k, Vraylar ~$25k)",
            "peak_revenue_conservative": "$500-800M (US-only, 5-6% penetration at $35k)",
            "peak_revenue_bull": "$1.2-2.0B (including EU approval, label expansion to negative symptoms)",
            "sizing_confidence": "Medium",
        },
        "adoption_barriers": [
            "Step-therapy requirements — payers will require documented failure on ≥2 generic antipsychotics first",
            "Psychiatrist habit: strong preference for known agents with decades of experience",
            "Twice-daily dosing with food requirement reduces compliance vs long-acting injectables",
            "GI side effects during titration may limit uptake in outpatient settings",
        ],
        "commercial_model": "Own salesforce",
        "commercial_risk_level": "Medium",
        "key_commercial_insight": "The commercial case is built entirely on tolerability differentiation. If physicians and payers don't value the EPS/metabolic-free profile enough to pay a brand premium over $2/day generic aripiprazole, peak revenue will disappoint.",
    },
    "team_execution": {
        "_module": "team_execution",
        "founding_team": {
            "scientific_credibility": "Strong",
            "clinical_dev_experience": "Strong",
            "business_building_experience": "Strong",
            "notes": "Andrew Miller (CEO) and Fredrick Bhatt (CMO) are seasoned CNS operators. The company identified and licensed the xanomeline IP from Eli Lilly — a sharp business development move that few teams would have spotted.",
        },
        "management_team": {
            "ceo_quality": "Strong",
            "cmo_quality": "Strong",
            "cso_quality": "Strong",
            "overall_team_depth": "Deep",
        },
        "board_and_investors": {
            "investor_quality": "Tier 1",
            "key_investors": ["RA Capital", "Fidelity", "BlackRock", "Redmile Group"],
            "board_credibility": "Strong",
        },
        "execution_track_record": {
            "prior_drug_successes": "Yes",
            "prior_exits": "Yes",
            "notable_achievements": [
                "Three positive Phase 3 trials executed on time and on budget",
                "NDA filing completed in 2023",
                "Raised $1B+ in public markets to fund pivotal program",
            ],
        },
        "key_man_risk": "Medium",
        "team_overall_score": "A",
        "team_verdict": "A-grade CNS team with demonstrated ability to execute complex Phase 3 programs and raise institutional capital. The Lilly IP identification and licensing was a defining strategic insight. This team de-risked the asset systematically.",
    },
    "ip_regulatory": {
        "_module": "ip_regulatory",
        "patent_position": {
            "composition_of_matter": "Strong",
            "method_of_use": "Exists",
            "key_patents": [
                "Composition of matter on xanomeline-trospium combination (KarXT formulation)",
                "Method of use patents covering treatment of CNS disorders",
                "Dosing regimen and titration method patents",
            ],
            "earliest_expiry": "2035-2040 (combination CoM)",
            "exclusivity_extensions": [
                "NCE exclusivity through 2028-2029",
                "Pediatric exclusivity potentially available",
            ],
        },
        "freedom_to_operate": {
            "fto_status": "Clear",
            "key_fto_risks": [
                "Generic xanomeline is in public domain from Lilly; combination IP must hold against ANDA challenges"
            ],
            "fto_confidence": "High",
        },
        "regulatory_pathway": {
            "primary_pathway": "NDA (505b1) — New Chemical Entity for combination",
            "designations": ["Fast Track (Schizophrenia)"],
            "fda_alignment": "Confirmed via Type B meeting",
            "pathway_complexity": "Moderate",
            "precedent_exists": True,
        },
        "regulatory_risks": [
            "FDA may require longer-term safety data for label without duration restriction",
            "AdCom could scrutinize cholinergic AE profile and REMS implications",
            "CRL risk if EMERGENT-3 data shows any inconsistency with prior trials (already positive — low risk)",
        ],
        "existential_ip_regulatory_flags": [],
        "ip_regulatory_risk_level": "Low",
        "key_ip_insight": "The combination CoM patent on xanomeline + trospium is the key defensive asset. Generic manufacturers cannot simply use public-domain xanomeline without the peripheral restriction strategy — which is protected. This is a durable moat.",
    },
    "risk_decomposition": {
        "_module": "risk_decomposition",
        "risks": {
            "scientific": {
                "score": "Low",
                "key_driver": "Mechanism validated in three RCTs including two Phase 3 studies",
                "mitigant": "Human target validation from Lilly's prior program; three positive trials",
                "notes": "Scientific risk is essentially resolved. Remaining uncertainty is around long-term safety and negative/cognitive efficacy.",
            },
            "clinical": {
                "score": "Low",
                "key_driver": "Approval probability — remaining risk is label scope, not proof of concept",
                "mitigant": "FDA has accepted NDA; PDUFA date set; two positive Phase 3 trials",
                "notes": "Clinical risk shifts post-NDA to launch execution risk, not trial outcome risk.",
            },
            "regulatory": {
                "score": "Low",
                "key_driver": "Cholinergic AE profile may trigger narrow label or REMS",
                "mitigant": "Fast Track designation; precedent from other combination drugs; clean overall safety profile",
                "notes": "Primary regulatory risk is label language, not approval itself.",
            },
            "commercial": {
                "score": "Medium",
                "key_driver": "Payer step-therapy and physician habit will slow uptake vs projections",
                "mitigant": "Clear tolerability narrative; no viable generic equivalent of the mechanism",
                "notes": "This is the key remaining investable risk. Peak revenue range is $500M-2B depending on payer coverage.",
            },
            "financing": {
                "score": "Low",
                "key_driver": "Company is well-capitalized; IPO completed; institutional investor base is deep",
                "mitigant": "Access to public markets; potential acquirer interest removes financing risk entirely",
                "notes": "Financing risk is negligible at this stage. Acquisition risk is now the dominant scenario.",
            },
        },
        "overall_ptcs": {
            "range": "65-75%",
            "benchmark": "Industry average Phase 3 PTCS is ~55-65%; CNS Phase 3 is ~50%",
            "qualitative_assessment": "Above industry benchmark for this stage and indication",
        },
        "risk_that_determines_outcome": "Commercial adoption rate — specifically, whether payer coverage allows rapid uptake or restricts to treatment-resistant patients only.",
    },
    "investment_framing": {
        "_module": "investment_framing",
        "bull_case": {
            "scenario": "KarXT approved with broad label Q1 2024. BMS or J&J acquires for $12-16B at approval, paying ~6-8x forward peak revenue. Strong commercial launch validates the tolerability narrative and pressures competitor timelines.",
            "key_conditions": [
                "FDA approval with broad label (not restricted to treatment-resistant patients)",
                "Clean AdCom with no major safety concerns surfaced",
                "Major pharma strategic acquisition at or shortly after approval",
            ],
            "return_multiple": "15-25x from Series B entry; 3-5x from IPO entry",
            "exit_value": "$12-18B (strategic acquisition)",
        },
        "bear_case": {
            "most_likely_failure_mode": "Approval with narrow label (treatment-resistant only) combined with aggressive payer step-therapy = peak revenue of $200-350M. Not a fund-returner at post-Phase 3 valuations. No acquirer at a premium.",
            "capital_at_risk": "$400-600M invested before failure mode is apparent (2-3 years post-launch)",
            "residual_value": "Platform IP for other CNS indications; licensing revenue from mechanism validation",
        },
        "base_case": {
            "scenario": "FDA approval in 2024 with acceptable label. Slow commercial ramp due to payer friction (years 1-2), then inflection as KOL adoption accelerates. Acquired by pharma for $8-12B within 18 months of launch when peak revenue trajectory becomes clear.",
            "return_multiple": "8-12x from pre-Phase 3 entry",
            "exit_value": "$8-12B strategic acquisition",
        },
        "exit_scenarios": [
            {
                "type": "M&A",
                "acquirer_or_path": "Bristol Myers Squibb (CNS portfolio gap post-Eliquis LOE)",
                "rationale": "BMS has psychiatry infrastructure and $10B+ in annual BD capacity. KarXT fills a CNS white space and generates immediate branded revenue.",
                "estimated_value": "$12-16B",
                "probability": "50%",
                "timing": "Within 12-18 months of approval",
            },
            {
                "type": "M&A",
                "acquirer_or_path": "AstraZeneca (rebuilding CNS after Seroquel genericization)",
                "rationale": "AZ has shown CNS interest (Rare Disease/Neuroscience) and has commercial infrastructure in psychiatry internationally.",
                "estimated_value": "$8-12B",
                "probability": "25%",
                "timing": "12-24 months post-approval",
            },
            {
                "type": "IPO",
                "acquirer_or_path": "Remain independent — build commercial team and launch",
                "rationale": "Management has stated preference for independence; possible if early commercial traction is strong",
                "estimated_value": "$6-10B market cap",
                "probability": "15%",
                "timing": "Ongoing as public company",
            },
            {
                "type": "Partnership",
                "acquirer_or_path": "Ex-US licensing to AZ, Otsuka, or Lundbeck",
                "rationale": "OUS markets require established psychiatry salesforces Karuna doesn't have",
                "estimated_value": "$500M-2B in milestones + royalties",
                "probability": "10%",
                "timing": "Pre-approval or at approval",
            },
        ],
        "strategic_value": {
            "platform_potential": True,
            "defensive_ma_rationale": "Any pharma with a D2-based antipsychotic franchise needs to own or block the muscarinic mechanism before it takes share",
            "unique_strategic_assets": [
                "Validated M1/M4 muscarinic platform applicable to Alzheimer's psychosis, bipolar, and cognitive disorders",
                "Three completed Phase 3 datasets — clinical development expertise is embedded in the organization",
            ],
        },
        "fund_returner_potential": "Yes",
        "key_investment_insight": "This is a late-stage de-risked bet on commercial execution, not science. The science is done. The question is: what multiple does BMS pay to avoid competing against this in their psychiatric portfolio for the next decade?",
    },
    "decision_engine": {
        "_module": "decision_engine",
        "verdict": "INVEST",
        "confidence": "High",
        "top_3_reasons": [
            "Three positive Phase 3 RCTs — scientific and clinical risk is essentially resolved. This is one of the strongest data packages in recent CNS history.",
            "Unmet need is massive and durable: tolerability-driven discontinuation affects 40% of schizophrenia patients on current SoC. KarXT addresses the #1 complaint from patients and prescribers.",
            "Strategic M&A is the base case, not an upside scenario. Multiple large pharma players (BMS, J&J, AZ) have existential reasons to own this mechanism. Acquirer universe is deep.",
        ],
        "top_3_risks": [
            "Commercial payer friction: step-therapy behind generics could limit peak revenue to $300-500M, destroying the strategic premium and acquisition thesis.",
            "Label scope: a narrow label (treatment-resistant only) halves the addressable market and kills the commercial narrative.",
            "Valuation entry point: at $6-7B market cap post-Phase 3, most of the upside is already priced. The investment case requires a $12B+ exit to generate 2x from this entry.",
        ],
        "watch_triggers": [],
        "pass_rationale": "",
        "suggested_entry_conditions": [
            "Entry at $4-5B or below (pre-NDA) captures full risk-adjusted return",
            "Preferred structure: equity with anti-dilution at IPO pricing or earlier",
            "Minimum 5% ownership target to be meaningful at exit",
        ],
        "comparable_deals": [
            "Roivant / Myovant (relugolix) — late-stage branded drug with strong mechanism differentiation; acquired by Sumitovant at $2.9B — similar commercial friction thesis but smaller market",
            "Global Blood Therapeutics (GBT, voxelotor) — late-stage rare disease with strong mechanism; acquired by Pfizer for $5.4B — comparable risk profile at Phase 3 completion",
            "Arena Pharmaceuticals (ralinepag) — Phase 3 CNS with clear acquirer thesis; acquired by Pfizer for $6.7B pre-launch",
        ],
        "ic_one_liner": "KarXT is the most de-risked investable CNS asset of the decade — three Phase 3 wins, a mechanism that solves the #1 schizophrenia problem, and a pharma acquirer universe that cannot afford to let this go independent.",
    },
}


def main():
    gen = MemoGenerator()
    memo = gen.generate(
        company="Karuna Therapeutics (KarXT — xanomeline-trospium)",
        results=MOCK_RESULTS,
        run_id="DEMO_20231001_000000",
        format="markdown",
    )

    out_path = Path("memos/karuna_therapeutics_DEMO.md")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(memo)
    print(f"Demo memo written to: {out_path}")
    print("\n" + "="*60)
    print(memo[:3000] + "\n... [truncated — see full file]")


if __name__ == "__main__":
    main()
