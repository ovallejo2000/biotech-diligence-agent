"""
Public Markets Diligence Orchestrator
Coordinates 10 analysis modules against live API data (EDGAR, ClinicalTrials,
FDA, USPTO, PubMed), then synthesises with an LLM.

Guardrails (from spec):
  - Every module response includes _confidence, _sources, _nulls_flagged, _data_ts
  - LLM is never used to infer missing data — only to analyse what was provided
  - Discrepancies between sources are flagged explicitly in each module
  - Confidence is rated High / Medium / Low per module based on data availability
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable

from ..providers import auto_client
from .data_fetchers import collect_all
from .rnpv import rnpv_portfolio, classify_indication, classify_modality, WACC_BASE


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM = """You are a senior public markets biotech analyst at a top-tier life sciences hedge fund.

Your job is rigorous, data-driven investment analysis of publicly listed biotech companies.
You work strictly from the primary data provided — you NEVER infer or interpolate missing fields.

Rules you always follow:
- Ground every conclusion in the specific data in the prompt. Quote exact figures.
- If a field is not in the provided data, return null and add it to "nulls_flagged".
- Cross-reference sources; call out any discrepancy between them explicitly.
- Rate confidence: High = built on SEC + ClinicalTrials.gov primary sources.
  Medium = one primary source or partial data. Low = limited data, heavy inference.
- Flag stale data: clinical data >90 days old, financial data >180 days old.
- China exposure: flag any Chinese co-development, licensing, or manufacturing.
- Patent cliff: flag loss-of-exclusivity risk within 5 years for approved assets.
- Be direct about risks. The most dangerous analysis omits inconvenient data.
- Output structured JSON only. No prose outside JSON values. No markdown fences."""


# ─────────────────────────────────────────────────────────────────────────────
# State persistence
# ─────────────────────────────────────────────────────────────────────────────

_STATE_DIR = Path(".diligence_state/public")


def _save_run(ticker: str, results: dict, data: dict) -> str:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug   = ticker.upper()
    d      = _STATE_DIR / slug
    d.mkdir(parents=True, exist_ok=True)
    payload = {"run_id": run_id, "ticker": slug, "results": results, "raw_data": data}
    (d / f"{run_id}.json").write_text(json.dumps(payload, default=str))

    # Update index
    idx_path = d / "index.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else []
    summary = {
        "run_id":    run_id,
        "ticker":    slug,
        "company":   data.get("_company", slug),
        "timestamp": run_id,
        "verdict":   results.get("m10_rnpv", {}).get("investment_stance"),
        "equity_value_base": results.get("m10_rnpv", {}).get("rnpv", {}).get("equity_value_base_usd_m"),
    }
    idx.insert(0, summary)
    idx_path.write_text(json.dumps(idx[:50], default=str))
    return run_id


def load_run(ticker: str, run_id: str) -> Optional[dict]:
    p = _STATE_DIR / ticker.upper() / f"{run_id}.json"
    return json.loads(p.read_text()) if p.exists() else None


def list_public_runs() -> list:
    """Return summary list for all public market analyses."""
    if not _STATE_DIR.exists():
        return []
    summaries = []
    for slug_dir in sorted(_STATE_DIR.iterdir()):
        idx = slug_dir / "index.json"
        if idx.exists():
            try:
                entries = json.loads(idx.read_text())
                summaries.extend(entries)
            except Exception:
                pass
    summaries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return summaries


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class PublicMarketsOrchestrator:
    """
    Runs 10-module public markets due diligence on a listed biotech company.

    Usage:
        orch = PublicMarketsOrchestrator()
        for event in orch.analyze_stream("MRNA"):
            print(event)
    """

    MODULE_PIPELINE = [
        ("m01_identity",   "M1: Company Identity & Overview"),
        ("m02_pipeline",   "M2: Pipeline Mapping"),
        ("m03_catalysts",  "M3: Catalyst Calendar"),
        ("m04_pos",        "M4: Clinical Probability of Success"),
        ("m05_commercial", "M5: Commercial Opportunity"),
        ("m06_ip",         "M6: IP & Exclusivity"),
        ("m07_financial",  "M7: Financial Health"),
        ("m08_ownership",  "M8: Ownership & Sentiment"),
        ("m09_competitive","M9: Competitive Landscape"),
        ("m10_rnpv",       "M10: Risk Synthesis & rNPV"),
    ]

    def __init__(self, model: Optional[str] = None):
        self.client, detected_model = auto_client()
        self.model = model or detected_model

    # ── LLM call ─────────────────────────────────────────────────────────────

    def _call(self, prompt: str, max_tokens: int = 2000) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    def _parse(self, raw: str) -> dict:
        text = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                try:
                    return json.loads(m.group())
                except Exception:
                    pass
        return {"raw_output": text, "_parse_error": True}

    # ── Data helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _fmt_usd(val) -> str:
        if val is None:
            return "null"
        m = float(val)
        if abs(m) >= 1e9:
            return f"${m/1e9:.2f}B"
        if abs(m) >= 1e6:
            return f"${m/1e6:.1f}M"
        return f"${m:,.0f}"

    @staticmethod
    def _confidence(data_sources: list[bool]) -> str:
        filled = sum(1 for x in data_sources if x)
        ratio  = filled / len(data_sources) if data_sources else 0
        if ratio >= 0.75:
            return "High"
        if ratio >= 0.40:
            return "Medium"
        return "Low"

    # ── Streaming entry point ─────────────────────────────────────────────────

    def analyze_stream(self, ticker: str, extra_context: str = "",
                       progress_cb: Optional[Callable] = None):
        """
        Generator that yields SSE-ready dicts.
        Yields: {"type": "fetch"|"module_start"|"module_done"|"complete"|"error", ...}
        """
        ticker = ticker.strip().upper()

        # ── Phase 1: data collection ──────────────────────────────────────────
        fetch_log = []

        def on_fetch(source, status):
            fetch_log.append({"source": source, "status": status})
            if progress_cb:
                progress_cb({"type": "fetch", "source": source, "status": status})

        try:
            data = collect_all(ticker, progress_cb=on_fetch)
        except Exception as e:
            yield {"type": "error", "message": f"Data collection failed: {e}"}
            return

        if data.get("_fatal"):
            yield {"type": "error", "message": data["_fatal"]}
            return

        # ── Phase 2: module analysis ──────────────────────────────────────────
        results   = {}
        ctx       = {}   # accumulates prior module outputs as context
        total_mods = len(self.MODULE_PIPELINE)

        module_fns = {
            "m01_identity":   self._m01_identity,
            "m02_pipeline":   self._m02_pipeline,
            "m03_catalysts":  self._m03_catalysts,
            "m04_pos":        self._m04_pos,
            "m05_commercial": self._m05_commercial,
            "m06_ip":         self._m06_ip,
            "m07_financial":  self._m07_financial,
            "m08_ownership":  self._m08_ownership,
            "m09_competitive":self._m09_competitive,
            "m10_rnpv":       self._m10_rnpv,
        }

        for i, (key, label) in enumerate(self.MODULE_PIPELINE, 1):
            if progress_cb:
                progress_cb({"type": "module_start", "step": i, "total": total_mods,
                              "module": label})
            yield {"type": "module_start", "step": i, "total": total_mods,
                   "module": label}
            try:
                result = module_fns[key](data, ctx, extra_context)
            except Exception as e:
                result = {"_error": str(e), "_confidence": "Low",
                          "_module": key, "_module_label": label}
            result["_module"]       = key
            result["_module_label"] = label
            results[key] = result
            ctx[key]     = result
            yield {"type": "module_done", "step": i, "total": total_mods,
                   "module": label, "result": result}

        # ── Phase 3: save + return ────────────────────────────────────────────
        run_id = _save_run(ticker, results, data)
        yield {"type": "complete", "run_id": run_id, "ticker": ticker,
               "company": data.get("_company", ticker)}

    # ─────────────────────────────────────────────────────────────────────────
    # Module implementations
    # ─────────────────────────────────────────────────────────────────────────

    # ── M1: Company Identity ──────────────────────────────────────────────────

    def _m01_identity(self, data: dict, ctx: dict, extra: str) -> dict:
        fins  = data.get("financials", {})
        info  = data.get("company_info", {})
        fda   = data.get("fda", {})
        trials= data.get("trials", {})

        cash_str    = self._fmt_usd(fins.get("cash_usd"))
        rev_str     = self._fmt_usd(fins.get("revenue_usd"))
        runway      = fins.get("implied_runway_months")
        n_trials    = len(trials.get("trials", []))
        n_approved  = fda.get("count", 0)

        prompt = f"""
TASK: MODULE 1 — COMPANY IDENTITY & OVERVIEW

Ticker: {data['_ticker']}
Company name (from EDGAR): {data['_company']}
CIK: {data['_cik']}
SIC: {info.get('sic')} — {info.get('sic_description')}
State of Incorporation: {info.get('state_of_inc')}
Fiscal year end: {info.get('fiscal_year_end')}

Financial snapshot (from SEC EDGAR XBRL, fetched {fins.get('_fetched_at', 'unknown')}):
  Cash & equivalents: {cash_str} (as of {fins.get('cash_period_end', 'unknown')}, {fins.get('cash_form', '')})
  Revenue: {rev_str} (as of {fins.get('revenue_period_end', 'unknown')})
  Implied cash runway: {f"{runway} months" if runway else "null"}
  Monthly burn: {self._fmt_usd(fins.get('monthly_burn_usd'))}
  Null fields: {fins.get('_nulls', [])}

Pipeline breadth (from ClinicalTrials.gov, fetched {trials.get('_fetched_at', 'unknown')}):
  Active/completed trials found: {n_trials}
  Approved products (openFDA): {n_approved}

Most recent SEC filings: {json.dumps(info.get('filings', {}), indent=2)}

Extra context from user: {extra or 'none'}

Produce a one-paragraph company snapshot plus a structured identity block.
Include: therapeutic focus, development stage, pre-revenue vs commercial, pipeline breadth,
and any immediate flags (e.g. runway <12 months, pre-revenue with no near-term catalyst).

Return ONLY this JSON:
{{
  "snapshot_paragraph": "2–3 sentences a non-scientist can read in 30 seconds",
  "therapeutic_area": "primary focus",
  "stage": "Pre-clinical | Phase 1–2 | Phase 2–3 | Phase 3/NDA | Commercial",
  "commercial_status": "Pre-revenue | Partial revenue | Fully commercial",
  "modality_flag": "small_molecule | biologic | gene_therapy | cell_therapy | rna | mixed | unknown",
  "pipeline_depth": "number of active programmes",
  "cash_usd_m": {fins.get('cash_usd', 'null')},
  "runway_months": {runway if runway else 'null'},
  "immediate_flags": ["flag1"],
  "china_exposure": "flag if any Chinese co-development, licensing, or manufacturing is apparent",
  "nulls_flagged": ["any field not available from primary sources"],
  "_confidence": "High | Medium | Low",
  "_sources": ["SEC EDGAR XBRL", "ClinicalTrials.gov", "openFDA"],
  "_data_ts": "{fins.get('_fetched_at', 'unknown')}"
}}"""

        raw    = self._call(prompt, max_tokens=1200)
        result = self._parse(raw)
        result.setdefault("_confidence",
                          self._confidence([bool(fins.get("cash_usd")),
                                            bool(n_trials > 0), bool(info.get("name"))]))
        return result

    # ── M2: Pipeline Mapping ──────────────────────────────────────────────────

    def _m02_pipeline(self, data: dict, ctx: dict, extra: str) -> dict:
        trials = data.get("trials", {}).get("trials", [])
        fda    = data.get("fda", {}).get("approved_products", [])

        trials_json = json.dumps(trials[:20], indent=2)
        fda_json    = json.dumps(fda[:10], indent=2)
        m1          = ctx.get("m01_identity", {})

        prompt = f"""
TASK: MODULE 2 — PIPELINE MAPPING

Company: {data['_company']} ({data['_ticker']})
Modality flag from M1: {m1.get('modality_flag', 'unknown')}

ClinicalTrials.gov data (fetched {data['trials'].get('_fetched_at', 'unknown')}):
{trials_json}

FDA-approved products (openFDA, fetched {data['fda'].get('_fetched_at', 'unknown')}):
{fda_json}

Extra context: {extra or 'none'}

INSTRUCTIONS:
1. For each active programme, output: asset name, indication, mechanism/modality, phase,
   primary endpoint, enrollment, estimated completion date, next catalyst.
2. Cross-reference ClinicalTrials.gov vs what the company discloses in filings.
   Flag any discrepancy (e.g. trial listed as active in CT.gov but not mentioned in
   known filings, or vice versa).
3. Flag if any trial shows: low enrollment vs expected, completion date passed without
   reported results, or multiple amendments (suggesting protocol difficulties).
4. Note any China-based trials or Chinese CRO partnerships.

Return ONLY this JSON:
{{
  "programmes": [
    {{
      "name": "asset name",
      "indication": "indication",
      "mechanism": "MOA or modality",
      "phase": "Phase X",
      "primary_endpoint": "endpoint or null",
      "enrollment_count": 0,
      "estimated_completion": "YYYY-MM or null",
      "next_catalyst": "description or null",
      "nct_id": "NCT...",
      "flags": ["any programme-specific concerns"]
    }}
  ],
  "discrepancies": ["list any CT.gov vs filing mismatches"],
  "pipeline_summary": "1 sentence",
  "nulls_flagged": [],
  "_confidence": "High | Medium | Low",
  "_sources": ["ClinicalTrials.gov v2", "openFDA"],
  "_data_ts": "{data['trials'].get('_fetched_at', 'unknown')}"
}}"""

        raw    = self._call(prompt, max_tokens=2200)
        result = self._parse(raw)
        result.setdefault("_confidence",
                          self._confidence([bool(trials), bool(fda)]))
        return result

    # ── M3: Catalyst Calendar ─────────────────────────────────────────────────

    def _m03_catalysts(self, data: dict, ctx: dict, extra: str) -> dict:
        trials   = data.get("trials", {}).get("trials", [])
        pipeline = ctx.get("m02_pipeline", {}).get("programmes", [])

        # Build upcoming completion dates from CT.gov
        upcoming = []
        from datetime import date
        today_str = date.today().isoformat()
        for t in trials:
            comp = t.get("primary_completion_date") or t.get("estimated_completion")
            if comp and comp >= today_str and t.get("status") in (
                "RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION",
                "NOT_YET_RECRUITING"
            ):
                upcoming.append({
                    "nct_id":   t.get("nct_id"),
                    "title":    t.get("title"),
                    "phase":    t.get("phase"),
                    "conditions": t.get("conditions", [])[:2],
                    "completion": comp,
                    "status":   t.get("status"),
                })
        upcoming.sort(key=lambda x: x.get("completion", "9999"))

        prompt = f"""
TASK: MODULE 3 — CATALYST CALENDAR

Company: {data['_company']} ({data['_ticker']})
Today's date: {today_str}

Upcoming trial completion dates from ClinicalTrials.gov (sorted soonest first):
{json.dumps(upcoming[:15], indent=2)}

Pipeline from Module 2:
{json.dumps(pipeline[:10], indent=2)}

Extra context (may include PDUFA dates, management guidance, etc.): {extra or 'none'}

INSTRUCTIONS:
1. Produce a ranked catalyst calendar — nearest catalyst first.
2. For each catalyst: event type (data readout / PDUFA / regulatory filing / IND),
   expected date (from CT.gov completion date or user context), clinical significance
   (binary / de-risking / label-expanding), and estimated price-action magnitude
   (High / Medium / Low for a small/mid-cap).
3. Flag any catalyst where the CT.gov completion date has already passed but no
   results have been announced — this is a potential red flag.
4. Note: PDUFA dates are NOT in CT.gov — flag explicitly if PDUFA dates were not
   provided in the extra context (this is a null that should be sourced from the
   FDA calendar or BioPharma Catalyst).

Return ONLY this JSON:
{{
  "catalysts": [
    {{
      "event":         "Data readout / PDUFA / Filing / IND",
      "programme":     "asset name",
      "indication":    "indication",
      "expected_date": "YYYY-QX or YYYY-MM",
      "significance":  "Binary | De-risking | Label-expanding",
      "price_impact":  "High | Medium | Low",
      "source":        "ClinicalTrials.gov / User context / Not confirmed",
      "flag":          "null or concern"
    }}
  ],
  "pdufa_dates_available": false,
  "pdufa_source_note": "PDUFA dates not included in free data sources; source from FDA calendar or BioPharma Catalyst",
  "overdue_readouts": ["trials where completion date passed with no announced results"],
  "nulls_flagged": [],
  "_confidence": "High | Medium | Low",
  "_sources": ["ClinicalTrials.gov v2"],
  "_data_ts": "{data['trials'].get('_fetched_at', 'unknown')}"
}}"""

        raw    = self._call(prompt, max_tokens=2000)
        result = self._parse(raw)
        result.setdefault("_confidence",
                          self._confidence([bool(upcoming), bool(pipeline)]))
        return result

    # ── M4: Clinical Probability of Success ──────────────────────────────────

    def _m04_pos(self, data: dict, ctx: dict, extra: str) -> dict:
        pipeline = ctx.get("m02_pipeline", {}).get("programmes", [])
        m1       = ctx.get("m01_identity", {})

        # Run the rNPV PoS calculation for each programme
        from .rnpv import pos_adjusted, normalise_phase, classify_modality

        pos_calcs = []
        for p in pipeline[:10]:
            phase     = p.get("phase", "")
            indication = p.get("indication", "")
            mechanism  = p.get("mechanism", "")
            modality   = classify_modality(mechanism)
            pos_result = pos_adjusted(phase, indication, modality)
            pos_calcs.append({
                "name":      p.get("name"),
                "phase":     phase,
                "indication":indication,
                "modality":  modality,
                "pos_pct":   pos_result["pos_pct"],
                "pos":       pos_result["pos"],
                "adjustments": pos_result["adjustments"],
            })

        prompt = f"""
TASK: MODULE 4 — CLINICAL PROBABILITY OF SUCCESS

Company: {data['_company']} ({data['_ticker']})
Primary modality: {m1.get('modality_flag', 'unknown')}

Pre-calculated PoS estimates per programme (BIO/Citeline 2023 benchmarks,
adjusted for indication and modality):
{json.dumps(pos_calcs, indent=2)}

Trial details from Module 2:
{json.dumps(pipeline[:10], indent=2)}

Extra context: {extra or 'none'}

INSTRUCTIONS:
1. For each programme, validate or adjust the calculated PoS based on:
   - Trial design quality (adaptive / surrogate endpoint / sample size adequacy)
   - Designation status (Breakthrough / Fast Track / Orphan — if mentioned in context)
   - Competitive context (me-too vs first-in-class)
   - Historical success rate for this specific indication and modality
2. Identify the key PoS swing factors: what one trial design choice or regulatory
   decision would most change the PoS?
3. For gene/cell therapy: note manufacturing scalability risk as a separate PoS factor.

Return ONLY this JSON:
{{
  "pos_estimates": [
    {{
      "name":             "asset name",
      "phase":            "Phase X",
      "indication":       "indication",
      "modality":         "modality",
      "calculated_pos":   "XX.X%",
      "analyst_pos":      "XX.X%",
      "analyst_rationale":"1–2 sentences",
      "key_swing_factor": "single most important PoS driver",
      "designations":     ["BTD / FT / Orphan — or null"],
      "confidence":       "High | Medium | Low"
    }}
  ],
  "portfolio_weighted_pos": "weighted average PoS across pipeline (weight by phase)",
  "key_risk_observations": ["2–3 cross-pipeline observations"],
  "nulls_flagged": ["fields not determinable from provided data"],
  "_confidence": "High | Medium | Low",
  "_sources": ["BIO/Citeline 2023 PoS benchmarks", "ClinicalTrials.gov"],
  "_data_ts": "{data['trials'].get('_fetched_at', 'unknown')}"
}}"""

        raw    = self._call(prompt, max_tokens=2000)
        result = self._parse(raw)
        result["_pos_calculations"] = pos_calcs   # Always include the raw math
        result.setdefault("_confidence",
                          self._confidence([bool(pipeline), bool(pos_calcs)]))
        return result

    # ── M5: Commercial Opportunity ────────────────────────────────────────────

    def _m05_commercial(self, data: dict, ctx: dict, extra: str) -> dict:
        pipeline    = ctx.get("m02_pipeline", {}).get("programmes", [])
        fda_approved = data.get("fda", {}).get("approved_products", [])
        m1          = ctx.get("m01_identity", {})

        from .rnpv import classify_indication, _PEAK_SALES_BENCHMARKS

        # Pre-populate market benchmarks for each programme's indication
        indication_benchmarks = {}
        for p in pipeline[:10]:
            ind  = p.get("indication", "")
            cat  = classify_indication(ind)
            bm   = _PEAK_SALES_BENCHMARKS.get(cat, _PEAK_SALES_BENCHMARKS["default"])
            indication_benchmarks[p.get("name", "?")] = {
                "indication_category": cat,
                "benchmark_bear_usdm": bm["bear"],
                "benchmark_base_usdm": bm["base"],
                "benchmark_bull_usdm": bm["bull"],
                "note": "Published drug launch comps — not company-specific guidance",
            }

        prompt = f"""
TASK: MODULE 5 — COMMERCIAL OPPORTUNITY

Company: {data['_company']} ({data['_ticker']})
Commercial status: {m1.get('commercial_status', 'unknown')}
Approved products: {json.dumps(fda_approved[:5], indent=2)}

Pipeline from Module 2:
{json.dumps(pipeline[:8], indent=2)}

Industry peak-sales benchmarks by indication (bear / base / bull, USD millions):
{json.dumps(indication_benchmarks, indent=2)}

Extra context (may include company guidance, analyst estimates): {extra or 'none'}

INSTRUCTIONS:
1. For each approved or late-stage (Ph3+) asset: estimate addressable patient population,
   current standard of care, pricing benchmarks from comparable drugs, and peak sales range.
2. For earlier-stage assets: summarise indication-level market size from the benchmarks
   provided; do NOT fabricate specific estimates — use the benchmark ranges.
3. Flag payer landscape risks: overcrowded indication, pricing pressure, combination vs
   monotherapy reimbursement challenges.
4. Flag patent cliff risk for any approved product: if loss of exclusivity appears
   within 5 years based on approval date, flag it explicitly.
5. China exposure: flag if any commercial strategy depends on China market access.

Return ONLY this JSON:
{{
  "commercial_assessments": [
    {{
      "asset":            "name",
      "stage":            "Approved / Ph3 / Ph2 / Ph1",
      "indication":       "indication",
      "patient_population":"estimate or null",
      "current_soc":      "standard of care description",
      "pricing_benchmark":"comparable drug + price range, or null",
      "peak_sales_bear_usdm": 0,
      "peak_sales_base_usdm": 0,
      "peak_sales_bull_usdm": 0,
      "peak_sales_source":"benchmark or analyst or null",
      "payer_risks":      ["risks"],
      "patent_cliff_flag":"null or 'LOE in YYYY — $XM revenue at risk'"
    }}
  ],
  "total_addressable_revenue_comment": "1–2 sentences on combined opportunity",
  "china_exposure_flag": "null or description",
  "nulls_flagged": [],
  "_confidence": "High | Medium | Low",
  "_sources": ["openFDA", "indication peak-sales benchmarks", "ClinicalTrials.gov"],
  "_data_ts": "{data['fda'].get('_fetched_at', 'unknown')}"
}}"""

        raw    = self._call(prompt, max_tokens=2200)
        result = self._parse(raw)
        result.setdefault("_confidence",
                          self._confidence([bool(fda_approved), bool(pipeline)]))
        return result

    # ── M6: IP & Exclusivity ──────────────────────────────────────────────────

    def _m06_ip(self, data: dict, ctx: dict, extra: str) -> dict:
        patents  = data.get("patents", {}).get("patents", [])
        fda_prod = data.get("fda", {}).get("approved_products", [])
        pipeline = ctx.get("m02_pipeline", {}).get("programmes", [])

        prompt = f"""
TASK: MODULE 6 — IP & EXCLUSIVITY

Company: {data['_company']} ({data['_ticker']})

USPTO PatentsView results (fetched {data['patents'].get('_fetched_at', 'unknown')}):
{json.dumps(patents[:15], indent=2)}
Total patents found: {data['patents'].get('count', 0)}

FDA-approved products (for Orange Book cross-reference):
{json.dumps(fda_prod[:5], indent=2)}

Pipeline summary:
{json.dumps([{{"name": p.get("name"), "indication": p.get("indication"),
               "mechanism": p.get("mechanism")}} for p in pipeline[:8]], indent=2)}

Extra context: {extra or 'none'}

INSTRUCTIONS:
1. Summarise the patent portfolio breadth (count, approximate date range of patents).
2. For any approved product: flag loss-of-exclusivity risk.
   Standard drug patent life is 20 years from filing; typical marketed exclusivity
   is 12–15 years after approval date. Flag if any approved asset appears to face
   LOE within 5 years from today.
3. Identify whether the pipeline IP appears to be composition-of-matter
   (strong protection) vs method-of-use (weaker, more easily challenged).
4. Flag freedom-to-operate risks if any competitor patent in the same mechanism
   class appears likely (based on indication / mechanism names).
5. IMPORTANT: Full Orange Book patent listings require the paid Drugs@FDA web
   interface. Flag this as a data gap if not available.

Return ONLY this JSON:
{{
  "patent_count": {data['patents'].get('count', 0)},
  "patent_date_range": {{"earliest": "YYYY", "latest": "YYYY"}} ,
  "portfolio_assessment": "1–2 sentences",
  "approved_asset_loe": [
    {{
      "asset":      "name",
      "approval_date": "YYYY",
      "estimated_loe": "YYYY or null",
      "flag":        "null or 'LOE within 5 years'"
    }}
  ],
  "ip_type_assessment": "composition-of-matter | method-of-use | mixed | unknown",
  "freedom_to_operate_risks": ["risks or 'none identified'"],
  "data_gaps": ["e.g. Orange Book listing requires manual lookup"],
  "nulls_flagged": [],
  "_confidence": "High | Medium | Low",
  "_sources": ["USPTO PatentsView", "openFDA"],
  "_data_ts": "{data['patents'].get('_fetched_at', 'unknown')}"
}}"""

        raw    = self._call(prompt, max_tokens=1800)
        result = self._parse(raw)
        result.setdefault("_confidence",
                          self._confidence([bool(patents), bool(fda_prod)]))
        return result

    # ── M7: Financial Health ──────────────────────────────────────────────────

    def _m07_financial(self, data: dict, ctx: dict, extra: str) -> dict:
        fins = data.get("financials", {})
        info = data.get("company_info", {})
        m1   = ctx.get("m01_identity", {})

        prompt = f"""
TASK: MODULE 7 — FINANCIAL HEALTH

Company: {data['_company']} ({data['_ticker']})
SEC EDGAR XBRL financial data (fetched {fins.get('_fetched_at', 'unknown')}):

  Cash & equivalents:  {self._fmt_usd(fins.get('cash_usd'))}
    Period:            {fins.get('cash_period_end', 'null')} ({fins.get('cash_form', '')})

  Revenue:             {self._fmt_usd(fins.get('revenue_usd'))}
    Period:            {fins.get('revenue_period_end', 'null')}

  Net income / loss:   {self._fmt_usd(fins.get('net_income_usd'))}
    Period:            {fins.get('net_income_period_end', 'null')}

  Operating cash flow (latest quarter): {self._fmt_usd(fins.get('operating_cashflow_quarterly_usd'))}
  Operating cash flow (annual):         {self._fmt_usd(fins.get('operating_cashflow_annual_usd'))}
    Period:            {fins.get('operating_cashflow_period_end', 'null')}

  Monthly burn (derived):  {self._fmt_usd(fins.get('monthly_burn_usd'))}
  Implied runway:          {fins.get('implied_runway_months', 'null')} months
  Shares outstanding:      {fins.get('shares_outstanding', 'null')}

Null fields from XBRL: {fins.get('_nulls', [])}

Most recent 10-K / 10-Q filing dates:
  10-K: {info.get('filings', {}).get('10-K', {}).get('date', 'null')}
  10-Q: {info.get('filings', {}).get('10-Q', {}).get('date', 'null')}

Extra context (may include debt, guidance, recent dilution): {extra or 'none'}

INSTRUCTIONS:
1. Assess financial health: cash position relative to burn, runway vs next major catalyst.
2. Flag if runway <12 months (high dilution risk), <6 months (critical).
3. Assess whether the company is negotiating from a position of strength or weakness
   (Jack Castle framing: short runway = weak negotiating position for partnerships/financings).
4. Revenue quality: distinguish product revenue vs licensing/milestone vs grant revenue.
5. Note data staleness: if the most recent 10-Q is >90 days old, flag it.

Return ONLY this JSON:
{{
  "cash_usd_m":          {round(fins.get('cash_usd', 0) / 1e6, 1) if fins.get('cash_usd') else "null"},
  "monthly_burn_usd_m":  {round(fins.get('monthly_burn_usd', 0) / 1e6, 2) if fins.get('monthly_burn_usd') else "null"},
  "runway_months":       {fins.get('implied_runway_months', 'null')},
  "runway_assessment":   "Comfortable (>18m) | Adequate (12-18m) | Tight (6-12m) | Critical (<6m) | null",
  "revenue_status":      "Pre-revenue | Product revenue | Mixed | null",
  "dilution_risk":       "Low | Medium | High | Imminent",
  "negotiating_position":"Strong | Adequate | Weak | null",
  "data_staleness_flag": "null or 'Most recent 10-Q is > 90 days old'",
  "key_observations":    ["2–3 most important financial observations"],
  "nulls_flagged":       {json.dumps(fins.get('_nulls', []))},
  "_confidence": "High | Medium | Low",
  "_sources": ["SEC EDGAR XBRL company concepts"],
  "_data_ts": "{fins.get('_fetched_at', 'unknown')}"
}}"""

        raw    = self._call(prompt, max_tokens=1500)
        result = self._parse(raw)
        result.setdefault("_confidence",
                          self._confidence([bool(fins.get("cash_usd")),
                                            bool(fins.get("monthly_burn_usd"))]))
        return result

    # ── M8: Ownership & Sentiment ─────────────────────────────────────────────

    def _m08_ownership(self, data: dict, ctx: dict, extra: str) -> dict:
        insider   = data.get("insider", {})
        ownership = data.get("ownership", {})
        trades    = insider.get("trades", [])
        sc_filings = ownership.get("filings", [])

        prompt = f"""
TASK: MODULE 8 — OWNERSHIP & SENTIMENT

Company: {data['_company']} ({data['_ticker']})

Form 4 insider transactions (last 90 days, from SEC EDGAR,
fetched {insider.get('_fetched_at', 'unknown')}):
{json.dumps(trades[:15], indent=2)}
Total Form 4 filings found: {insider.get('count', 0)}

SC 13G/D institutional ownership filings (from SEC EDGAR,
fetched {ownership.get('_fetched_at', 'unknown')}):
{json.dumps(sc_filings[:10], indent=2)}

Data limitation note: {ownership.get('_note', 'none')}

Extra context: {extra or 'none'}

INSTRUCTIONS:
1. Summarise insider buying vs selling pattern over the last 90 days.
   Flag significant selling by C-suite or board members (bearish signal).
   Flag insider buying (bullish signal, especially if discretionary).
2. From SC 13G/D filings, identify any known institutional holders and recent
   changes in their positions (if available from the filing titles/summaries).
3. Note data limitations: full 13F holdings require parsing XBRL 13F filings.
   Be explicit about what is and is not available from the provided data.
4. Flag if no Form 4 activity at all — could indicate a blackout period or
   simply no recent transactions.

Return ONLY this JSON:
{{
  "insider_summary": {{
    "buys_90d":      0,
    "sells_90d":     0,
    "net_signal":    "Bullish | Neutral | Bearish | Insufficient data",
    "notable_events":["e.g. CEO sold X shares on DATE"]
  }},
  "institutional_summary": {{
    "known_holders": ["institution names from SC 13G/D if available"],
    "recent_changes":"description or 'Insufficient data from free sources'",
    "data_limitation":"13F full holdings data not available without paid source"
  }},
  "sentiment_flag":  "Positive | Neutral | Negative | Insufficient data",
  "nulls_flagged":   [],
  "_confidence": "Low",
  "_sources": ["SEC EDGAR Form 4", "SEC EDGAR SC 13G/D"],
  "_data_ts": "{insider.get('_fetched_at', 'unknown')}",
  "_note": "Full institutional ownership analysis requires 13F XBRL parsing or a paid data provider (Bloomberg, FactSet). This module uses only publicly available filing metadata."
}}"""

        raw    = self._call(prompt, max_tokens=1200)
        result = self._parse(raw)
        result.setdefault("_confidence", "Low")   # Always Low — 13F limitation
        return result

    # ── M9: Competitive Landscape ─────────────────────────────────────────────

    def _m09_competitive(self, data: dict, ctx: dict, extra: str) -> dict:
        pipeline    = ctx.get("m02_pipeline", {}).get("programmes", [])
        our_trials  = data.get("trials", {}).get("trials", [])

        # For each indication in our pipeline, collect competing programmes
        # from the CT.gov results (we already fetched all trials, so we can
        # search for matching indications in the full dataset)
        indications = list({p.get("indication", "") for p in pipeline[:6]
                           if p.get("indication")})

        prompt = f"""
TASK: MODULE 9 — COMPETITIVE LANDSCAPE

Company: {data['_company']} ({data['_ticker']})

Our pipeline (from Module 2):
{json.dumps(pipeline[:8], indent=2)}

Key indications we're pursuing: {indications}

All clinical trials data available (may include competitor trials):
{json.dumps(our_trials[:20], indent=2)}

Extra context (may include known competitors, analyst commentary): {extra or 'none'}

INSTRUCTIONS:
1. For each key indication in our pipeline, identify the competitive landscape:
   - Stage of competition (Phase 1/2/3, approved, withdrawn)
   - Same mechanism vs adjacent mechanism
   - Lead vs laggard positioning
2. Identify if any competitor appears to have a faster timeline to market
   (earlier completion date in CT.gov or already approved).
3. Assess differentiation: is our asset first-in-class, best-in-class, or me-too?
4. Flag indication-level crowding risk (>5 competitors in same mechanism class
   = high crowding risk).
5. Note: Full competitive intelligence requires GlobalData/Citeline/Evaluate Pharma.
   Flag where the CT.gov data provides incomplete competitive visibility.

Return ONLY this JSON:
{{
  "competitive_assessments": [
    {{
      "indication":       "indication",
      "our_asset":        "our programme name",
      "our_phase":        "Phase X",
      "competitors_identified": [
        {{
          "company":    "name (if determinable)",
          "mechanism":  "MOA",
          "phase":      "Phase X",
          "timeline":   "estimated completion or 'unknown'",
          "source":     "CT.gov / User context"
        }}
      ],
      "positioning":      "First-in-class | Best-in-class | Me-too | Unclear",
      "crowding_risk":    "Low (<3 competitors) | Medium (3-5) | High (>5)",
      "key_differentiator": "what makes our asset distinct, or null"
    }}
  ],
  "data_gap_note": "Full competitive intelligence requires Citeline/GlobalData/Evaluate Pharma",
  "nulls_flagged": [],
  "_confidence": "Medium",
  "_sources": ["ClinicalTrials.gov", "user context"],
  "_data_ts": "{data['trials'].get('_fetched_at', 'unknown')}"
}}"""

        raw    = self._call(prompt, max_tokens=2000)
        result = self._parse(raw)
        result.setdefault("_confidence", "Medium")
        return result

    # ── M10: rNPV Synthesis ───────────────────────────────────────────────────

    def _m10_rnpv(self, data: dict, ctx: dict, extra: str) -> dict:
        pipeline   = ctx.get("m02_pipeline", {}).get("programmes", [])
        pos_est    = ctx.get("m04_pos", {}).get("pos_estimates", [])
        commercial = ctx.get("m05_commercial", {}).get("commercial_assessments", [])
        fins       = ctx.get("m07_financial", {})
        m1         = ctx.get("m01_identity", {})

        # Build asset list for rNPV engine
        assets_for_rnpv = []
        for p in pipeline[:8]:
            name = p.get("name", "Unknown")
            ind  = p.get("indication", "")
            mech = p.get("mechanism", "")
            mod  = classify_modality(mech)
            phase = p.get("phase", "")

            # Peak sales from M5 if available
            ps_base = None
            for ca in commercial:
                if ca.get("asset") == name:
                    ps_base = ca.get("peak_sales_base_usdm")
                    break

            assets_for_rnpv.append({
                "name":      name,
                "indication":ind,
                "modality":  mod,
                "phase":     phase,
                "peak_sales_usd_m": ps_base,
            })

        # Pull cash and shares from financials
        cash_raw  = data.get("financials", {}).get("cash_usd") or 0
        shares_raw = data.get("financials", {}).get("shares_outstanding") or 0
        cash_m     = round(cash_raw / 1e6, 1) if cash_raw else 0.0
        shares_m   = round(shares_raw / 1e6, 1) if shares_raw else 0.0

        # Run the rNPV calculation
        rnpv_result = rnpv_portfolio(
            assets    = assets_for_rnpv,
            cash_usd_m = cash_m,
            wacc      = WACC_BASE,
            shares_m  = shares_m,
        )

        # Also run bear/bull WACC scenarios
        rnpv_low  = rnpv_portfolio(assets_for_rnpv, cash_m, wacc=0.10, shares_m=shares_m)
        rnpv_high = rnpv_portfolio(assets_for_rnpv, cash_m, wacc=0.15, shares_m=shares_m)

        # Synthesise with LLM
        prompt = f"""
TASK: MODULE 10 — RISK SYNTHESIS & rNPV

Company: {data['_company']} ({data['_ticker']})

=== QUANTITATIVE rNPV OUTPUT (calculated, not estimated) ===
Base case (WACC 12%):
{json.dumps(rnpv_result, indent=2)}

Sensitivity — Low WACC 10%:
  Equity value: bear ${rnpv_low.get('equity_value_bear_usd_m')}M /
                base ${rnpv_low.get('equity_value_base_usd_m')}M /
                bull ${rnpv_low.get('equity_value_bull_usd_m')}M

Sensitivity — High WACC 15%:
  Equity value: bear ${rnpv_high.get('equity_value_bear_usd_m')}M /
                base ${rnpv_high.get('equity_value_base_usd_m')}M /
                bull ${rnpv_high.get('equity_value_bull_usd_m')}M

=== PRIOR MODULE SYNTHESIS ===
Stage / commercial status: {m1.get('stage')} / {m1.get('commercial_status')}
Runway: {fins.get('runway_months', 'null')} months ({fins.get('dilution_risk', 'unknown')} dilution risk)
PoS estimates: {json.dumps(pos_est[:5], indent=2)}
Commercial assessments: {json.dumps(commercial[:4], indent=2)}

Key flags from all modules:
- Immediate flags (M1): {m1.get('immediate_flags', [])}
- Discrepancies (M2): {ctx.get('m02_pipeline', {}).get('discrepancies', [])}
- Overdue readouts (M3): {ctx.get('m03_catalysts', {}).get('overdue_readouts', [])}
- IP risks (M6): {ctx.get('m06_ip', {}).get('freedom_to_operate_risks', [])}
- Insider signal (M8): {ctx.get('m08_ownership', {}).get('sentiment_flag', 'Insufficient data')}
- Competitive crowding: {[c.get('crowding_risk') for c in ctx.get('m09_competitive', {}).get('competitive_assessments', [])[:3]]}

Extra context: {extra or 'none'}

INSTRUCTIONS:
1. Validate the quantitative rNPV output — do the numbers make sense given the
   pipeline stage and indication? Flag any assumptions that seem aggressive or
   conservative given what you know from the prior modules.
2. Identify the top 3 swing factors: the assumptions or events that would most
   change the rNPV output.
3. Identify key risks not captured in the quantitative model
   (management execution, partnership dependency, regulatory unpredictability).
4. Produce an investment stance: Constructive / Neutral / Cautious / Avoid.
5. Write a one-page executive summary a non-scientist can read in 2 minutes.

Return ONLY this JSON:
{{
  "rnpv": {{
    "equity_value_bear_usd_m": {rnpv_result.get('equity_value_bear_usd_m')},
    "equity_value_base_usd_m": {rnpv_result.get('equity_value_base_usd_m')},
    "equity_value_bull_usd_m": {rnpv_result.get('equity_value_bull_usd_m')},
    "per_share_base_usd":      {rnpv_result.get('per_share_base_usd', 'null')},
    "wacc_sensitivity": {{
      "10_pct_base_usd_m": {rnpv_low.get('equity_value_base_usd_m')},
      "12_pct_base_usd_m": {rnpv_result.get('equity_value_base_usd_m')},
      "15_pct_base_usd_m": {rnpv_high.get('equity_value_base_usd_m')}
    }},
    "methodology_note": "{rnpv_result.get('methodology', '')}"
  }},
  "top_swing_factors": [
    {{"factor": "description", "direction": "upside | downside", "magnitude": "high | medium"}}
  ],
  "key_risks_not_in_model": ["risk1", "risk2"],
  "investment_stance": "Constructive | Neutral | Cautious | Avoid",
  "investment_rationale": "2–3 sentences",
  "executive_summary": "One-page plain-English summary (4–6 sentences) covering: what this company does, where it is in development, key catalysts, financial position, main risk, and bottom line.",
  "china_exposure_aggregate": "Aggregated China flag from all modules or null",
  "patent_cliff_aggregate":   "Aggregated LOE flag from all modules or null",
  "data_confidence_aggregate": "High | Medium | Low — overall confidence in this analysis",
  "_confidence": "Medium",
  "_sources": ["All prior modules", "BIO/Citeline rNPV model"],
  "_data_ts": "{data.get('_collected_at', 'unknown')}"
}}"""

        raw    = self._call(prompt, max_tokens=2500)
        result = self._parse(raw)
        result["_rnpv_detail"] = rnpv_result   # Always attach raw rNPV output
        result.setdefault("_confidence", "Medium")
        return result
