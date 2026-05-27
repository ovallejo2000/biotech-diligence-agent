"""
rNPV (risk-adjusted NPV) engine for public markets biotech diligence.

Methodology (BIO/Citeline 2023 + standard industry practice):
  rNPV_asset = PoS × PeakSales × RevMultiple × DiscountFactor

  Where:
    PoS           = Phase-transition probability adjusted for indication + modality
    PeakSales     = Estimated peak annual net sales (USD millions)
    RevMultiple   = 5.0× peak sales (standard EV/peak-sales for biotech assets)
    DiscountFactor = 1 / (1 + WACC)^years_to_market

  Portfolio rNPV = Σ(rNPV per asset) + net cash position

Sources:
  - BIO / Citeline "Clinical Development Success Rates 2011-2020" (2023 update)
  - Standard buy-side biotech WACC range: 10-15%
  - Revenue multiples: Berenberg / Morgan Stanley biotech models (5x peak sales)

All assumptions are returned in the output dict so they can be audited.
"""

from __future__ import annotations
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Phase-transition PoS benchmarks  (Phase → Approval)
# Source: BIO/Citeline 2023 — overall across all indications
# ─────────────────────────────────────────────────────────────────────────────

_POS_BASE: dict[str, float] = {
    "PHASE1":   0.079,   # 7.9%  Phase 1 → Approval
    "PHASE2":   0.154,   # 15.4% Phase 2 → Approval
    "PHASE3":   0.577,   # 57.7% Phase 3 → Approval
    "NDA/BLA":  0.900,   # 90.0% Filed → Approval
    "APPROVED": 1.000,
}

# ─────────────────────────────────────────────────────────────────────────────
# Indication-level PoS adjustment multipliers (per phase)
# Source: BIO/Citeline indication-level sub-analysis
# ─────────────────────────────────────────────────────────────────────────────

_INDICATION_ADJ: dict[str, dict[str, float]] = {
    # Oncology has the lowest Phase 1 PoS (~5.1%) but BTD helps
    "oncology":          {"PHASE1": 0.65, "PHASE2": 0.75, "PHASE3": 0.88, "NDA/BLA": 0.96},
    # CNS has poor Phase 2 → Phase 3 attrition
    "cns":               {"PHASE1": 0.79, "PHASE2": 0.82, "PHASE3": 0.91, "NDA/BLA": 0.97},
    # Cardiovascular has better Phase 3 success historically
    "cardiovascular":    {"PHASE1": 1.12, "PHASE2": 1.18, "PHASE3": 1.08, "NDA/BLA": 1.00},
    # Infectious disease benefits from unmet need; anti-bacterial has lower PoS
    "infectious_disease":{"PHASE1": 1.18, "PHASE2": 1.25, "PHASE3": 1.05, "NDA/BLA": 1.00},
    # Rare disease often gets BTD, expedited review
    "rare_disease":      {"PHASE1": 1.10, "PHASE2": 1.15, "PHASE3": 1.20, "NDA/BLA": 1.05},
    "metabolic":         {"PHASE1": 1.00, "PHASE2": 1.05, "PHASE3": 1.00, "NDA/BLA": 1.00},
    "immunology":        {"PHASE1": 0.97, "PHASE2": 1.00, "PHASE3": 1.00, "NDA/BLA": 1.00},
    "respiratory":       {"PHASE1": 1.05, "PHASE2": 1.05, "PHASE3": 1.05, "NDA/BLA": 1.00},
}

# ─────────────────────────────────────────────────────────────────────────────
# Modality-level PoS adjustment multipliers
# Source: BIO/Citeline modality sub-analysis
# ─────────────────────────────────────────────────────────────────────────────

_MODALITY_ADJ: dict[str, float] = {
    "small_molecule":     1.10,  # Best-established, most historical data
    "monoclonal_antibody":1.05,
    "biologic":           1.00,
    "vaccine":            1.10,
    "rna":                0.85,  # mRNA / siRNA / ASO — newer modality, less history
    "aso":                0.82,
    "gene_therapy":       0.62,  # High unmet need but high failure rate
    "cell_therapy":       0.58,
    "unknown":            1.00,
}

# ─────────────────────────────────────────────────────────────────────────────
# Peak annual sales benchmarks by indication (USD millions)
# Source: Published drug launches, consensus analyst models
# ─────────────────────────────────────────────────────────────────────────────

_PEAK_SALES_BENCHMARKS: dict[str, dict[str, float]] = {
    "oncology_solid":       {"bear": 200,  "base": 900,   "bull": 3000},
    "oncology_hematology":  {"bear": 300,  "base": 1500,  "bull": 6000},
    "cns_alzheimer":        {"bear": 400,  "base": 2000,  "bull": 12000},
    "cns_other":            {"bear": 150,  "base": 700,   "bull": 2500},
    "cardiovascular":       {"bear": 300,  "base": 1200,  "bull": 5000},
    "rare_disease":         {"bear": 200,  "base": 700,   "bull": 2000},
    "infectious_disease":   {"bear": 200,  "base": 600,   "bull": 2000},
    "immunology":           {"bear": 500,  "base": 2200,  "bull": 9000},
    "metabolic":            {"bear": 400,  "base": 1500,  "bull": 6000},
    "respiratory":          {"bear": 300,  "base": 1000,  "bull": 3500},
    "default":              {"bear": 200,  "base": 800,   "bull": 3000},
}

# ─────────────────────────────────────────────────────────────────────────────
# Revenue multiple  (EV / peak annual sales) — standard buy-side biotech model
# ─────────────────────────────────────────────────────────────────────────────

REVENUE_MULTIPLE = 5.0     # 5× peak annual net sales = asset enterprise value

# ─────────────────────────────────────────────────────────────────────────────
# WACC range
# ─────────────────────────────────────────────────────────────────────────────

WACC_LOW  = 0.10   # Large cap, de-risked pipeline
WACC_BASE = 0.12   # Typical mid-cap biotech
WACC_HIGH = 0.15   # Small cap, high binary event risk

# ─────────────────────────────────────────────────────────────────────────────
# Years to commercialisation from current phase (median estimate)
# ─────────────────────────────────────────────────────────────────────────────

_YEARS_TO_MARKET: dict[str, float] = {
    "PHASE1":   7.5,
    "PHASE2":   5.0,
    "PHASE3":   2.5,
    "NDA/BLA":  1.0,
    "APPROVED": 0.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalise_phase(phase_raw: str) -> str:
    """Map a raw phase string to one of our standard keys."""
    p = (phase_raw or "").upper().replace(" ", "")
    if any(x in p for x in ["NDA", "BLA", "FILED", "SUBMISSION"]):
        return "NDA/BLA"
    if "APPROVED" in p or "MARKETED" in p or "COMMERCIAL" in p:
        return "APPROVED"
    if "4" in p:
        return "APPROVED"    # Phase 4 = post-approval, treat as commercial
    if "3" in p:
        return "PHASE3"
    if "2" in p:
        return "PHASE2"
    if "1" in p:
        return "PHASE1"
    return "PHASE2"          # Default to Phase 2 if unknown


def classify_indication(indication_text: str) -> str:
    """Map free-text indication to a benchmark category key."""
    t = (indication_text or "").lower()
    if any(w in t for w in ["leukemia", "lymphoma", "myeloma", "aml", "cll", "dll"]):
        return "oncology_hematology"
    if any(w in t for w in ["cancer", "carcinoma", "tumor", "sarcoma", "glioma",
                             "melanoma", "oncol", "nscl", "glioblas"]):
        return "oncology_solid"
    if any(w in t for w in ["alzheimer", "dementia"]):
        return "cns_alzheimer"
    if any(w in t for w in ["parkinson", "als", "huntington", "ms ", "multiple sclerosis",
                             "depression", "schizophrenia", "epilepsy", "neurol", "cns",
                             "migraine", "pain"]):
        return "cns_other"
    if any(w in t for w in ["heart failure", "coronary", "atrial fibril", "hfref",
                             "hypertension", "cardio", "myocardial", "cardiov"]):
        return "cardiovascular"
    if any(w in t for w in ["rare", "orphan", "duchenne", "sickle", "gaucher",
                             "fabry", "spinal muscular"]):
        return "rare_disease"
    if any(w in t for w in ["hiv", "infection", "hepatitis", "virus", "bacteria",
                             "sepsis", "covid", "influenza", "rsv"]):
        return "infectious_disease"
    if any(w in t for w in ["rheumatoid", "crohn", "colitis", "psoriasis", "lupus",
                             "autoimmune", "spondyl", "atopic"]):
        return "immunology"
    if any(w in t for w in ["diabetes", "obesity", "nash", "nafld", "lipid",
                             "metabolic", "thyroid", "gout"]):
        return "metabolic"
    if any(w in t for w in ["asthma", "copd", "pulmonary fibrosis", "respiratory",
                             "cystic fibrosis"]):
        return "respiratory"
    return "default"


def classify_modality(intervention_text: str) -> str:
    """Classify drug modality from intervention name."""
    t = (intervention_text or "").lower()
    if any(w in t for w in ["gene therapy", "aav", "lentiviral vector", "viral vector"]):
        return "gene_therapy"
    if any(w in t for w in ["car-t", "cart", "cell therapy", "t-cell therapy", "til"]):
        return "cell_therapy"
    if any(w in t for w in ["mrna", "sirna", "rnai", "antisense oligonucleotide", " aso",
                             "oligonucleotide"]):
        return "rna"
    if any(w in t for w in ["monoclonal antibody", " mab", "-mab", "bispecific antibody",
                             "antibody-drug conjugate", " adc"]):
        return "monoclonal_antibody"
    if any(w in t for w in ["vaccine", "immunization", "mrna vaccine"]):
        return "vaccine"
    if any(w in t for w in ["biologic", "protein", "fusion protein", "peptide", "enzyme"]):
        return "biologic"
    return "small_molecule"   # Default assumption


def indicator_to_pos_key(indication: str) -> str:
    """Map indication category to the _INDICATION_ADJ key."""
    ind_cat = classify_indication(indication)
    if ind_cat.startswith("oncology"):
        return "oncology"
    return {
        "cns_alzheimer": "cns",
        "cns_other":     "cns",
        "default":       "metabolic",   # neutral default
    }.get(ind_cat, ind_cat)


# ─────────────────────────────────────────────────────────────────────────────
# Core calculation functions
# ─────────────────────────────────────────────────────────────────────────────

def pos_adjusted(
    phase: str,
    indication: str,
    modality: str,
    has_breakthrough: bool = False,
    has_fast_track:   bool = False,
    adaptive_design:  bool = False,
    surrogate_endpoint: bool = False,
) -> dict:
    """
    Return probability of approval from current phase, adjusted for
    indication, modality, and trial design features.

    Returns {pos: float, pos_pct: str, adjustments: dict}
    """
    phase_key = normalise_phase(phase)
    base_pos  = _POS_BASE.get(phase_key, _POS_BASE["PHASE2"])

    # Indication adjustment
    ind_key = indicator_to_pos_key(indication)
    ind_adjs = _INDICATION_ADJ.get(ind_key, {})
    ind_adj  = ind_adjs.get(phase_key, 1.0)

    # Modality adjustment
    mod_adj = _MODALITY_ADJ.get(modality, 1.0)

    # Trial feature adjustments
    btd_adj  = 1.12 if has_breakthrough else 1.0   # BTD improves PoS ~10-15%
    ft_adj   = 1.06 if has_fast_track else 1.0
    ada_adj  = 1.04 if adaptive_design else 1.0
    sur_adj  = 0.92 if surrogate_endpoint else 1.0  # Surrogate endpoints can add FDA uncertainty

    pos = base_pos * ind_adj * mod_adj * btd_adj * ft_adj * ada_adj * sur_adj
    pos = round(min(pos, 0.98), 4)   # cap at 98%

    return {
        "pos":      pos,
        "pos_pct":  f"{pos * 100:.1f}%",
        "adjustments": {
            "base_phase":        f"{base_pos*100:.1f}% (BIO/Citeline 2023)",
            "indication_adj":    f"{ind_adj:.2f}× ({ind_key})",
            "modality_adj":      f"{mod_adj:.2f}× ({modality})",
            "breakthrough":      f"1.12×" if has_breakthrough else "none",
            "fast_track":        f"1.06×" if has_fast_track else "none",
            "adaptive_design":   f"1.04×" if adaptive_design else "none",
            "surrogate_endpoint":f"0.92×" if surrogate_endpoint else "none",
        },
    }


def rnpv_asset(
    name:             str,
    indication:       str,
    modality:         str,
    phase:            str,
    peak_sales_usd_m: Optional[float] = None,
    has_breakthrough: bool = False,
    has_fast_track:   bool = False,
    adaptive_design:  bool = False,
    surrogate_endpoint: bool = False,
    royalty_rate:     float = 1.0,    # 1.0 = fully owned; 0.15 = licensed at 15% royalty
    wacc:             float = WACC_BASE,
) -> dict:
    """
    Calculate rNPV for a single pipeline asset.

    Returns a fully-auditable dict with all intermediate values.
    """
    pos_result   = pos_adjusted(phase, indication, modality,
                                has_breakthrough, has_fast_track,
                                adaptive_design, surrogate_endpoint)
    pos          = pos_result["pos"]
    phase_key    = normalise_phase(phase)
    years        = _YEARS_TO_MARKET.get(phase_key, 5.0)

    # Peak sales estimate
    if peak_sales_usd_m is not None:
        peak_sales_base  = float(peak_sales_usd_m)
        peak_sales_src   = "provided"
        ind_cat = classify_indication(indication)
        benchmarks = _PEAK_SALES_BENCHMARKS.get(ind_cat, _PEAK_SALES_BENCHMARKS["default"])
        peak_sales_bear  = min(peak_sales_base * 0.45, benchmarks["bear"])
        peak_sales_bull  = max(peak_sales_base * 1.80, benchmarks["bull"])
    else:
        ind_cat = classify_indication(indication)
        benchmarks = _PEAK_SALES_BENCHMARKS.get(ind_cat, _PEAK_SALES_BENCHMARKS["default"])
        peak_sales_bear  = benchmarks["bear"]
        peak_sales_base  = benchmarks["base"]
        peak_sales_bull  = benchmarks["bull"]
        peak_sales_src   = f"indication benchmark ({ind_cat})"

    # Discount factor
    discount = 1 / ((1 + wacc) ** years)

    # rNPV = PoS × PeakSales × RevMultiple × DiscountFactor × RoyaltyRate
    def calc(ps):
        return round(pos * ps * REVENUE_MULTIPLE * discount * royalty_rate, 1)

    return {
        "asset":             name,
        "indication":        indication,
        "modality":          modality,
        "phase":             phase,
        "pos":               pos,
        "pos_pct":           pos_result["pos_pct"],
        "pos_adjustments":   pos_result["adjustments"],
        "peak_sales_bear":   peak_sales_bear,
        "peak_sales_base":   peak_sales_base,
        "peak_sales_bull":   peak_sales_bull,
        "peak_sales_source": peak_sales_src,
        "years_to_market":   years,
        "wacc":              f"{wacc*100:.0f}%",
        "revenue_multiple":  REVENUE_MULTIPLE,
        "royalty_rate":      royalty_rate,
        "rnpv_bear_usd_m":  calc(peak_sales_bear),
        "rnpv_base_usd_m":  calc(peak_sales_base),
        "rnpv_bull_usd_m":  calc(peak_sales_bull),
    }


def rnpv_portfolio(
    assets:       list[dict],
    cash_usd_m:   float = 0.0,
    debt_usd_m:   float = 0.0,
    shares_m:     float = 0.0,
    wacc:         float = WACC_BASE,
) -> dict:
    """
    Compute portfolio rNPV and implied equity value per share.

    assets: list of dicts accepted by rnpv_asset() — keys:
        name, indication, modality, phase,
        peak_sales_usd_m (opt), has_breakthrough, has_fast_track,
        adaptive_design, surrogate_endpoint, royalty_rate
    """
    asset_results = []
    bear_total = base_total = bull_total = 0.0

    for a in assets:
        r = rnpv_asset(
            name             = a.get("name", "Unnamed"),
            indication       = a.get("indication", ""),
            modality         = a.get("modality", "unknown"),
            phase            = a.get("phase", "PHASE2"),
            peak_sales_usd_m = a.get("peak_sales_usd_m"),
            has_breakthrough = a.get("has_breakthrough", False),
            has_fast_track   = a.get("has_fast_track", False),
            adaptive_design  = a.get("adaptive_design", False),
            surrogate_endpoint = a.get("surrogate_endpoint", False),
            royalty_rate     = a.get("royalty_rate", 1.0),
            wacc             = wacc,
        )
        asset_results.append(r)
        bear_total  += r["rnpv_bear_usd_m"]
        base_total  += r["rnpv_base_usd_m"]
        bull_total  += r["rnpv_bull_usd_m"]

    net_cash    = cash_usd_m - debt_usd_m
    eq_bear     = round(bear_total + net_cash, 1)
    eq_base     = round(base_total + net_cash, 1)
    eq_bull     = round(bull_total + net_cash, 1)

    result: dict = {
        "assets":                    asset_results,
        "pipeline_rnpv_bear_usd_m":  round(bear_total, 1),
        "pipeline_rnpv_base_usd_m":  round(base_total, 1),
        "pipeline_rnpv_bull_usd_m":  round(bull_total, 1),
        "cash_usd_m":                cash_usd_m,
        "debt_usd_m":                debt_usd_m,
        "net_cash_usd_m":            round(net_cash, 1),
        "equity_value_bear_usd_m":   eq_bear,
        "equity_value_base_usd_m":   eq_base,
        "equity_value_bull_usd_m":   eq_bull,
        "wacc_used":                 f"{wacc*100:.0f}%",
        "revenue_multiple_used":     REVENUE_MULTIPLE,
        "methodology": (
            "rNPV = Σ[PoS × PeakSales × 5.0× revenue_multiple / (1+WACC)^t]. "
            "PoS from BIO/Citeline 2023 success rates, adjusted by indication and modality. "
            "Peak sales: indication-level benchmarks where not supplied. "
            "Bear/base/bull peak-sales scenarios propagated through model. "
            "Equity value = pipeline rNPV + net cash."
        ),
    }

    if shares_m and shares_m > 0:
        result["per_share_bear_usd"] = round(eq_bear / shares_m, 2) if eq_bear else None
        result["per_share_base_usd"] = round(eq_base / shares_m, 2) if eq_base else None
        result["per_share_bull_usd"] = round(eq_bull / shares_m, 2) if eq_bull else None
        result["shares_m"]           = shares_m

    return result
