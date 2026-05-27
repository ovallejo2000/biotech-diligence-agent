"""
Data fetchers for public markets biotech diligence.

Sources used (all free, no API keys required):
  - SEC EDGAR submissions + XBRL company concepts  (data.sec.gov)
  - ClinicalTrials.gov v2 API                      (clinicaltrials.gov)
  - openFDA drug label / NDA API                   (api.fda.gov)
  - PubMed E-utilities                             (eutils.ncbi.nlm.nih.gov)
  - USPTO PatentsView                              (search.rpatentsview.org)

Guardrails implemented here:
  - Every result includes _fetched_at (UTC ISO-8601 timestamp).
  - Errors return {"_error": "...", "_data_available": False} — never raise.
  - No inference: missing fields are explicitly null, never estimated.
  - SEC rate limit respected: ≤10 req/s (0.12 s sleep between calls).
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
import urllib.parse
import urllib.error
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Optional

_UA = "BiotechPublicDiligence/1.0 research@biotechdiligence.app"
_EDGAR_SLEEP = 0.12   # SEC EDGAR: max 10 req/s


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


import ssl as _ssl
_SSL_CTX = _ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode    = _ssl.CERT_NONE


def _fetch_json(url: str, *, headers: dict | None = None,
                timeout: int = 20, body: bytes | None = None,
                method: str = "GET") -> dict:
    """
    Fetch JSON from url.  Always returns a dict.
    On failure returns {"_error": "...", "_data_available": False}.
    """
    h = {"User-Agent": _UA, "Accept": "application/json"}
    if headers:
        h.update(headers)
    try:
        req = urllib.request.Request(url, data=body, headers=h, method=method)
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as r:
            raw = r.read()
            # Handle gzip
            ce = r.headers.get("Content-Encoding", "")
            if "gzip" in ce:
                import gzip
                raw = gzip.decompress(raw)
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return {"_results": parsed, "_data_available": True}
        if isinstance(parsed, dict):
            parsed.setdefault("_data_available", True)
            return parsed
        return {"_raw": parsed, "_data_available": True}
    except urllib.error.HTTPError as e:
        return {"_error": f"HTTP {e.code} — {e.reason}", "_data_available": False, "_url": url}
    except urllib.error.URLError as e:
        return {"_error": f"Network error — {e.reason}", "_data_available": False, "_url": url}
    except json.JSONDecodeError as e:
        return {"_error": f"JSON parse error — {e}", "_data_available": False, "_url": url}
    except Exception as e:
        return {"_error": str(e), "_data_available": False, "_url": url}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  EDGAR — ticker → CIK
# ─────────────────────────────────────────────────────────────────────────────

def ticker_to_cik(ticker: str) -> dict:
    """
    Resolve a ticker symbol to a zero-padded 10-digit SEC CIK.
    Returns {"cik": "0001234567", "name": "...", "_fetched_at": "..."}
    or {"_error": "...", "_data_available": False}.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    time.sleep(_EDGAR_SLEEP)
    data = _fetch_json(url)
    if not data.get("_data_available", False):
        data["_fetched_at"] = _now()
        return data

    ticker_upper = ticker.strip().upper()
    for entry in data.values():
        if not isinstance(entry, dict):
            continue
        if entry.get("ticker", "").upper() == ticker_upper:
            cik = str(entry["cik_str"]).zfill(10)
            return {
                "cik": cik,
                "name": entry.get("title"),
                "ticker": ticker_upper,
                "_fetched_at": _now(),
                "_data_available": True,
            }
    return {
        "_error": f"Ticker '{ticker}' not found in SEC company tickers list.",
        "_data_available": False,
        "_fetched_at": _now(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2.  EDGAR — company submissions (recent filings index)
# ─────────────────────────────────────────────────────────────────────────────

def get_company_submissions(cik: str) -> dict:
    """
    Fetch EDGAR submissions for a CIK.
    Returns company metadata + index of recent 10-K / 10-Q / 8-K filings.
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    time.sleep(_EDGAR_SLEEP)
    data = _fetch_json(url)
    if not data.get("_data_available", False):
        data["_fetched_at"] = _now()
        return data

    recent = data.get("filings", {}).get("recent", {})
    forms      = recent.get("form", [])
    dates      = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    docs       = recent.get("primaryDocument", [])

    # Index most-recent filing per form type we care about
    index: dict[str, dict] = {}
    target_forms = {"10-K", "10-K/A", "10-Q", "10-Q/A", "8-K", "DEF 14A",
                    "SC 13G", "SC 13G/A", "SC 13D", "SC 13D/A"}
    for i, form in enumerate(forms):
        if form in target_forms and form not in index:
            acc_clean = accessions[i].replace("-", "") if i < len(accessions) else None
            index[form] = {
                "form": form,
                "date": dates[i] if i < len(dates) else None,
                "accession": acc_clean,
                "primary_doc": docs[i] if i < len(docs) else None,
                "url": (
                    f"https://www.sec.gov/Archives/edgar/full-index/"
                    f"{acc_clean}" if acc_clean else None
                ),
            }

    return {
        "name":            data.get("name"),
        "cik":             cik,
        "sic":             data.get("sic"),
        "sic_description": data.get("sicDescription"),
        "state_of_inc":    data.get("stateOfIncorporation"),
        "fiscal_year_end": data.get("fiscalYearEnd"),
        "filings":         index,
        "_fetched_at":     _now(),
        "_data_available": True,
        "_source":         f"SEC EDGAR submissions CIK {cik}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  EDGAR — XBRL concept fetcher (targeted, avoids huge company-facts blob)
# ─────────────────────────────────────────────────────────────────────────────

def _get_concept(cik: str, taxonomy: str, concept: str) -> list[dict]:
    """
    Fetch all reported values for a single XBRL concept.
    Returns list of {val, end, form} dicts, most-recent first, or [].
    """
    url = (f"https://data.sec.gov/api/xbrl/companyconcept/"
           f"CIK{cik}/{taxonomy}/{concept}.json")
    time.sleep(_EDGAR_SLEEP)
    data = _fetch_json(url)
    if not data.get("_data_available", False):
        return []

    unit_key = "USD" if taxonomy == "us-gaap" and "Shares" not in concept else "shares"
    entries = data.get("units", {}).get(unit_key, [])
    # Keep only 10-K and 10-Q entries; sort newest first
    keep = [e for e in entries if e.get("form", "").startswith("10-")]
    keep.sort(key=lambda x: x.get("end", ""), reverse=True)
    return keep


def _latest(entries: list[dict], form_prefix: str = "10-K") -> Optional[dict]:
    """Return the most-recent entry for a given form prefix."""
    for e in entries:
        if e.get("form", "").startswith(form_prefix):
            return e
    return entries[0] if entries else None


def get_financial_metrics(cik: str) -> dict:
    """
    Pull key financial metrics from EDGAR XBRL concept endpoints.
    Returns cash, revenue, net income, quarterly operating cash flow, shares,
    and derived burn rate / implied runway.

    Null fields are explicitly flagged — never estimated.
    """
    result: dict = {"_fetched_at": _now(), "_source": "SEC EDGAR XBRL concepts",
                    "_data_available": True, "_nulls": []}

    def fetch_usd(concept: str, fallback: str = None) -> Optional[dict]:
        entries = _get_concept(cik, "us-gaap", concept)
        if not entries and fallback:
            entries = _get_concept(cik, "us-gaap", fallback)
        best = _latest(entries, "10-K") or _latest(entries, "10-Q")
        return best   # {val, end, form} or None

    # Cash and equivalents
    cash = (fetch_usd("CashAndCashEquivalentsAtCarryingValue") or
            fetch_usd("CashCashEquivalentsAndShortTermInvestments") or
            fetch_usd("CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"))
    if cash:
        result["cash_usd"] = cash["val"]
        result["cash_period_end"] = cash["end"]
        result["cash_form"] = cash["form"]
    else:
        result["cash_usd"] = None
        result["_nulls"].append("cash_usd")

    # Revenue
    rev = (fetch_usd("Revenues") or
           fetch_usd("RevenueFromContractWithCustomerExcludingAssessedTax") or
           fetch_usd("RevenueFromContractWithCustomerIncludingAssessedTax"))
    if rev:
        result["revenue_usd"] = rev["val"]
        result["revenue_period_end"] = rev["end"]
        result["revenue_form"] = rev["form"]
    else:
        result["revenue_usd"] = None
        result["_nulls"].append("revenue_usd")

    # Net income / loss
    net = fetch_usd("NetIncomeLoss")
    if net:
        result["net_income_usd"] = net["val"]
        result["net_income_period_end"] = net["end"]
    else:
        result["net_income_usd"] = None
        result["_nulls"].append("net_income_usd")

    # Quarterly operating cash flow (most recent 10-Q)
    op_entries = _get_concept(cik, "us-gaap", "NetCashProvidedByUsedInOperatingActivities")
    q_op = _latest(op_entries, "10-Q")
    a_op = _latest(op_entries, "10-K")
    if q_op:
        result["operating_cashflow_quarterly_usd"] = q_op["val"]
        result["operating_cashflow_period_end"] = q_op["end"]
        # Derive monthly burn and implied runway
        if q_op["val"] and q_op["val"] < 0 and cash and cash["val"]:
            monthly_burn = abs(q_op["val"]) / 3
            runway_months = cash["val"] / monthly_burn
            result["monthly_burn_usd"] = round(monthly_burn)
            result["implied_runway_months"] = round(runway_months, 1)
        else:
            result["monthly_burn_usd"] = None
            result["implied_runway_months"] = None
    elif a_op:
        result["operating_cashflow_annual_usd"] = a_op["val"]
        result["operating_cashflow_period_end"] = a_op["end"]
        if a_op["val"] and a_op["val"] < 0 and cash and cash["val"]:
            monthly_burn = abs(a_op["val"]) / 12
            runway_months = cash["val"] / monthly_burn
            result["monthly_burn_usd"] = round(monthly_burn)
            result["implied_runway_months"] = round(runway_months, 1)
        else:
            result["monthly_burn_usd"] = None
            result["implied_runway_months"] = None
    else:
        result["operating_cashflow_quarterly_usd"] = None
        result["monthly_burn_usd"] = None
        result["implied_runway_months"] = None
        result["_nulls"].extend(["operating_cashflow", "monthly_burn_usd", "implied_runway_months"])

    # Shares outstanding
    sh_entries = _get_concept(cik, "us-gaap", "CommonStockSharesOutstanding")
    sh = _latest(sh_entries, "10-Q") or _latest(sh_entries, "10-K")
    result["shares_outstanding"] = sh["val"] if sh else None
    if not result["shares_outstanding"]:
        result["_nulls"].append("shares_outstanding")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4.  ClinicalTrials.gov v2
# ─────────────────────────────────────────────────────────────────────────────

def get_clinical_trials(company_name: str, max_results: int = 50) -> dict:
    """
    Fetch active and completed clinical trials for a sponsor.
    Returns list of trials with phase, status, conditions, dates, enrollment.
    """
    params = urllib.parse.urlencode({
        "query.spons":  company_name,
        "pageSize":     max_results,
        "format":       "json",
    })
    url = f"https://clinicaltrials.gov/api/v2/studies?{params}"
    data = _fetch_json(url, timeout=25)

    if not data.get("_data_available", False):
        return {"trials": [], "_error": data.get("_error"),
                "_fetched_at": _now(), "_data_available": False}

    trials = []
    for study in data.get("studies", []):
        ps = study.get("protocolSection", {})
        id_mod  = ps.get("identificationModule", {})
        st_mod  = ps.get("statusModule", {})
        des_mod = ps.get("designModule", {})
        con_mod = ps.get("conditionsModule", {})
        iv_mod  = ps.get("armsInterventionsModule", {})
        sp_mod  = ps.get("sponsorsCollaboratorsModule", {})
        oc_mod  = ps.get("outcomesModule", {})

        phases = des_mod.get("phases", [])
        ivs    = [iv.get("name") for iv in iv_mod.get("interventions", [])
                  if iv.get("name")]

        # Primary outcome
        prim = oc_mod.get("primaryOutcomes", [])
        primary_endpoint = prim[0].get("measure") if prim else None

        trials.append({
            "nct_id":                  id_mod.get("nctId"),
            "title":                   id_mod.get("briefTitle"),
            "status":                  st_mod.get("overallStatus"),
            "phase":                   phases[0] if phases else "Unknown",
            "conditions":              con_mod.get("conditions", [])[:3],
            "interventions":           ivs[:3],
            "primary_endpoint":        primary_endpoint,
            "enrollment":              des_mod.get("enrollmentInfo", {}).get("count"),
            "start_date":              st_mod.get("startDateStruct", {}).get("date"),
            "primary_completion_date": st_mod.get("primaryCompletionDateStruct", {}).get("date"),
            "estimated_completion":    st_mod.get("completionDateStruct", {}).get("date"),
            "sponsor":                 sp_mod.get("leadSponsor", {}).get("name"),
            "is_sponsor":              (
                (sp_mod.get("leadSponsor", {}).get("name") or "").lower()
                in company_name.lower()
                or company_name.lower()
                in (sp_mod.get("leadSponsor", {}).get("name") or "").lower()
            ),
        })

    return {
        "trials":          trials,
        "total_found":     data.get("totalCount", len(trials)),
        "_fetched_at":     _now(),
        "_data_available": bool(trials),
        "_source":         "ClinicalTrials.gov v2 API",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  openFDA — approved drugs
# ─────────────────────────────────────────────────────────────────────────────

def get_fda_approvals(company_name: str) -> dict:
    """
    Search openFDA Drugs@FDA for approved products by this company.
    """
    encoded = urllib.parse.quote(f'"{company_name}"')
    # Try drugsfda endpoint (NDA/BLA records)
    url = (f"https://api.fda.gov/drug/drugsfda.json?"
           f"search=sponsor_name:{encoded}&limit=20")
    data = _fetch_json(url, timeout=20)

    products = []
    if data.get("_data_available") and "results" in data:
        for r in data["results"][:15]:
            for p in r.get("products", []):
                products.append({
                    "brand_name":        p.get("brand_name"),
                    "active_ingredient": (p.get("active_ingredients") or [{}])[0].get("name"),
                    "dosage_form":       p.get("dosage_form"),
                    "marketing_status":  p.get("marketing_status"),
                    "approval_date":     p.get("approval_date"),
                    "application_number": r.get("application_number"),
                    "sponsor":           r.get("sponsor_name"),
                })

    return {
        "approved_products":  products,
        "count":              len(products),
        "_fetched_at":        _now(),
        "_data_available":    bool(products),
        "_source":            "openFDA Drugs@FDA API",
        "_note":              (
            None if products else
            "No approved products found via openFDA. "
            "Company may be pre-commercial or name may differ from sponsor record."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  SEC EDGAR — Form 4 insider activity
# ─────────────────────────────────────────────────────────────────────────────

def get_insider_trades(cik: str, days: int = 90) -> dict:
    """
    Fetch recent Form 4 insider trading filings from EDGAR.
    Returns list of recent insider transactions with date, insider, type.
    """
    from datetime import timedelta
    import datetime as _dt
    cutoff = (_dt.datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    url = (f"https://www.sec.gov/cgi-bin/browse-edgar"
           f"?action=getcompany&CIK={cik}&type=4"
           f"&dateb=&owner=include&count=40&output=atom")
    time.sleep(_EDGAR_SLEEP)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as r:
            xml_data = r.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return {"trades": [], "_error": str(e), "_data_available": False,
                "_fetched_at": _now()}

    try:
        ns = "{http://www.w3.org/2005/Atom}"
        root = ET.fromstring(xml_data)
        trades = []
        for entry in root.findall(f"{ns}entry")[:30]:
            date_el  = entry.find(f"{ns}updated")
            title_el = entry.find(f"{ns}title")
            link_el  = entry.find(f"{ns}link")
            date_str = (date_el.text or "")[:10] if date_el is not None else None
            if date_str and date_str < cutoff:
                continue
            trades.append({
                "date":    date_str,
                "summary": title_el.text if title_el is not None else None,
                "url":     link_el.get("href") if link_el is not None else None,
            })
    except Exception as e:
        return {"trades": [], "_error": f"XML parse error: {e}",
                "_data_available": False, "_fetched_at": _now()}

    return {
        "trades":          trades,
        "period_days":     days,
        "count":           len(trades),
        "_fetched_at":     _now(),
        "_data_available": True,
        "_source":         f"SEC EDGAR Form 4 (last {days} days)",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7.  SEC EDGAR — SC 13G/D ownership filings
# ─────────────────────────────────────────────────────────────────────────────

def get_ownership_filings(cik: str) -> dict:
    """
    Return recent SC 13G / 13D institutional ownership filings.
    Note: For full 13F holdings parsing, a dedicated parser is needed.
    This returns the filing metadata (dates, types) as a proxy.
    """
    url = (f"https://www.sec.gov/cgi-bin/browse-edgar"
           f"?action=getcompany&CIK={cik}&type=SC+13"
           f"&dateb=&owner=include&count=20&output=atom")
    time.sleep(_EDGAR_SLEEP)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as r:
            xml_data = r.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return {"filings": [], "_error": str(e), "_data_available": False,
                "_fetched_at": _now()}

    try:
        ns = "{http://www.w3.org/2005/Atom}"
        root = ET.fromstring(xml_data)
        filings = []
        for entry in root.findall(f"{ns}entry")[:20]:
            title_el = entry.find(f"{ns}title")
            date_el  = entry.find(f"{ns}updated")
            link_el  = entry.find(f"{ns}link")
            filings.append({
                "title": title_el.text if title_el is not None else None,
                "date":  (date_el.text or "")[:10] if date_el is not None else None,
                "url":   link_el.get("href") if link_el is not None else None,
            })
    except Exception:
        filings = []

    return {
        "filings":         filings,
        "count":           len(filings),
        "_fetched_at":     _now(),
        "_data_available": bool(filings),
        "_source":         "SEC EDGAR SC 13G/D filings",
        "_note":           (
            "Full 13F holdings data requires parsing XML XBRL filings. "
            "This returns SC 13G/D ownership declaration metadata only."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8.  PubMed — publication history
# ─────────────────────────────────────────────────────────────────────────────

def get_publications(company_name: str, max_results: int = 10) -> dict:
    """
    Get recent clinical-trial publications affiliated with the company from PubMed.
    """
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    search_params = urllib.parse.urlencode({
        "db":      "pubmed",
        "term":    f'"{company_name}"[Affiliation] AND "clinical trial"[Publication Type]',
        "retmode": "json",
        "retmax":  max_results,
        "sort":    "relevance",
    })
    search_data = _fetch_json(f"{base}esearch.fcgi?{search_params}", timeout=20)
    ids = search_data.get("esearchresult", {}).get("idlist", [])

    if not ids:
        return {
            "publications":    [],
            "total_found":     0,
            "_fetched_at":     _now(),
            "_data_available": False,
            "_source":         "PubMed E-utilities",
            "_note":           "No clinical trial publications found for this affiliate name.",
        }

    summary_params = urllib.parse.urlencode({
        "db":      "pubmed",
        "id":      ",".join(ids[:10]),
        "retmode": "json",
    })
    summary_data = _fetch_json(f"{base}esummary.fcgi?{summary_params}", timeout=20)

    pubs = []
    for pmid, item in summary_data.get("result", {}).items():
        if pmid == "uids":
            continue
        pubs.append({
            "pmid":    pmid,
            "title":   item.get("title"),
            "journal": item.get("fulljournalname"),
            "date":    item.get("pubdate"),
            "authors": [a.get("name") for a in item.get("authors", [])[:3]],
        })

    return {
        "publications":    pubs,
        "total_found":     int(search_data.get("esearchresult", {}).get("count", len(pubs))),
        "_fetched_at":     _now(),
        "_data_available": bool(pubs),
        "_source":         "PubMed E-utilities",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 9.  USPTO PatentsView — patent portfolio
# ─────────────────────────────────────────────────────────────────────────────

def get_patents(company_name: str, max_results: int = 20) -> dict:
    """
    Search USPTO PatentsView for patents assigned to this company.
    """
    body = json.dumps({
        "q": {"_contains": {"assignee_organization": company_name}},
        "f": ["patent_number", "patent_title", "patent_date",
              "assignee_organization"],
        "o": {"per_page": max_results, "sort_by": "patent_date",
              "sort_order": "desc"},
    }).encode()

    data = _fetch_json(
        "https://search.rpatentsview.org/api/v1/patent/",
        body=body,
        method="POST",
        headers={"Content-Type": "application/json"},
        timeout=20,
    )

    if not data.get("_data_available", False):
        return {
            "patents":         [],
            "_fetched_at":     _now(),
            "_data_available": False,
            "_error":          data.get("_error"),
            "_source":         "USPTO PatentsView",
        }

    patents = []
    for p in data.get("patents", []):
        patents.append({
            "number": p.get("patent_number"),
            "title":  p.get("patent_title"),
            "date":   p.get("patent_date"),
        })

    return {
        "patents":         patents,
        "count":           len(patents),
        "_fetched_at":     _now(),
        "_data_available": bool(patents),
        "_source":         "USPTO PatentsView API",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 10. Master collector — fetch everything for a ticker in one call
# ─────────────────────────────────────────────────────────────────────────────

def collect_all(ticker: str,
                progress_cb: Optional[callable] = None) -> dict:
    """
    Orchestrate all data fetches for a given ticker.

    progress_cb(source_name, status) — called before and after each fetch.
    Returns a bundle dict with all fetched data and a top-level _errors list.
    """
    def _step(name: str, fn, *args, **kwargs):
        if progress_cb:
            progress_cb(name, "fetching")
        result = fn(*args, **kwargs)
        if progress_cb:
            ok = result.get("_data_available", False)
            progress_cb(name, "ok" if ok else "error")
        return result

    # Step 1: CIK resolution
    cik_info = _step("SEC EDGAR (CIK lookup)", ticker_to_cik, ticker)
    if not cik_info.get("_data_available"):
        return {
            "_ticker": ticker.upper(),
            "_fatal":  cik_info.get("_error",
                        f"Could not resolve ticker '{ticker}' to an SEC CIK."),
            "_errors": [cik_info.get("_error")],
        }

    cik          = cik_info["cik"]
    company_name = cik_info.get("name") or ticker

    # Step 2: Company submissions
    subs = _step("SEC EDGAR (submissions)", get_company_submissions, cik)

    # Step 3: Financial metrics
    fins = _step("SEC EDGAR (financials)", get_financial_metrics, cik)

    # Step 4: Clinical trials
    trials = _step("ClinicalTrials.gov", get_clinical_trials, company_name)

    # Step 5: FDA approvals
    fda = _step("openFDA (approvals)", get_fda_approvals, company_name)

    # Step 6: Insider trades
    insider = _step("SEC EDGAR (Form 4 insider)", get_insider_trades, cik)

    # Step 7: Ownership filings
    ownership = _step("SEC EDGAR (SC 13G/D)", get_ownership_filings, cik)

    # Step 8: Publications
    pubs = _step("PubMed", get_publications, company_name)

    # Step 9: Patents
    patents = _step("USPTO PatentsView", get_patents, company_name)

    # Aggregate errors
    errors = [
        d.get("_error") for d in
        [subs, fins, trials, fda, insider, ownership, pubs, patents]
        if d.get("_error")
    ]

    return {
        "_ticker":      ticker.upper(),
        "_company":     company_name,
        "_cik":         cik,
        "_collected_at": _now(),
        "_errors":      errors,
        "company_info": {**cik_info, **subs},
        "financials":   fins,
        "trials":       trials,
        "fda":          fda,
        "insider":      insider,
        "ownership":    ownership,
        "publications": pubs,
        "patents":      patents,
    }
