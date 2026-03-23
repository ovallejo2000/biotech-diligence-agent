#!/usr/bin/env python3
"""
Biotech Diligence Agent — FastAPI Web Server
"""

import os
import json
import queue
import threading
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, StreamingResponse, Response
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
except ImportError:
    raise SystemExit("Run: pip3 install fastapi uvicorn")

from biotech_diligence.orchestrator import DiligenceOrchestrator, MODULE_MAP

app = FastAPI(title="Biotech Diligence Agent", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATIC_DIR = Path(__file__).parent / "static"

@app.get("/static/app.js")
def serve_js():
    return Response(content=(STATIC_DIR / "app.js").read_text(), media_type="application/javascript")


def get_orchestrator():
    return DiligenceOrchestrator(verbose=False)


def _friendly_error(raw: str) -> str:
    """Convert raw API error strings into human-readable messages."""
    import re
    # Groq / OpenAI rate limit: parse out the retry time
    if "rate_limit_exceeded" in raw or "Rate limit reached" in raw:
        retry_match = re.search(r"Please try again in ([\d]+m[\d.]+s|[\d.]+s|[\d.]+m)", raw)
        tpd_match = re.search(r"tokens per day.*?Limit ([\d,]+).*?Used ([\d,]+)", raw, re.DOTALL)
        if retry_match:
            wait = retry_match.group(1)
            if tpd_match:
                limit = int(tpd_match.group(1).replace(",", ""))
                used = int(tpd_match.group(2).replace(",", ""))
                pct = int(used / limit * 100)
                return (
                    f"Daily token limit reached ({pct}% used). "
                    f"Try again in {wait} — Groq's free tier refreshes on a 24-hour rolling window."
                )
            return f"Rate limit reached. Try again in {wait}."
        return "Rate limit reached. Please wait a few minutes and try again."
    return raw


# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------

class DiligenceRequest(BaseModel):
    company: str
    inputs: Optional[str] = None
    format: str = "markdown"

class ModuleRequest(BaseModel):
    company: str
    module: str
    inputs: Optional[str] = None
    prior_results: Optional[dict] = None

class UpdateRequest(BaseModel):
    company: str
    new_inputs: str
    modules_to_rerun: Optional[list[str]] = None
    format: str = "markdown"


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# Tokens used per full run (empirical average, used for daily estimate)
_TOKENS_PER_RUN     = 10_500
_DAILY_LIMIT        = 100_000
_MIN_TOKENS_TO_RUN  = 12_000   # hard block below this — not enough for a full run
_TOKEN_LOG          = Path(".diligence_state/token_log.json")


def _log_run_tokens(tokens: int = _TOKENS_PER_RUN):
    """Append a token-usage entry to the local daily log."""
    import time
    _TOKEN_LOG.parent.mkdir(exist_ok=True)
    entries = []
    if _TOKEN_LOG.exists():
        try:
            entries = json.loads(_TOKEN_LOG.read_text())
        except Exception:
            pass
    entries.append({"ts": time.time(), "tokens": tokens})
    _TOKEN_LOG.write_text(json.dumps(entries))


def _estimated_remaining() -> int:
    """Sum tokens used in the last 24 h from the local log."""
    import time
    if not _TOKEN_LOG.exists():
        return _DAILY_LIMIT
    try:
        entries = json.loads(_TOKEN_LOG.read_text())
        cutoff = time.time() - 86_400
        used = sum(e["tokens"] for e in entries if e["ts"] > cutoff)
        return max(0, _DAILY_LIMIT - used)
    except Exception:
        return _DAILY_LIMIT


@app.get("/tokens/check")
def check_tokens():
    """
    Pre-flight token check. Returns:
      available (bool), estimated_remaining (int), warn (bool), message (str)
    """
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        return {"available": True, "provider": "other", "warn": False}

    estimated = _estimated_remaining()

    # Hard block on daily estimate before even hitting the API
    if estimated < _MIN_TOKENS_TO_RUN:
        remainK = round(estimated / 1000, 1)
        return {
            "available": False,
            "provider": "groq",
            "estimated_remaining": estimated,
            "daily_limit": _DAILY_LIMIT,
            "message": (
                f"Not enough tokens for a full analysis (~{remainK}k remaining, need ~12k). "
                "Groq's free tier refreshes on a 24-hour rolling window — check back in a few hours."
            ),
        }

    # Live 1-token call to detect 429s and read per-minute headroom
    try:
        from openai import OpenAI
        model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key)
        raw = client.chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        minute_remaining = int(raw.headers.get("x-ratelimit-remaining-tokens", -1))
        return {
            "available": True,
            "provider": "groq",
            "estimated_remaining": estimated,
            "daily_limit": _DAILY_LIMIT,
            "minute_remaining": minute_remaining,
        }
    except Exception as e:
        err = str(e)
        if "rate_limit_exceeded" in err or "429" in err:
            return {
                "available": False,
                "provider": "groq",
                "estimated_remaining": estimated,
                "daily_limit": _DAILY_LIMIT,
                "message": _friendly_error(err),
            }
        # Unknown error — don't block the run
        return {"available": True, "provider": "groq", "estimated_remaining": estimated}

@app.get("/modules")
def list_modules():
    return {"modules": list(MODULE_MAP.keys())}

@app.get("/diligence/stream")
def stream_diligence(company: str, inputs: Optional[str] = None):
    """SSE endpoint — streams progress events then the final memo."""
    q = queue.Queue()

    def progress_callback(event: dict):
        q.put(("progress", event))

    def run():
        try:
            orch = get_orchestrator()
            memo = orch.run_full_diligence(
                company=company,
                inputs=inputs,
                output_format="markdown",
                progress_callback=progress_callback,
            )
            _log_run_tokens()
            q.put(("memo", memo))
        except Exception as e:
            q.put(("error", _friendly_error(str(e))))
        finally:
            q.put(("done", None))

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    def event_stream():
        while True:
            event_type, data = q.get()
            if event_type == "progress":
                yield f"event: progress\ndata: {json.dumps(data)}\n\n"
            elif event_type == "memo":
                yield f"event: memo\ndata: {json.dumps({'memo': data})}\n\n"
            elif event_type == "error":
                yield f"event: error\ndata: {json.dumps({'message': data})}\n\n"
                break
            elif event_type == "done":
                yield f"event: done\ndata: {{}}\n\n"
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.post("/diligence")
def run_full_diligence(req: DiligenceRequest):
    try:
        orch = get_orchestrator()
        memo = orch.run_full_diligence(company=req.company, inputs=req.inputs, output_format=req.format)
        return {"company": req.company, "memo": memo}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/diligence/module")
def run_single_module(req: ModuleRequest):
    if req.module not in MODULE_MAP:
        raise HTTPException(status_code=400, detail=f"Unknown module: {req.module}")
    try:
        orch = get_orchestrator()
        result = orch.run_module(req.module, req.company, req.inputs, req.prior_results)
        return {"company": req.company, "module": req.module, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/diligence/update")
def update_diligence(req: UpdateRequest):
    try:
        orch = get_orchestrator()
        memo = orch.update_with_new_data(req.company, req.new_inputs, req.modules_to_rerun, req.format)
        return {"company": req.company, "memo": memo}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{company}")
def get_history(company: str):
    from biotech_diligence.state_manager import StateManager
    return {"company": company, "runs": StateManager().list_runs(company)}

@app.get("/companies")
def list_companies():
    from biotech_diligence.state_manager import StateManager
    return {"companies": StateManager().list_companies()}

@app.get("/demo")
def get_demo():
    """Return the pre-built Karuna Therapeutics demo memo (no API key needed)."""
    import subprocess
    demo_path = Path(__file__).parent / "memos" / "karuna_therapeutics_DEMO.md"
    if not demo_path.exists():
        subprocess.run(["python3", "demo.py"], cwd=Path(__file__).parent)
    return {"company": "Karuna Therapeutics (KarXT \u2014 xanomeline-trospium)", "memo": demo_path.read_text()}


# ------------------------------------------------------------------
# Methodology page
# ------------------------------------------------------------------

@app.get("/methodology", response_class=HTMLResponse)
def methodology():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Diligence Methodology \u2014 Biotech Diligence Agent</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #7d8590; --blue: #388bfd;
    --green: #3fb950; --yellow: #d29922; --red: #f85149;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }

  /* Header */
  header { background: var(--surface); border-bottom: 1px solid var(--border);
           padding: 1rem 2rem; display: flex; align-items: center; gap: 1rem; }
  header h1 { font-size: 1.1rem; font-weight: 700; color: var(--text); }
  .badge-header { font-size: 0.7rem; background: #1f3a5f; color: #79c0ff;
                  padding: 2px 8px; border-radius: 10px; font-weight: 600; }
  .header-right { margin-left: auto; }
  .header-link { color: var(--muted); font-size: 0.8rem; text-decoration: none; }
  .header-link:hover { color: var(--text); }

  /* Page layout */
  .page { max-width: 860px; margin: 0 auto; padding: 3rem 2rem 5rem; }
  .page-title { font-size: 1.75rem; font-weight: 700; margin-bottom: 0.5rem; }
  .page-subtitle { color: var(--muted); font-size: 0.95rem; margin-bottom: 3rem; max-width: 600px; line-height: 1.6; }

  /* Timeline */
  .timeline { position: relative; padding-left: 2.5rem; }
  .timeline::before {
    content: ""; position: absolute; left: 0.85rem; top: 0; bottom: 0;
    width: 2px; background: var(--border);
  }

  /* Module card */
  .module-card { position: relative; margin-bottom: 2rem; }
  .module-card::before {
    content: ""; position: absolute; left: -2.5rem; top: 1.3rem;
    width: 10px; height: 10px; border-radius: 50%;
    background: var(--blue); border: 2px solid var(--bg);
    box-shadow: 0 0 0 2px var(--blue);
  }
  .module-inner { background: var(--surface); border: 1px solid var(--border);
                  border-radius: 10px; padding: 1.4rem 1.6rem; }
  .module-inner:hover { border-color: #388bfd55; }

  .module-header { display: flex; align-items: flex-start; gap: 1rem; margin-bottom: 0.75rem; }
  .module-badge { font-size: 0.7rem; font-weight: 700; background: #1f3a5f;
                  color: #79c0ff; padding: 3px 10px; border-radius: 10px;
                  white-space: nowrap; margin-top: 2px; }
  .module-name { font-size: 1.05rem; font-weight: 600; color: var(--text); }
  .module-desc { color: var(--muted); font-size: 0.875rem; line-height: 1.6; margin-bottom: 1rem; }

  /* Output tags */
  .outputs-label { font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
                   letter-spacing: 0.05em; color: var(--muted); margin-bottom: 0.4rem; }
  .outputs { display: flex; flex-wrap: wrap; gap: 0.4rem; }
  .tag { font-size: 0.72rem; background: #1c2128; border: 1px solid var(--border);
         color: #8b949e; padding: 2px 9px; border-radius: 20px; }

  /* Verdict section */
  .verdict-section { margin-top: 3rem; padding: 1.5rem; background: var(--surface);
                     border: 1px solid var(--border); border-radius: 10px; }
  .verdict-section h2 { font-size: 1rem; font-weight: 600; margin-bottom: 0.75rem; }
  .verdict-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem; }
  .verdict-card { padding: 1rem; border-radius: 8px; border: 1px solid; text-align: center; }
  .verdict-card.invest { background: #0d2818; border-color: #238636; }
  .verdict-card.watch  { background: #1a1500; border-color: #4a3800; }
  .verdict-card.pass   { background: #2b0a0a; border-color: #5c1a1a; }
  .verdict-card .v-emoji { font-size: 1.5rem; margin-bottom: 0.25rem; }
  .verdict-card .v-label { font-weight: 700; font-size: 0.9rem; }
  .verdict-card .v-desc  { font-size: 0.75rem; color: var(--muted); margin-top: 0.25rem; line-height: 1.5; }
  .invest .v-label { color: #3fb950; }
  .watch  .v-label { color: #d29922; }
  .pass   .v-label { color: #f85149; }
</style>
</head>
<body>
<header>
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#388bfd" stroke-width="2">
    <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
  </svg>
  <h1>Biotech Diligence Agent</h1>
  <span class="badge-header">VC-Grade Analysis</span>
  <div class="header-right">
    <a href="/" class="header-link">&larr; Back to Agent</a>
  </div>
</header>

<div class="page">
  <h1 class="page-title">Diligence Methodology</h1>
  <p class="page-subtitle">
    Every analysis runs 11 sequential modules, each feeding context into the next.
    The pipeline mirrors how a VC investment committee works &mdash; from a rapid
    go/no-go filter through to a final conviction verdict.
  </p>

  <div class="timeline">

    <div class="module-card">
      <div class="module-inner">
        <div class="module-header">
          <span class="module-badge">A</span>
          <span class="module-name">Rapid Screening</span>
        </div>
        <p class="module-desc">
          A fast 20-minute VC filter. Evaluates science quality, proof of concept, venture scale
          potential, and existential red flags to decide whether the opportunity warrants deeper
          diligence. Returns a PASS / SOFT PASS / FAIL verdict before any resources are committed.
        </p>
        <div class="outputs-label">Key outputs</div>
        <div class="outputs">
          <span class="tag">Verdict</span><span class="tag">Science quality</span>
          <span class="tag">Proof of concept</span><span class="tag">Venture scale</span>
          <span class="tag">Red flags</span><span class="tag">Top concern</span>
        </div>
      </div>
    </div>

    <div class="module-card">
      <div class="module-inner">
        <div class="module-header">
          <span class="module-badge">B</span>
          <span class="module-name">Scientific &amp; Mechanistic Validation</span>
        </div>
        <p class="module-desc">
          Assesses the soundness of the biological hypothesis and validates the target pathway.
          Classifies the asset as first-in-class, fast-follow, or me-too, and identifies the
          critical assumptions that must hold for the science to work.
        </p>
        <div class="outputs-label">Key outputs</div>
        <div class="outputs">
          <span class="tag">Biological hypothesis</span><span class="tag">Pathway validation</span>
          <span class="tag">Asset classification</span><span class="tag">Critical assumptions</span>
          <span class="tag">Scientific strength score</span><span class="tag">Key insight</span>
        </div>
      </div>
    </div>

    <div class="module-card">
      <div class="module-inner">
        <div class="module-header">
          <span class="module-badge">C</span>
          <span class="module-name">Data &amp; Evidence Quality</span>
        </div>
        <p class="module-desc">
          Evaluates the rigor and strength of the supporting data package &mdash; trial design,
          signal quality, biomarker consistency, and reproducibility. Classifies the overall
          evidence as compelling, emerging, or weak, and flags the most critical data gap.
        </p>
        <div class="outputs-label">Key outputs</div>
        <div class="outputs">
          <span class="tag">Most advanced data</span><span class="tag">Trial design assessment</span>
          <span class="tag">Signal quality</span><span class="tag">Data red flags</span>
          <span class="tag">Evidence classification</span><span class="tag">Key data gap</span>
        </div>
      </div>
    </div>

    <div class="module-card">
      <div class="module-inner">
        <div class="module-header">
          <span class="module-badge">D</span>
          <span class="module-name">Development Pathway &amp; Inflection Points</span>
        </div>
        <p class="module-desc">
          Maps the full development roadmap to fundable value inflection points. For each
          milestone, estimates capital requirements, timeframes, probability of success,
          and valuation impact &mdash; giving investors a clear picture of the risk/reward
          at each stage gate.
        </p>
        <div class="outputs-label">Key outputs</div>
        <div class="outputs">
          <span class="tag">Current stage</span><span class="tag">Inflection points</span>
          <span class="tag">Capital to value event</span><span class="tag">Critical path risks</span>
          <span class="tag">Partnership trigger</span><span class="tag">Development confidence</span>
        </div>
      </div>
    </div>

    <div class="module-card">
      <div class="module-inner">
        <div class="module-header">
          <span class="module-badge">E</span>
          <span class="module-name">Competitive Landscape &amp; Positioning</span>
        </div>
        <p class="module-desc">
          Identifies direct competitors and mechanism-level competition, then evaluates
          differentiation across efficacy, safety, convenience, modality, and pricing power.
          Determines whether the asset has a defensible competitive moat or risks being
          commoditised.
        </p>
        <div class="outputs-label">Key outputs</div>
        <div class="outputs">
          <span class="tag">Direct competitors</span><span class="tag">Differentiation dimensions</span>
          <span class="tag">Competitive moat</span><span class="tag">Competitive verdict</span>
          <span class="tag">Biggest threat</span>
        </div>
      </div>
    </div>

    <div class="module-card">
      <div class="module-inner">
        <div class="module-header">
          <span class="module-badge">F</span>
          <span class="module-name">Market &amp; Commercial Reality</span>
        </div>
        <p class="module-desc">
          Produces a realistic bottom-up market assessment. Sizes the patient population,
          evaluates standard of care, models pricing and peak revenue scenarios, and
          identifies the adoption barriers most likely to limit commercial uptake.
        </p>
        <div class="outputs-label">Key outputs</div>
        <div class="outputs">
          <span class="tag">Patient population</span><span class="tag">Market sizing</span>
          <span class="tag">Standard of care</span><span class="tag">Adoption barriers</span>
          <span class="tag">Commercial risk level</span><span class="tag">Key commercial insight</span>
        </div>
      </div>
    </div>

    <div class="module-card">
      <div class="module-inner">
        <div class="module-header">
          <span class="module-badge">G</span>
          <span class="module-name">Team &amp; Execution Risk</span>
        </div>
        <p class="module-desc">
          Assesses the quality and experience of the founding team, key executives, and board.
          Evaluates execution track record, key-man risk, and whether the team has the domain
          expertise to navigate clinical development and eventual commercialisation.
        </p>
        <div class="outputs-label">Key outputs</div>
        <div class="outputs">
          <span class="tag">Founding team</span><span class="tag">Management team</span>
          <span class="tag">Board &amp; investors</span><span class="tag">Execution track record</span>
          <span class="tag">Key-man risk</span><span class="tag">Team verdict</span>
        </div>
      </div>
    </div>

    <div class="module-card">
      <div class="module-inner">
        <div class="module-header">
          <span class="module-badge">H</span>
          <span class="module-name">IP &amp; Regulatory Risk</span>
        </div>
        <p class="module-desc">
          Evaluates patent strength and freedom-to-operate, assesses regulatory pathway
          clarity (breakthrough designation, orphan status, standard review), and flags
          any existential IP or regulatory risks that could undermine the investment thesis.
        </p>
        <div class="outputs-label">Key outputs</div>
        <div class="outputs">
          <span class="tag">Patent position</span><span class="tag">Freedom to operate</span>
          <span class="tag">Regulatory pathway</span><span class="tag">Regulatory risks</span>
          <span class="tag">IP risk level</span><span class="tag">Key IP insight</span>
        </div>
      </div>
    </div>

    <div class="module-card">
      <div class="module-inner">
        <div class="module-header">
          <span class="module-badge">I</span>
          <span class="module-name">Risk Decomposition</span>
        </div>
        <p class="module-desc">
          Decomposes all investment risk into five categories &mdash; scientific, clinical,
          regulatory, commercial, and financing &mdash; scoring each with key drivers and
          mitigants. Synthesises these into an overall Probability of Technical and
          Commercial Success (PTCS).
        </p>
        <div class="outputs-label">Key outputs</div>
        <div class="outputs">
          <span class="tag">Scientific risk score</span><span class="tag">Clinical risk score</span>
          <span class="tag">Regulatory risk score</span><span class="tag">Commercial risk score</span>
          <span class="tag">Financing risk score</span><span class="tag">Overall PTCS</span>
        </div>
      </div>
    </div>

    <div class="module-card">
      <div class="module-inner">
        <div class="module-header">
          <span class="module-badge">J</span>
          <span class="module-name">Investment Framing</span>
        </div>
        <p class="module-desc">
          Frames the investment opportunity for an IC presentation. Articulates bull, base,
          and bear cases with return multiples, models realistic exit scenarios, and assesses
          strategic value and fund-returner potential.
        </p>
        <div class="outputs-label">Key outputs</div>
        <div class="outputs">
          <span class="tag">Bull / base / bear cases</span><span class="tag">Exit scenarios</span>
          <span class="tag">Return multiples</span><span class="tag">Strategic value</span>
          <span class="tag">Fund-returner potential</span><span class="tag">Key investment insight</span>
        </div>
      </div>
    </div>

    <div class="module-card">
      <div class="module-inner">
        <div class="module-header">
          <span class="module-badge" style="background:#1a3a1a;color:#3fb950;">K</span>
          <span class="module-name">Decision Engine</span>
        </div>
        <p class="module-desc">
          Synthesises all prior modules into a decisive investment verdict. Produces
          an IC-ready one-liner, the top 3 reasons to invest, the top 3 risks, and
          (where relevant) the conditions under which a PASS could become a WATCH,
          or a WATCH an INVEST.
        </p>
        <div class="outputs-label">Key outputs</div>
        <div class="outputs">
          <span class="tag">Verdict (INVEST / WATCH / PASS)</span>
          <span class="tag">Confidence</span><span class="tag">Top 3 reasons</span>
          <span class="tag">Top 3 risks</span><span class="tag">Watch triggers</span>
          <span class="tag">Entry conditions</span><span class="tag">IC one-liner</span>
        </div>
      </div>
    </div>

  </div><!-- /timeline -->

  <div class="verdict-section">
    <h2>Final Verdicts</h2>
    <p style="color:var(--muted);font-size:0.875rem;">Module K produces one of three verdicts based on the full diligence synthesis.</p>
    <div class="verdict-grid">
      <div class="verdict-card invest">
        <div class="v-emoji">&#x2705;</div>
        <div class="v-label">INVEST</div>
        <div class="v-desc">Strong conviction. Science, data, team, and market align. Recommend proceeding to term sheet.</div>
      </div>
      <div class="verdict-card watch">
        <div class="v-emoji">&#x26A0;&#xFE0F;</div>
        <div class="v-label">WATCH</div>
        <div class="v-desc">Interesting but not yet actionable. Clear triggers identified that would upgrade to INVEST.</div>
      </div>
      <div class="verdict-card pass">
        <div class="v-emoji">&#x274C;</div>
        <div class="v-label">PASS</div>
        <div class="v-desc">Insufficient conviction at this time. Fundamental concerns outweigh the opportunity.</div>
      </div>
    </div>
  </div>

</div>
</body>
</html>"""


# ------------------------------------------------------------------
# Web UI
# ------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Biotech Diligence Agent</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #7d8590; --blue: #388bfd;
    --green: #3fb950; --yellow: #d29922; --red: #f85149;
    --purple: #a371f7; --teal: #39d353;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: var(--bg); color: var(--text); min-height: 100vh; }

  /* ── Header ── */
  header { background: var(--surface); border-bottom: 1px solid var(--border);
           padding: 1rem 2rem; display: flex; align-items: center; gap: 1rem; }
  header h1 { font-size: 1.1rem; font-weight: 700; color: var(--text); }
  .badge-header { font-size: 0.7rem; background: #1f3a5f; color: #79c0ff;
                  padding: 2px 8px; border-radius: 10px; font-weight: 600; }
  .header-right { margin-left: auto; display: flex; gap: 0.5rem; align-items: center; }
  .header-link { color: var(--muted); font-size: 0.8rem; text-decoration: none; }
  .header-link:hover { color: var(--text); }

  /* ── Layout ── */
  .layout { display: grid; grid-template-columns: 340px 1fr; height: calc(100vh - 57px); overflow: hidden; }

  /* ── Left Panel ── */
  .left-panel { background: var(--surface); border-right: 1px solid var(--border); overflow-y: auto;
                padding: 1.5rem; display: flex; flex-direction: column; gap: 1rem; }
  label { font-size: 0.75rem; color: var(--muted); font-weight: 600;
          text-transform: uppercase; letter-spacing: 0.05em; display: block; margin-bottom: 0.4rem; }
  input, textarea { width: 100%; background: var(--bg); border: 1px solid var(--border);
                    color: var(--text); padding: 0.6rem 0.8rem; border-radius: 6px;
                    font-size: 0.875rem; font-family: inherit; transition: border-color 0.15s; }
  input:focus, textarea:focus { outline: none; border-color: var(--blue); }
  textarea { min-height: 120px; resize: vertical; }
  .btn { width: 100%; padding: 0.65rem; border-radius: 6px; border: none; cursor: pointer;
         font-size: 0.875rem; font-weight: 600; transition: all 0.15s; }
  .btn-primary { background: var(--blue); color: white; }
  .btn-primary:hover { background: #58a6ff; }
  .btn-primary:disabled { background: var(--border); color: var(--muted); cursor: not-allowed; }
  .btn-secondary { background: transparent; color: var(--muted);
                   border: 1px solid var(--border); }
  .btn-secondary:hover { border-color: var(--muted); color: var(--text); }
  .divider { border: none; border-top: 1px solid var(--border); margin: 0.25rem 0; }

  /* ── Progress ── */
  #progress-panel { display: none; flex-direction: column; gap: 0.35rem; }
  .module-row { display: flex; align-items: center; gap: 0.5rem; font-size: 0.78rem; }
  .module-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--border); flex-shrink: 0; }
  .module-dot.running { background: var(--yellow); animation: pulse 1s infinite; }
  .module-dot.done { background: var(--green); }
  .module-label { color: var(--muted); }
  .module-label.running { color: var(--yellow); font-weight: 600; }
  .module-label.done { color: var(--muted); }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
  .progress-bar-wrap { background: var(--border); border-radius: 4px; height: 4px; margin: 0.5rem 0; }
  .progress-bar { background: var(--blue); height: 4px; border-radius: 4px;
                  width: 0%; transition: width 0.4s ease; }
  #progress-label { font-size: 0.75rem; color: var(--muted); text-align: center; }

  /* ── History ── */
  .history-item { padding: 0.5rem 0.6rem; border-radius: 6px; cursor: pointer;
                  font-size: 0.78rem; border: 1px solid transparent; transition: all 0.15s; }
  .history-item:hover { background: var(--bg); border-color: var(--border); }
  .history-item .hi-company { font-weight: 600; color: var(--text); }
  .history-item .hi-meta { color: var(--muted); margin-top: 1px; }

  /* ── Right Panel ── */
  .right-panel { padding: 2rem; overflow-y: auto; }
  .empty-state { display: flex; flex-direction: column; align-items: center; justify-content: center;
                 height: 60vh; color: var(--muted); gap: 0.75rem; text-align: center; }
  .empty-state .icon { font-size: 3rem; opacity: 0.3; }
  .empty-state h2 { font-size: 1rem; font-weight: 600; color: var(--muted); }
  .empty-state p { font-size: 0.85rem; max-width: 300px; line-height: 1.5; }

  /* ── Memo Toolbar ── */
  .memo-toolbar { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;
                  padding-bottom: 1rem; border-bottom: 1px solid var(--border); }
  .memo-company { font-size: 1.1rem; font-weight: 700; flex: 1; }
  .toolbar-btn { background: var(--surface); border: 1px solid var(--border); color: var(--muted);
                 padding: 0.35rem 0.75rem; border-radius: 6px; font-size: 0.78rem; cursor: pointer;
                 transition: all 0.15s; }
  .toolbar-btn:hover { border-color: var(--muted); color: var(--text); }

  /* ── Verdict Badge ── */
  .verdict-banner { display: flex; align-items: center; gap: 1rem; padding: 1rem 1.25rem;
                    border-radius: 8px; margin-bottom: 1.5rem; border: 1px solid; }
  .verdict-banner.invest { background: #0d2b1a; border-color: #1a5c35; }
  .verdict-banner.watch  { background: #2b1f0a; border-color: #5c3d0a; }
  .verdict-banner.pass   { background: #2b0a0a; border-color: #5c1a1a; }
  .verdict-emoji { font-size: 2rem; }
  .verdict-text h2 { font-size: 1.3rem; font-weight: 800; }
  .verdict-text.invest h2 { color: var(--green); }
  .verdict-text.watch  h2 { color: var(--yellow); }
  .verdict-text.pass   h2 { color: var(--red); }
  .verdict-text p { font-size: 0.85rem; color: var(--muted); margin-top: 0.25rem; line-height: 1.4; }
  .verdict-meta { margin-left: auto; text-align: right; font-size: 0.78rem; color: var(--muted); }
  .verdict-meta span { display: block; }

  /* ── Markdown content ── */
  .memo-content { max-width: 820px; }
  .memo-content h1 { font-size: 1.4rem; font-weight: 700; margin: 1.5rem 0 0.5rem;
                     padding-bottom: 0.4rem; border-bottom: 1px solid var(--border); }
  .memo-content h2 { font-size: 1rem; font-weight: 700; color: #79c0ff; margin: 1.5rem 0 0.75rem;
                     text-transform: uppercase; letter-spacing: 0.04em; }
  .memo-content h3 { font-size: 0.95rem; font-weight: 600; color: var(--text); margin: 1rem 0 0.4rem; }
  .memo-content p { font-size: 0.875rem; line-height: 1.7; color: #c9d1d9; margin: 0.5rem 0; }
  .memo-content ul, .memo-content ol { padding-left: 1.25rem; margin: 0.5rem 0; }
  .memo-content li { font-size: 0.875rem; line-height: 1.6; color: #c9d1d9; margin: 0.2rem 0; }
  .memo-content blockquote { border-left: 3px solid var(--blue); padding: 0.5rem 1rem;
                              margin: 0.75rem 0; background: #1c2333; border-radius: 0 6px 6px 0; }
  .memo-content blockquote p { color: #a8b6c8; font-style: italic; }
  .memo-content table { width: 100%; border-collapse: collapse; margin: 0.75rem 0; font-size: 0.82rem; }
  .memo-content th { background: #1c2333; padding: 0.5rem 0.75rem; text-align: left;
                     border: 1px solid var(--border); color: #79c0ff; font-weight: 600; }
  .memo-content td { padding: 0.45rem 0.75rem; border: 1px solid var(--border); color: #c9d1d9; }
  .memo-content tr:nth-child(even) td { background: #0d1117; }
  .memo-content strong { color: var(--text); }
  .memo-content em { color: var(--muted); }
  .memo-content code { background: #1c2333; padding: 1px 6px; border-radius: 4px;
                       font-family: monospace; font-size: 0.82rem; color: #a5d6ff; }
  .memo-content hr { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }

  /* ── Error ── */
  .error-box { background: #2b0a0a; border: 1px solid #5c1a1a; border-radius: 8px;
               padding: 1rem 1.25rem; color: var(--red); font-size: 0.875rem; }
  .rate-limit-box { background: #1a1500; border: 1px solid #4a3800; border-radius: 8px;
                    padding: 1rem 1.25rem; color: #c8a84b; font-size: 0.875rem; }
  .rate-limit-box p { margin: 0.4rem 0 0; color: #a08830; }
</style>
</head>
<body>

<header>
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#388bfd" stroke-width="2">
    <path d="M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2V9M9 21H5a2 2 0 0 1-2-2V9m0 0h18"/>
  </svg>
  <h1>Biotech Diligence Agent</h1>
  <span class="badge-header">VC-Grade Analysis</span>
  <div class="header-right">
    <a href="/methodology" class="header-link">How It Works</a>
  </div>
</header>

<div class="layout">

  <!-- Left panel -->
  <div class="left-panel">
    <div>
      <label>Company or Asset</label>
      <input id="company" type="text" placeholder="e.g. Relay Therapeutics, KarXT…" />
    </div>
    <div>
      <label>Raw inputs <span style="text-transform:none;font-weight:400">(optional)</span></label>
      <textarea id="inputs" placeholder="Paste trial data, press releases, deck bullet points…"></textarea>
    </div>
    <button class="btn btn-primary" id="runBtn" onclick="runDiligence()">Run Full Diligence</button>

    <hr class="divider">

    <!-- Progress -->
    <div id="progress-panel">
      <label>Progress</label>
      <div class="progress-bar-wrap"><div class="progress-bar" id="progressBar"></div></div>
      <div id="progress-label">Starting…</div>
      <div id="module-list"></div>
    </div>

    <hr class="divider" id="history-divider" style="display:none">

    <!-- History -->
    <div id="history-section" style="display:none">
      <label>Recent Runs</label>
      <div id="history-list"></div>
    </div>
  </div>

  <!-- Right panel -->
  <div class="right-panel" id="rightPanel">
    <div class="empty-state">
      <div class="icon">🔬</div>
      <h2>No analysis yet</h2>
      <p>Enter a company name on the left and click <strong>Run Full Diligence</strong> to generate an investment memo.</p>
    </div>
  </div>

</div>

<script src="/static/app.js"></script>
</body>
</html>"""


if __name__ == "__main__":
    try:
        import uvicorn
    except ImportError:
        raise SystemExit("Run: pip3 install uvicorn")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
