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
    from fastapi import FastAPI, HTTPException, UploadFile, File
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, StreamingResponse, Response
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
except ImportError:
    raise SystemExit("Run: pip3 install fastapi uvicorn")

from biotech_diligence.orchestrator import DiligenceOrchestrator, MODULE_MAP
from biotech_diligence.public_markets.orchestrator import (
    PublicMarketsOrchestrator, list_public_runs, load_run as load_public_run,
)

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

class PublicRequest(BaseModel):
    ticker: str
    inputs: Optional[str] = None
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

@app.get("/companies/all")
def list_all_companies():
    """Return all companies with their latest run summary, sorted by date."""
    state_dir = Path(".diligence_state")
    companies = []
    if not state_dir.exists():
        return {"companies": []}
    for d in sorted(state_dir.iterdir()):
        if not d.is_dir():
            continue
        index_path = d / "index.json"
        if not index_path.exists():
            continue
        try:
            index = json.loads(index_path.read_text())
            runs = index.get("runs", [])
            if not runs:
                continue
            latest = runs[-1]
            companies.append({
                "slug": d.name,
                "company": index.get("company", d.name),
                "run_count": len(runs),
                "runs": runs,
                "latest_verdict": latest.get("verdict", ""),
                "latest_date": latest.get("timestamp", "")[:10],
                "latest_one_liner": latest.get("ic_one_liner", ""),
            })
        except Exception:
            continue
    companies.sort(key=lambda x: x["latest_date"], reverse=True)
    return {"companies": companies}

@app.get("/memo/{slug}/{run_id}")
def get_memo(slug: str, run_id: str):
    """Regenerate memo from stored run results."""
    from biotech_diligence.state_manager import StateManager
    from biotech_diligence.memo_generator import MemoGenerator
    # Load by slug directly (bypasses _company_slug re-processing)
    run_path = Path(".diligence_state") / slug / f"{run_id}.json"
    if not run_path.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    run = json.loads(run_path.read_text())
    memo = MemoGenerator().generate(
        company=run["company"], results=run["results"], run_id=run_id
    )
    return {"company": run["company"], "run_id": run_id,
            "timestamp": run.get("timestamp", ""), "memo": memo}

# ------------------------------------------------------------------
# Public Markets routes
# ------------------------------------------------------------------

@app.post("/public/run")
def public_run_stream(req: PublicRequest):
    """SSE endpoint — streams data-fetch progress then 10 module results."""
    ticker = req.ticker.strip().upper()
    inputs = req.inputs or ""
    q = queue.Queue()

    def on_progress(event: dict):
        q.put(("event", event))

    def run():
        try:
            orch = PublicMarketsOrchestrator()
            for evt in orch.analyze_stream(ticker, extra_context=inputs,
                                           progress_cb=on_progress):
                q.put(("event", evt))
            _log_run_tokens(20_000)   # public run uses ~2× tokens
        except Exception as e:
            q.put(("event", {"type": "error", "message": _friendly_error(str(e))}))
        finally:
            q.put(("done", None))

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    def event_stream():
        while True:
            kind, data = q.get()
            if kind == "event":
                yield f"event: public\ndata: {json.dumps(data, default=str)}\n\n"
                if data.get("type") in ("complete", "error"):
                    break
            elif kind == "done":
                yield "event: done\ndata: {}\n\n"
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/public/history/data")
def public_history_data():
    return {"runs": list_public_runs()}


@app.get("/public/memo/{ticker}/{run_id}")
def get_public_memo(ticker: str, run_id: str):
    data = load_public_run(ticker.upper(), run_id)
    if not data:
        raise HTTPException(status_code=404, detail="Run not found")
    return data


@app.get("/demo")
def get_demo():
    """Return the pre-built Karuna Therapeutics demo memo (no API key needed)."""
    import subprocess
    demo_path = Path(__file__).parent / "memos" / "karuna_therapeutics_DEMO.md"
    if not demo_path.exists():
        subprocess.run(["python3", "demo.py"], cwd=Path(__file__).parent)
    return {"company": "Karuna Therapeutics (KarXT \u2014 xanomeline-trospium)", "memo": demo_path.read_text()}


@app.post("/extract")
async def extract_files(files: list[UploadFile] = File(...)):
    """Extract text from uploaded files (PDF, DOCX, TXT). Returns combined text."""
    results = []
    for f in files:
        try:
            content = await f.read()
            name = f.filename or ""
            ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""

            if ext == "pdf":
                from pdfminer.high_level import extract_text_to_fp
                from pdfminer.layout import LAParams
                import io
                out = io.StringIO()
                extract_text_to_fp(io.BytesIO(content), out, laparams=LAParams())
                text = out.getvalue().strip()

            elif ext in ("docx", "doc"):
                import docx, io
                doc = docx.Document(io.BytesIO(content))
                text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())

            elif ext in ("txt", "md", "csv"):
                text = content.decode("utf-8", errors="ignore").strip()

            else:
                text = content.decode("utf-8", errors="ignore").strip()

            if text:
                results.append(f"[{name}]\n{text}")
        except Exception as e:
            results.append(f"[{f.filename} — could not extract: {e}]")

    return {"text": "\n\n---\n\n".join(results), "file_count": len(results)}


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
  .header-right { margin-left: auto; display: flex; gap: 1rem; align-items: center; }
  .header-link { color: var(--muted); font-size: 0.8rem; text-decoration: none; }
  .header-link:hover { color: var(--text); }
  .nav-sep { color: var(--border); font-size: 0.8rem; user-select: none; }

  /* Page layout */
  .page { max-width: 860px; margin: 0 auto; padding: 3rem 2rem 5rem; }
  .page-title { font-size: 1.75rem; font-weight: 700; margin-bottom: 0.5rem; }
  .page-subtitle { color: var(--muted); font-size: 0.95rem; margin-bottom: 3rem; max-width: 600px; line-height: 1.6; }

  /* Timeline */
  .timeline { display: flex; flex-direction: column; }

  /* Each card row: [step column] [content] */
  .module-card { display: flex; gap: 1.25rem; align-items: flex-start; }
  .step-col { display: flex; flex-direction: column; align-items: center;
              align-self: stretch; padding-bottom: 1.25rem; flex-shrink: 0; }
  .module-card:last-child .step-col { padding-bottom: 0; }

  .step-letter { width: 38px; height: 38px; border-radius: 50%;
                 background: #1f3a5f; color: #79c0ff; border: 2px solid #388bfd;
                 font-size: 0.8rem; font-weight: 700;
                 display: flex; align-items: center; justify-content: center;
                 flex-shrink: 0; }
  .step-connector { width: 2px; flex: 1; background: var(--border);
                    min-height: 12px; margin-top: 6px; }
  .module-card:last-child .step-connector { display: none; }

  .module-inner { flex: 1; background: var(--surface); border: 1px solid var(--border);
                  border-radius: 10px; padding: 1.25rem 1.5rem; margin-bottom: 1.25rem; }
  .module-inner:hover { border-color: #388bfd55; }
  .module-name { font-size: 1.05rem; font-weight: 600; color: var(--text); margin-bottom: 0.6rem; }
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
    <a href="/public" class="header-link">Public Markets</a>
    <span class="nav-sep">|</span>
    <a href="/news" class="header-link">News</a>
    <span class="nav-sep">|</span>
    <a href="/history" class="header-link">History</a>
    <span class="nav-sep">|</span>
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
      <div class="step-col"><div class="step-letter">A</div><div class="step-connector"></div></div>
      <div class="module-inner">
        <div class="module-name">Rapid Screening</div>
        <p class="module-desc">A fast 20-minute VC filter. Evaluates science quality, proof of concept, venture scale potential, and existential red flags to decide whether the opportunity warrants deeper diligence. Returns a PASS / SOFT PASS / FAIL verdict before any resources are committed.</p>
        <div class="outputs-label">Key outputs</div>
        <div class="outputs">
          <span class="tag">Verdict</span><span class="tag">Science quality</span><span class="tag">Proof of concept</span><span class="tag">Venture scale</span><span class="tag">Red flags</span><span class="tag">Top concern</span>
        </div>
      </div>
    </div>

    <div class="module-card">
      <div class="step-col"><div class="step-letter">B</div><div class="step-connector"></div></div>
      <div class="module-inner">
        <div class="module-name">Scientific &amp; Mechanistic Validation</div>
        <p class="module-desc">Assesses the soundness of the biological hypothesis and validates the target pathway. Classifies the asset as first-in-class, fast-follow, or me-too, and identifies the critical assumptions that must hold for the science to work.</p>
        <div class="outputs-label">Key outputs</div>
        <div class="outputs">
          <span class="tag">Biological hypothesis</span><span class="tag">Pathway validation</span><span class="tag">Asset classification</span><span class="tag">Critical assumptions</span><span class="tag">Scientific strength score</span><span class="tag">Key insight</span>
        </div>
      </div>
    </div>

    <div class="module-card">
      <div class="step-col"><div class="step-letter">C</div><div class="step-connector"></div></div>
      <div class="module-inner">
        <div class="module-name">Data &amp; Evidence Quality</div>
        <p class="module-desc">Evaluates the rigor and strength of the supporting data package &mdash; trial design, signal quality, biomarker consistency, and reproducibility. Classifies the overall evidence as compelling, emerging, or weak, and flags the most critical data gap.</p>
        <div class="outputs-label">Key outputs</div>
        <div class="outputs">
          <span class="tag">Most advanced data</span><span class="tag">Trial design assessment</span><span class="tag">Signal quality</span><span class="tag">Data red flags</span><span class="tag">Evidence classification</span><span class="tag">Key data gap</span>
        </div>
      </div>
    </div>

    <div class="module-card">
      <div class="step-col"><div class="step-letter">D</div><div class="step-connector"></div></div>
      <div class="module-inner">
        <div class="module-name">Development Pathway &amp; Inflection Points</div>
        <p class="module-desc">Maps the full development roadmap to fundable value inflection points. For each milestone, estimates capital requirements, timeframes, probability of success, and valuation impact &mdash; giving investors a clear picture of the risk/reward at each stage gate.</p>
        <div class="outputs-label">Key outputs</div>
        <div class="outputs">
          <span class="tag">Current stage</span><span class="tag">Inflection points</span><span class="tag">Capital to value event</span><span class="tag">Critical path risks</span><span class="tag">Partnership trigger</span><span class="tag">Development confidence</span>
        </div>
      </div>
    </div>

    <div class="module-card">
      <div class="step-col"><div class="step-letter">E</div><div class="step-connector"></div></div>
      <div class="module-inner">
        <div class="module-name">Competitive Landscape &amp; Positioning</div>
        <p class="module-desc">Identifies direct competitors and mechanism-level competition, then evaluates differentiation across efficacy, safety, convenience, modality, and pricing power. Determines whether the asset has a defensible competitive moat or risks being commoditised.</p>
        <div class="outputs-label">Key outputs</div>
        <div class="outputs">
          <span class="tag">Direct competitors</span><span class="tag">Differentiation dimensions</span><span class="tag">Competitive moat</span><span class="tag">Competitive verdict</span><span class="tag">Biggest threat</span>
        </div>
      </div>
    </div>

    <div class="module-card">
      <div class="step-col"><div class="step-letter">F</div><div class="step-connector"></div></div>
      <div class="module-inner">
        <div class="module-name">Market &amp; Commercial Reality</div>
        <p class="module-desc">Produces a realistic bottom-up market assessment. Sizes the patient population, evaluates standard of care, models pricing and peak revenue scenarios, and identifies the adoption barriers most likely to limit commercial uptake.</p>
        <div class="outputs-label">Key outputs</div>
        <div class="outputs">
          <span class="tag">Patient population</span><span class="tag">Market sizing</span><span class="tag">Standard of care</span><span class="tag">Adoption barriers</span><span class="tag">Commercial risk level</span><span class="tag">Key commercial insight</span>
        </div>
      </div>
    </div>

    <div class="module-card">
      <div class="step-col"><div class="step-letter">G</div><div class="step-connector"></div></div>
      <div class="module-inner">
        <div class="module-name">Team &amp; Execution Risk</div>
        <p class="module-desc">Assesses the quality and experience of the founding team, key executives, and board. Evaluates execution track record, key-man risk, and whether the team has the domain expertise to navigate clinical development and eventual commercialisation.</p>
        <div class="outputs-label">Key outputs</div>
        <div class="outputs">
          <span class="tag">Founding team</span><span class="tag">Management team</span><span class="tag">Board &amp; investors</span><span class="tag">Execution track record</span><span class="tag">Key-man risk</span><span class="tag">Team verdict</span>
        </div>
      </div>
    </div>

    <div class="module-card">
      <div class="step-col"><div class="step-letter">H</div><div class="step-connector"></div></div>
      <div class="module-inner">
        <div class="module-name">IP &amp; Regulatory Risk</div>
        <p class="module-desc">Evaluates patent strength and freedom-to-operate, assesses regulatory pathway clarity (breakthrough designation, orphan status, standard review), and flags any existential IP or regulatory risks that could undermine the investment thesis.</p>
        <div class="outputs-label">Key outputs</div>
        <div class="outputs">
          <span class="tag">Patent position</span><span class="tag">Freedom to operate</span><span class="tag">Regulatory pathway</span><span class="tag">Regulatory risks</span><span class="tag">IP risk level</span><span class="tag">Key IP insight</span>
        </div>
      </div>
    </div>

    <div class="module-card">
      <div class="step-col"><div class="step-letter">I</div><div class="step-connector"></div></div>
      <div class="module-inner">
        <div class="module-name">Risk Decomposition</div>
        <p class="module-desc">Decomposes all investment risk into five categories &mdash; scientific, clinical, regulatory, commercial, and financing &mdash; scoring each with key drivers and mitigants. Synthesises these into an overall Probability of Technical and Commercial Success (PTCS).</p>
        <div class="outputs-label">Key outputs</div>
        <div class="outputs">
          <span class="tag">Scientific risk score</span><span class="tag">Clinical risk score</span><span class="tag">Regulatory risk score</span><span class="tag">Commercial risk score</span><span class="tag">Financing risk score</span><span class="tag">Overall PTCS</span>
        </div>
      </div>
    </div>

    <div class="module-card">
      <div class="step-col"><div class="step-letter">J</div><div class="step-connector"></div></div>
      <div class="module-inner">
        <div class="module-name">Investment Framing</div>
        <p class="module-desc">Frames the investment opportunity for an IC presentation. Articulates bull, base, and bear cases with return multiples, models realistic exit scenarios, and assesses strategic value and fund-returner potential.</p>
        <div class="outputs-label">Key outputs</div>
        <div class="outputs">
          <span class="tag">Bull / base / bear cases</span><span class="tag">Exit scenarios</span><span class="tag">Return multiples</span><span class="tag">Strategic value</span><span class="tag">Fund-returner potential</span><span class="tag">Key investment insight</span>
        </div>
      </div>
    </div>

    <div class="module-card">
      <div class="step-col"><div class="step-letter" style="background:#1a3a1a;color:#3fb950;border-color:#3fb950;">K</div></div>
      <div class="module-inner">
        <div class="module-name">Decision Engine</div>
        <p class="module-desc">Synthesises all prior modules into a decisive investment verdict. Produces an IC-ready one-liner, the top 3 reasons to invest, the top 3 risks, and (where relevant) the conditions under which a PASS could become a WATCH, or a WATCH an INVEST.</p>
        <div class="outputs-label">Key outputs</div>
        <div class="outputs">
          <span class="tag">Verdict (INVEST / WATCH / PASS)</span><span class="tag">Confidence</span><span class="tag">Top 3 reasons</span><span class="tag">Top 3 risks</span><span class="tag">Watch triggers</span><span class="tag">Entry conditions</span><span class="tag">IC one-liner</span>
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
# History page
# ------------------------------------------------------------------

@app.get("/history", response_class=HTMLResponse)
def history_page():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>History \u2014 Biotech Diligence Agent</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #7d8590; --blue: #388bfd;
    --green: #3fb950; --yellow: #d29922; --red: #f85149;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text);
         font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         height: 100vh; display: flex; flex-direction: column; overflow: hidden; }

  /* Header */
  header { background: var(--surface); border-bottom: 1px solid var(--border);
           padding: 1rem 2rem; display: flex; align-items: center; gap: 1rem; flex-shrink: 0; }
  header h1 { font-size: 1.1rem; font-weight: 700; }
  .badge-header { font-size: 0.7rem; background: #1f3a5f; color: #79c0ff;
                  padding: 2px 8px; border-radius: 10px; font-weight: 600; }
  .header-right { margin-left: auto; display: flex; gap: 1rem; align-items: center; }
  .header-link { color: var(--muted); font-size: 0.8rem; text-decoration: none; }
  .header-link:hover { color: var(--text); }
  .nav-sep { color: var(--border); font-size: 0.8rem; user-select: none; }

  /* Two-panel layout */
  .layout { display: flex; flex: 1; overflow: hidden; }
  .left-panel { width: 340px; flex-shrink: 0; border-right: 1px solid var(--border);
                display: flex; flex-direction: column; overflow: hidden; }
  .left-header { padding: 1.25rem 1.25rem 0.75rem; border-bottom: 1px solid var(--border); flex-shrink: 0; }
  .left-header h2 { font-size: 0.85rem; font-weight: 600; color: var(--muted);
                    text-transform: uppercase; letter-spacing: 0.06em; }
  .company-list { overflow-y: auto; flex: 1; padding: 0.5rem 0; }

  /* Company row */
  .company-row { padding: 0.85rem 1.25rem; cursor: pointer; border-bottom: 1px solid #1c2128; }
  .company-row:hover { background: #1c2128; }
  .company-row.active { background: #1c2128; border-left: 3px solid var(--blue); padding-left: calc(1.25rem - 3px); }
  .cr-name { font-size: 0.875rem; font-weight: 600; color: var(--text); margin-bottom: 0.25rem; }
  .cr-meta { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
  .cr-date { font-size: 0.75rem; color: var(--muted); }
  .cr-oneliner { font-size: 0.75rem; color: var(--muted); margin-top: 0.2rem;
                 white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 280px; }

  /* Run sub-list */
  .run-list { background: #0d1117; border-bottom: 1px solid var(--border); display: none; }
  .run-list.open { display: block; }
  .run-row { padding: 0.6rem 1.5rem; cursor: pointer; display: flex; align-items: center; gap: 0.75rem; }
  .run-row:hover { background: #161b22; }
  .run-row.active { background: #161b22; }
  .run-date { font-size: 0.75rem; color: var(--muted); flex-shrink: 0; }
  .run-oneliner { font-size: 0.75rem; color: #8b949e;
                  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

  /* Verdict badges */
  .vbadge { font-size: 0.65rem; font-weight: 700; padding: 1px 7px; border-radius: 10px;
            text-transform: uppercase; white-space: nowrap; flex-shrink: 0; }
  .vbadge.invest { background: #0d2818; color: #3fb950; border: 1px solid #238636; }
  .vbadge.watch  { background: #1a1500; color: #d29922; border: 1px solid #4a3800; }
  .vbadge.pass   { background: #2b0a0a; color: #f85149; border: 1px solid #5c1a1a; }
  .vbadge.na     { background: #1c2128; color: var(--muted); border: 1px solid var(--border); }

  /* Right panel */
  .right-panel { flex: 1; overflow-y: auto; padding: 2rem; }
  .empty-state { display: flex; flex-direction: column; align-items: center;
                 justify-content: center; height: 100%; text-align: center; color: var(--muted); gap: 0.75rem; }
  .empty-state .icon { font-size: 2.5rem; }
  .empty-state h2 { font-size: 1.1rem; font-weight: 600; color: var(--text); }
  .empty-state p { font-size: 0.875rem; max-width: 320px; line-height: 1.6; }

  /* Memo rendering — same as main page */
  .memo-content { max-width: 800px; }
  .memo-toolbar { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.5rem;
                  padding-bottom: 1rem; border-bottom: 1px solid var(--border); }
  .memo-company { font-weight: 700; font-size: 1rem; flex: 1; }
  .toolbar-btn { background: var(--surface); border: 1px solid var(--border); color: var(--text);
                 padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; cursor: pointer; }
  .toolbar-btn:hover { border-color: var(--blue); }
  .memo-content h1 { font-size: 1.4rem; margin: 1.5rem 0 0.75rem; color: var(--text); }
  .memo-content h2 { font-size: 1.1rem; margin: 1.25rem 0 0.5rem; color: var(--text);
                     padding-bottom: 0.4rem; border-bottom: 1px solid var(--border); }
  .memo-content h3 { font-size: 0.95rem; margin: 1rem 0 0.4rem; color: #79c0ff; }
  .memo-content p  { font-size: 0.875rem; line-height: 1.7; color: #c9d1d9; margin-bottom: 0.75rem; }
  .memo-content ul, .memo-content ol { padding-left: 1.25rem; margin-bottom: 0.75rem; }
  .memo-content li { font-size: 0.875rem; line-height: 1.6; color: #c9d1d9; margin-bottom: 0.2rem; }
  .memo-content table { border-collapse: collapse; width: 100%; margin-bottom: 1rem; font-size: 0.8rem; }
  .memo-content th, .memo-content td { border: 1px solid var(--border); padding: 6px 10px; text-align: left; }
  .memo-content th { background: var(--surface); color: var(--text); font-weight: 600; }
  .memo-content td { color: #c9d1d9; }
  .memo-content blockquote { border-left: 3px solid var(--blue); padding: 0.5rem 1rem;
                              margin: 0.75rem 0; background: #1c2128; border-radius: 0 6px 6px 0; }
  .memo-content code { background: #1c2128; padding: 1px 5px; border-radius: 4px;
                       font-family: monospace; font-size: 0.82rem; }
  .memo-content strong { color: var(--text); }
  .verdict-banner { display: flex; align-items: center; gap: 1rem; padding: 1rem 1.25rem;
                    border-radius: 10px; margin-bottom: 1.5rem; border: 1px solid; }
  .verdict-banner.invest { background: #0d2818; border-color: #238636; }
  .verdict-banner.watch  { background: #1a1500; border-color: #4a3800; }
  .verdict-banner.pass   { background: #2b0a0a; border-color: #5c1a1a; }
  .verdict-emoji { font-size: 2rem; }
  .verdict-text h2 { border: none; padding: 0; margin: 0 0 0.2rem; font-size: 1rem; }
  .verdict-text.invest h2 { color: #3fb950; }
  .verdict-text.watch  h2 { color: #d29922; }
  .verdict-text.pass   h2 { color: #f85149; }
  .verdict-text p { margin: 0; font-size: 0.8rem; }
  .verdict-meta { margin-left: auto; display: flex; flex-direction: column; gap: 0.25rem;
                  font-size: 0.75rem; color: var(--muted); text-align: right; }
  .loading-state { color: var(--muted); font-size: 0.875rem; padding: 2rem; text-align: center; }
  .empty-companies { padding: 2rem 1.25rem; text-align: center; color: var(--muted); font-size: 0.875rem; line-height: 1.6; }
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
    <span class="nav-sep">|</span>
    <a href="/public" class="header-link">Public Markets</a>
    <span class="nav-sep">|</span>
    <a href="/news" class="header-link">News</a>
    <span class="nav-sep">|</span>
    <a href="/methodology" class="header-link">How It Works</a>
  </div>
</header>

<div class="layout">
  <div class="left-panel">
    <div class="left-header"><h2>Past Diligences</h2></div>
    <div class="company-list" id="companyList">
      <div class="empty-companies">Loading&hellip;</div>
    </div>
  </div>
  <div class="right-panel" id="rightPanel">
    <div class="empty-state">
      <div class="icon">&#x1F4CB;</div>
      <h2>Select a diligence</h2>
      <p>Choose a company from the left to view the investment memo.</p>
    </div>
  </div>
</div>

<script src="/static/app.js"></script>
<script>
const VMAP = { INVEST: "invest", WATCH: "watch", PASS: "pass" };

function verdictBadge(v) {
  const cls = VMAP[v] || "na";
  return '<span class="vbadge ' + cls + '">' + (v || "N/A") + '</span>';
}

async function loadCompanies() {
  const list = document.getElementById("companyList");
  try {
    const res = await fetch("/companies/all");
    const data = await res.json();
    const companies = data.companies || [];
    if (!companies.length) {
      list.innerHTML = '<div class="empty-companies">No diligences run yet.<br>Head back to the agent to get started.</div>';
      return;
    }
    list.innerHTML = companies.map(function(c) {
      const runsHtml = c.runs.slice().reverse().map(function(r) {
        return `<div class="run-row" onclick="loadMemo('${c.slug}','${r.run_id}','${escHtml(c.company)}',this)">` +
          verdictBadge(r.verdict) +
          '<span class="run-date">' + (r.timestamp||"").slice(0,10) + '</span>' +
          '<span class="run-oneliner">' + escHtml((r.ic_one_liner||"").slice(0,60)) + '</span>' +
          '</div>';
      }).join("");
      return '<div class="company-row" onclick="toggleCompany(this)">' +
        '<div class="cr-name">' + escHtml(c.company) + '</div>' +
        '<div class="cr-meta">' + verdictBadge(c.latest_verdict) +
        '<span class="cr-date">' + c.latest_date + '</span>' +
        '<span style="color:var(--muted);font-size:0.72rem">' + c.run_count + ' run' + (c.run_count!==1?"s":"") + '</span></div>' +
        '<div class="cr-oneliner">' + escHtml((c.latest_one_liner||"").slice(0,80)) + '</div>' +
        '</div>' +
        '<div class="run-list">' + runsHtml + '</div>';
    }).join("");
  } catch(e) {
    list.innerHTML = '<div class="empty-companies">Failed to load history.</div>';
  }
}

function escHtml(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

function toggleCompany(row) {
  const runList = row.nextElementSibling;
  const isOpen = runList.classList.contains("open");
  // Close all
  document.querySelectorAll(".run-list.open").forEach(function(el){ el.classList.remove("open"); });
  document.querySelectorAll(".company-row.active").forEach(function(el){ el.classList.remove("active"); });
  if (!isOpen) {
    runList.classList.add("open");
    row.classList.add("active");
    // Auto-load latest run
    const firstRun = runList.querySelector(".run-row");
    if (firstRun) firstRun.click();
  }
}

let currentMemoText = "";
let currentMemoCompany = "";

async function loadMemo(slug, runId, company, runRow) {
  document.querySelectorAll(".run-row.active").forEach(function(el){ el.classList.remove("active"); });
  runRow.classList.add("active");
  currentMemoCompany = company;
  document.getElementById("rightPanel").innerHTML = '<div class="loading-state">Loading memo&hellip;</div>';
  try {
    const res = await fetch("/memo/" + encodeURIComponent(slug) + "/" + encodeURIComponent(runId));
    if (!res.ok) throw new Error("Not found");
    const data = await res.json();
    currentMemoText = data.memo;
    renderMemo(data.company, data.memo);
  } catch(e) {
    document.getElementById("rightPanel").innerHTML = '<div class="loading-state">Failed to load memo.</div>';
  }
}

function copyMemo() {
  navigator.clipboard.writeText(currentMemoText).then(function() {
    var btn = event.target;
    var orig = btn.textContent;
    btn.textContent = "Copied!";
    setTimeout(function(){ btn.textContent = orig; }, 1500);
  });
}

function downloadMemo() {
  var slug = currentMemoCompany.toLowerCase().replace(/\\s+/g, "_");
  var blob = new Blob([currentMemoText], { type: "text/markdown" });
  var a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = slug + "_diligence.md";
  a.click();
}

loadCompanies();
</script>
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
  .header-right { margin-left: auto; display: flex; gap: 1rem; align-items: center; }
  .header-link { color: var(--muted); font-size: 0.8rem; text-decoration: none; }
  .header-link:hover { color: var(--text); }
  .nav-sep { color: var(--border); font-size: 0.8rem; user-select: none; }

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

  /* ── Upload zone ── */
  .upload-zone { border: 1px dashed var(--border); border-radius: 8px; padding: 0.75rem 1rem;
                 display: flex; align-items: center; gap: 0.6rem; cursor: pointer;
                 color: var(--muted); font-size: 0.8rem; transition: border-color 0.15s; }
  .upload-zone:hover { border-color: var(--blue); color: var(--text); }
  .upload-zone.drag-over { border-color: var(--blue); background: #1f3a5f22; }
  .upload-hint { font-size: 0.72rem; color: var(--muted); margin-left: auto; }
  #fileList { display: flex; flex-direction: column; gap: 0.3rem; margin-top: 0.4rem; }
  .file-chip { display: flex; align-items: center; gap: 0.5rem; background: #1c2128;
               border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px;
               font-size: 0.75rem; color: var(--text); }
  .file-chip .fname { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .file-chip .fsize { color: var(--muted); flex-shrink: 0; }
  .file-chip .fremove { cursor: pointer; color: var(--muted); flex-shrink: 0; font-size: 1rem;
                        line-height: 1; padding: 0 2px; }
  .file-chip .fremove:hover { color: var(--red); }

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
    <a href="/public" class="header-link">Public Markets</a>
    <span class="nav-sep">|</span>
    <a href="/news" class="header-link">News</a>
    <span class="nav-sep">|</span>
    <a href="/history" class="header-link">History</a>
    <span class="nav-sep">|</span>
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
      <textarea id="inputs" placeholder="Paste any relevant context: clinical trial readouts, pipeline overview, mechanism of action, IP / patent summary, leadership bios, competitive positioning, press releases, financial highlights, meeting notes…"></textarea>
    </div>
    <div>
      <label>Attachments <span style="text-transform:none;font-weight:400">(optional)</span></label>
      <div class="upload-zone" id="uploadZone" onclick="document.getElementById('fileInput').click()">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
        </svg>
        <span>Click to attach files</span>
        <span class="upload-hint">Pitch decks, data rooms, investor materials, clinical summaries &mdash; PDF, DOCX, TXT</span>
      </div>
      <input type="file" id="fileInput" multiple accept=".pdf,.docx,.doc,.txt,.md,.csv" style="display:none" onchange="handleFiles(this.files)">
      <div id="fileList"></div>
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


# ------------------------------------------------------------------
# Biotech fundraising news feed
# ------------------------------------------------------------------

import urllib.request as _urllib_req
import xml.etree.ElementTree as _ET
import re as _re
import time as _time

_NEWS_CACHE     = Path(".diligence_state/news_cache.json")
_NEWS_LOG       = Path(".diligence_state/fundraise_log.json")
_NEWS_TTL       = 7200   # 2 hours between RSS refreshes
_NEWS_HISTORY   = 365 * 86400  # keep up to 12 months in the log

_RSS_FEEDS = [
    ("Fierce Biotech",  "https://www.fiercebiotech.com/rss/xml"),
    ("BioPharma Dive",  "https://www.biopharmadive.com/feeds/news/"),
    ("STAT News",       "https://www.statnews.com/feed/"),
    ("BioSpace",        "https://www.biospace.com/rss/news.xml"),
]

_RAISE_RE = _re.compile(
    r'\braises?|secures?|closes?\s+(?:\$|funding|\w+\s+round)'
    r'|\$\d+[\.,]?\d*\s*[mb]illion|\bseries\s+[a-e]\b'
    r'|seed\s+round|funding\s+round|\binitial\s+public\b',
    _re.IGNORECASE
)


def _fetch_rss(url: str) -> list:
    try:
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = _urllib_req.Request(url, headers={"User-Agent": "BiotechDigest/1.0"})
        with _urllib_req.urlopen(req, timeout=8, context=ctx) as r:
            data = r.read()
        root = _ET.fromstring(data)
        def _el_text(parent, tag):
            el = parent.find(tag)
            if el is None:
                return ""
            return "".join(el.itertext()).strip()

        items = []
        for el in root.findall(".//item"):
            items.append({
                "title":   _el_text(el, "title"),
                "link":    _el_text(el, "link"),
                "summary": _el_text(el, "description"),
                "date":    _el_text(el, "pubDate"),
            })
        return items
    except Exception:
        return []


# Regex patterns for zero-LLM news extraction
_AMT_RE = _re.compile(
    r'\$\s*(\d+(?:\.\d+)?)\s*(billion|million|bn|b\b|m\b)',
    _re.IGNORECASE,
)
_ROUND_PAT = _re.compile(
    r'\b(series\s+[a-e](?:\+)?|seed(?:\s+round)?|ipo|initial\s+public\s+offer\w*'
    r'|bridge(?:\s+round)?|convertible(?:\s+note)?|strategic(?:\s+round)?)\b',
    _re.IGNORECASE,
)
# Verbs that signal a company is *actively* raising capital right now
_RAISE_VERB = _re.compile(
    r'\b(raises?\s+\$|banks?\s+\$|nets?\s+\$|lands?\s+\$|gets?\s+\$'
    r'|secures?\s+\$|secures?\s+(?:funding|investment|round|capital)'
    r'|closes?\s+\$|closes?\s+(?:series|seed|bridge|round|funding)'
    r'|completes?\s+\$|completes?\s+(?:series|seed|bridge|round)'
    r'|announces?\s+\$|raises?\s+(?:series|seed|bridge)\b)',
    _re.IGNORECASE,
)
# Phrases that mark a historical/retrospective reference — not a current raise
_HISTORICAL = _re.compile(
    r'\bafter\s+(its|their|the)\b|\bfollowing\s+(its|their|the)\b|\bbuilding\s+on\b'
    r'|\blast\s+year\b|\bin\s+20(?:2[0-4])\b|\bsince\s+its\b'
    r'|\bwhat\s+\w+\s+is\s+build\w+\b|\bpost[- ]\$'
    r'|\bhad\s+raised\b|\bpreviously\s+raised\b|\bwhich\s+raised\b'
    r'|\bwho\s+raised\b|\bthat\s+raised\b|\bonce\s+raised\b'
    r'|\bcompleted\s+(?:a\s+)?\$|\bclosed\s+(?:a\s+)?\$',
    _re.IGNORECASE,
)


def _pub_year(date_str: str) -> int:
    """Extract 4-digit year from an RSS pubDate string (e.g. 'Mon, 26 May 2026 …')."""
    if not date_str:
        return 0
    m = _re.search(r'\b(20\d{2})\b', date_str)
    return int(m.group(1)) if m else 0


def _is_stale_reference(title: str, snippet: str, pub_date_str: str) -> bool:
    """
    Return True when the article is referencing a *past* fundraise rather than
    announcing a new one.

    Strategy:
    1. Check known retrospective phrases in title + first 300 chars of snippet.
    2. Year-staleness check: if the article explicitly associates a dollar amount
       or round-type with a year older than the article's own publication year,
       the article is a retrospective mention, not a fresh announcement.

    Example caught: March-2026 article that says "Xaira Therapeutics, which raised
    $1.2B in 2024, begins Phase I trials" → year 2024 < pub_year 2026 → stale.
    """
    full = title + " " + snippet[:300]

    # Hard phrase patterns
    if _HISTORICAL.search(full):
        return True

    # Year-staleness check
    pub_yr = _pub_year(pub_date_str)
    if pub_yr >= 2024:
        # Find every "in YYYY" or "from YYYY" or "during YYYY" near the fundraise
        for ym in _re.finditer(r'\b(?:in|from|during|back\s+in)\s+(20\d{2})\b', full, _re.IGNORECASE):
            y = int(ym.group(1))
            if y >= pub_yr:
                continue  # same year — could be a real announcement
            # Check if a dollar amount or round label appears within ±120 chars
            ctx = full[max(0, ym.start() - 120): ym.end() + 120]
            if _AMT_RE.search(ctx) or _ROUND_PAT.search(ctx):
                return True

    return False


def _parse_amount(text: str) -> str:
    m = _AMT_RE.search(text)
    if not m:
        return "Undisclosed"
    num = float(m.group(1))
    unit = m.group(2).lower()
    if unit in ("billion", "bn", "b"):
        return f"${num:g}B"
    return f"${int(num) if num == int(num) else num}M"


def _parse_round(text: str) -> str:
    m = _ROUND_PAT.search(text)
    if not m:
        return "Unknown"
    r = m.group(1).lower()
    if r.startswith("series"):
        letter = _re.sub(r"series\s*", "", r).strip().upper().rstrip("+")
        return f"Series {letter}" if letter in "ABCDE" else "Series E+"
    if "seed" in r:
        return "Seed"
    if "ipo" in r or "initial public" in r:
        return "IPO"
    if "bridge" in r:
        return "Bridge"
    if "convert" in r:
        return "Convertible"
    if "strategic" in r:
        return "Strategic"
    return "Unknown"


def _parse_company(title: str) -> str:
    # Strip "SOURCE: " prefixes like "STAT+: " or "EXCLUSIVE: "
    title = _re.sub(r'^[A-Z][A-Za-z0-9+]*:\s*', '', title).strip()
    m = _RAISE_VERB.search(title)
    if not m:
        return "Unknown"
    before = title[:m.start()].strip().rstrip(",;")
    words = before.split()
    # Take up to 5 words — enough for "Relay Therapeutics" or "Karuna Therapeutics Inc"
    return " ".join(words[:5]) if words else "Unknown"


def _extract_fundraises(articles: list) -> list:
    """Pure regex extraction — zero LLM tokens consumed."""
    results = []
    for a in articles:
        title   = a.get("title", "")
        snippet = _re.sub(r'<[^>]+>', '', a.get("summary", ""))
        pub_date = a.get("date", "")
        full_text = title + " " + snippet[:300]

        # Must have an active raise verb in the headline
        if not _RAISE_VERB.search(title):
            continue

        # Reject retrospective references and stale year mismatches
        if _is_stale_reference(title, snippet, pub_date):
            continue

        company = _parse_company(title)
        if company == "Unknown":
            continue

        results.append({
            "company":    company,
            "hq":         "Unknown",
            "amount_usd": _parse_amount(full_text),
            "round_type": _parse_round(full_text),
            "date":       a.get("date", "")[:16],
            "title":      title,
            "url":        a.get("link", ""),
            "source":     a.get("source", ""),
        })
    return results


def _load_log() -> dict:
    """Load the persistent fundraise log. Returns {url_key: fundraise_dict}."""
    if not _NEWS_LOG.exists():
        return {}
    try:
        entries = json.loads(_NEWS_LOG.read_text())
        return {e["_key"]: e for e in entries if isinstance(e, dict) and "_key" in e}
    except Exception:
        return {}


def _save_log(log: dict):
    """Persist log, pruning entries older than 12 months."""
    cutoff = _time.time() - _NEWS_HISTORY
    entries = [e for e in log.values() if e.get("_saved_at", 0) >= cutoff]
    _NEWS_LOG.parent.mkdir(exist_ok=True)
    try:
        _NEWS_LOG.write_text(json.dumps(entries))
    except Exception:
        pass


def _entry_key(f: dict) -> str:
    """Stable dedup key: prefer URL, fall back to normalised title."""
    if f.get("url"):
        return f["url"]
    return _re.sub(r'\W+', '', f.get("title", "").lower())[:60]


def _get_news(force: bool = False) -> dict:
    now = _time.time()

    # Return cached result if still fresh (skip RSS fetch, still return full log)
    if not force and _NEWS_CACHE.exists():
        try:
            cached = json.loads(_NEWS_CACHE.read_text())
            if now - cached.get("fetched_at", 0) < _NEWS_TTL:
                return cached
        except Exception:
            pass

    # Fetch RSS and extract new fundraises
    all_articles: list = []
    for source_name, url in _RSS_FEEDS:
        items = _fetch_rss(url)
        for it in items:
            it["source"] = source_name
        matching = [a for a in items if _RAISE_RE.search(
            a.get("title", "") + " " + a.get("summary", "")[:200])]
        all_articles.extend(matching[:12])

    seen: set = set()
    deduped = []
    for a in all_articles:
        key = _re.sub(r'\W+', '', a["title"].lower())[:50]
        if key not in seen:
            seen.add(key)
            deduped.append(a)

    fresh = _extract_fundraises(deduped[:30])

    # Merge into persistent log
    log = _load_log()
    for f in fresh:
        k = _entry_key(f)
        if k not in log:
            f["_key"] = k
            f["_saved_at"] = now
            log[k] = f
    _save_log(log)

    # Return all log entries sorted newest-first
    all_entries = sorted(log.values(), key=lambda x: x.get("_saved_at", 0), reverse=True)
    # Strip internal fields before returning
    fundraises = [{k: v for k, v in e.items() if not k.startswith("_")} for e in all_entries]

    result = {"fundraises": fundraises, "fetched_at": now}
    _NEWS_CACHE.parent.mkdir(exist_ok=True)
    try:
        _NEWS_CACHE.write_text(json.dumps(result))
    except Exception:
        pass
    return result


@app.get("/news/data")
def news_data(refresh: bool = False):
    return _get_news(force=refresh)


@app.get("/news", response_class=HTMLResponse)
def news_page():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fundraising News \u2014 Biotech Diligence Agent</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #7d8590; --blue: #388bfd;
    --green: #3fb950; --yellow: #d29922; --red: #f85149;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text);
         font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }

  header { background: var(--surface); border-bottom: 1px solid var(--border);
           padding: 1rem 2rem; display: flex; align-items: center; gap: 1rem; }
  header h1 { font-size: 1.1rem; font-weight: 700; }
  .badge-header { font-size: 0.7rem; background: #1f3a5f; color: #79c0ff;
                  padding: 2px 8px; border-radius: 10px; font-weight: 600; }
  .header-right { margin-left: auto; display: flex; gap: 1rem; align-items: center; }
  .header-link { color: var(--muted); font-size: 0.8rem; text-decoration: none; }
  .header-link:hover { color: var(--text); }
  .nav-sep { color: var(--border); font-size: 0.8rem; user-select: none; }

  .page { max-width: 1060px; margin: 0 auto; padding: 2.5rem 2rem 5rem; }
  .page-header { display: flex; align-items: baseline; gap: 1rem; margin-bottom: 0.4rem; flex-wrap: wrap; }
  .page-title { font-size: 1.6rem; font-weight: 700; }
  .page-meta { color: var(--muted); font-size: 0.8rem; }
  .page-subtitle { color: var(--muted); font-size: 0.9rem; margin-bottom: 1.5rem; }

  .toolbar { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.5rem; }
  .btn-refresh { background: var(--surface); border: 1px solid var(--border); color: var(--text);
                 padding: 0.4rem 0.9rem; border-radius: 6px; font-size: 0.8rem; cursor: pointer; }
  .btn-refresh:hover { border-color: var(--blue); color: var(--blue); }
  .btn-refresh:disabled { opacity: 0.5; cursor: default; }

  /* Table */
  .news-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  .news-table th { text-align: left; padding: 0.5rem 0.75rem; color: var(--muted);
                   font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
                   letter-spacing: 0.05em; border-bottom: 1px solid var(--border); }
  .news-table td { padding: 0.7rem 0.75rem; border-bottom: 1px solid #21262d; vertical-align: middle; }
  .news-table tr:last-child td { border-bottom: none; }
  .news-table tr:hover td { background: #161b22; }

  .company-cell { font-weight: 600; color: var(--text); }
  .company-cell a { color: var(--text); text-decoration: none; }
  .company-cell a:hover { color: var(--blue); text-decoration: underline; }
  .hq-cell { color: var(--muted); }

  .round-badge { display: inline-block; padding: 2px 7px; border-radius: 10px;
                 font-size: 0.72rem; font-weight: 600; white-space: nowrap; }
  .round-seed    { background: #1a3a1a; color: #3fb950; }
  .round-a       { background: #1f3a5f; color: #79c0ff; }
  .round-b       { background: #1f3a5f; color: #79c0ff; }
  .round-c       { background: #2d1f5f; color: #c9b1ff; }
  .round-d       { background: #3a1f2d; color: #ff7b93; }
  .round-ipo     { background: #2d2400; color: #d29922; }
  .round-other   { background: #1c2128; color: var(--muted); }

  .amount-cell { font-weight: 600; color: var(--green); white-space: nowrap; }
  .date-cell   { color: var(--muted); white-space: nowrap; }
  .source-cell { font-size: 0.75rem; }
  .source-link { color: var(--blue); text-decoration: none; }
  .source-link:hover { text-decoration: underline; }

  .empty-state, .loading-state { text-align: center; padding: 4rem 2rem; color: var(--muted); }
  .empty-state .icon, .loading-state .icon { font-size: 2.5rem; margin-bottom: 1rem; }
  .empty-state h2, .loading-state h2 { font-size: 1.1rem; color: var(--text); margin-bottom: 0.5rem; }

  .spinner { display: inline-block; width: 28px; height: 28px; border: 3px solid var(--border);
             border-top-color: var(--blue); border-radius: 50%;
             animation: spin 0.8s linear infinite; margin-bottom: 1rem; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<header>
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#388bfd" stroke-width="2">
    <path d="M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2V9M9 21H5a2 2 0 0 1-2-2V9m0 0h18"/>
  </svg>
  <h1>Biotech Diligence Agent</h1>
  <span class="badge-header">VC-Grade Analysis</span>
  <div class="header-right">
    <a href="/" class="header-link">&larr; Back to Agent</a>
    <span class="nav-sep">|</span>
    <a href="/public" class="header-link">Public Markets</a>
    <span class="nav-sep">|</span>
    <a href="/history" class="header-link">History</a>
    <span class="nav-sep">|</span>
    <a href="/methodology" class="header-link">How It Works</a>
  </div>
</header>

<div class="page">
  <div class="page-header">
    <h2 class="page-title">Biotech Fundraising News</h2>
    <span class="page-meta" id="lastUpdated"></span>
  </div>
  <p class="page-subtitle">Recent capital raises across biotech and pharma — sourced from industry news feeds.</p>

  <div class="toolbar">
    <button class="btn-refresh" id="refreshBtn" onclick="loadNews(true)">&#8635; Refresh</button>
    <span id="statusMsg" style="color:var(--muted);font-size:0.8rem;"></span>
  </div>

  <div id="tableContainer">
    <div class="loading-state">
      <div class="spinner"></div>
      <h2>Loading fundraising news&hellip;</h2>
      <p>Fetching and analysing recent raises.</p>
    </div>
  </div>
</div>

<script>
const ROUND_CLASS = {
  "Seed": "round-seed",
  "Series A": "round-a",
  "Series B": "round-b",
  "Series C": "round-c",
  "Series D": "round-d",
  "Series E+": "round-d",
  "IPO": "round-ipo",
};

function roundClass(r) {
  return ROUND_CLASS[r] || "round-other";
}

function formatDate(raw) {
  if (!raw) return "";
  const d = new Date(raw);
  if (isNaN(d.getTime())) return raw.slice(0, 10);
  return d.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
}

function timeSince(ts) {
  const mins = Math.round((Date.now() / 1000 - ts) / 60);
  if (mins < 2) return "just now";
  if (mins < 60) return mins + " min ago";
  const hrs = Math.round(mins / 60);
  return hrs + "h ago";
}

function esc(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

async function loadNews(refresh) {
  const btn = document.getElementById("refreshBtn");
  const status = document.getElementById("statusMsg");
  btn.disabled = true;
  status.textContent = refresh ? "Refreshing\u2026" : "Loading\u2026";

  try {
    const url = "/news/data" + (refresh ? "?refresh=true" : "");
    const res = await fetch(url);
    const data = await res.json();
    render(data);
    const el = document.getElementById("lastUpdated");
    if (data.fetched_at) el.textContent = "Updated " + timeSince(data.fetched_at);
    status.textContent = "";
  } catch(e) {
    document.getElementById("tableContainer").innerHTML =
      '<div class="empty-state"><div class="icon">\u26a0\ufe0f</div><h2>Could not load news</h2><p>' + esc(String(e)) + '</p></div>';
    status.textContent = "";
  }
  btn.disabled = false;
}

function render(data) {
  const rows = (data.fundraises || []);
  if (!rows.length) {
    document.getElementById("tableContainer").innerHTML =
      '<div class="empty-state"><div class="icon">&#x1F4F0;</div>' +
      '<h2>No fundraising news found</h2>' +
      '<p>Try refreshing \u2014 news feeds update throughout the day.</p></div>';
    return;
  }

  let html = '<table class="news-table"><thead><tr>' +
    '<th>Company</th><th>HQ</th><th>Round</th><th>Amount</th><th>Date</th><th>Source</th>' +
    '</tr></thead><tbody>';

  for (const r of rows) {
    const cls = roundClass(r.round_type);
    const companyCell = r.url
      ? '<a href="' + esc(r.url) + '" target="_blank" rel="noopener">' + esc(r.company) + '</a>'
      : esc(r.company);
    html += '<tr>' +
      '<td class="company-cell">' + companyCell + '</td>' +
      '<td class="hq-cell">' + esc(r.hq) + '</td>' +
      '<td><span class="round-badge ' + cls + '">' + esc(r.round_type) + '</span></td>' +
      '<td class="amount-cell">' + esc(r.amount_usd) + '</td>' +
      '<td class="date-cell">' + formatDate(r.date) + '</td>' +
      '<td class="source-cell">' + (r.url
        ? '<a class="source-link" href="' + esc(r.url) + '" target="_blank" rel="noopener">' + esc(r.source) + '</a>'
        : esc(r.source)) + '</td>' +
      '</tr>';
  }
  html += '</tbody></table>';
  document.getElementById("tableContainer").innerHTML = html;
}

loadNews(false);
</script>
</body>
</html>"""


@app.get("/public", response_class=HTMLResponse)
def public_page():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Public Markets Diligence — Biotech Diligence Agent</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #7d8590; --blue: #388bfd;
    --green: #3fb950; --yellow: #d29922; --red: #f85149;
    --purple: #8b5cf6;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text);
         font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         height: 100vh; display: flex; flex-direction: column; }

  header { background: var(--surface); border-bottom: 1px solid var(--border);
           padding: 1rem 2rem; display: flex; align-items: center; gap: 1rem;
           flex-shrink: 0; }
  header h1 { font-size: 1.1rem; font-weight: 700; }
  .badge-header { font-size: 0.7rem; background: #2d1b5f; color: #b794f6;
                  padding: 2px 8px; border-radius: 10px; font-weight: 600; }
  .header-right { margin-left: auto; display: flex; gap: 1rem; align-items: center; }
  .header-link { color: var(--muted); font-size: 0.8rem; text-decoration: none; }
  .header-link:hover { color: var(--text); }
  .nav-sep { color: var(--border); font-size: 0.8rem; user-select: none; }

  .layout { display: flex; flex: 1; overflow: hidden; }

  /* ── Left panel ── */
  .left-panel { width: 320px; flex-shrink: 0; border-right: 1px solid var(--border);
                padding: 1.5rem; overflow-y: auto; display: flex;
                flex-direction: column; gap: 1.25rem; }
  label { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.07em;
          text-transform: uppercase; color: var(--muted); display: block;
          margin-bottom: 0.4rem; }
  input, textarea { width: 100%; background: var(--bg); border: 1px solid var(--border);
                    border-radius: 6px; color: var(--text); padding: 0.55rem 0.75rem;
                    font-size: 0.875rem; }
  input:focus, textarea:focus { outline: none; border-color: var(--blue); }
  textarea { min-height: 100px; resize: vertical; }
  .btn { width: 100%; padding: 0.65rem; border-radius: 8px; border: none;
         font-size: 0.9rem; font-weight: 600; cursor: pointer; }
  .btn-primary { background: var(--purple); color: #fff; }
  .btn-primary:hover { background: #7c3aed; }
  .btn-primary:disabled { opacity: 0.5; cursor: default; }
  .divider { border: none; border-top: 1px solid var(--border); }

  /* ── Data source badges ── */
  .sources-grid { display: flex; flex-direction: column; gap: 0.4rem; }
  .source-row { display: flex; align-items: center; gap: 0.5rem;
                font-size: 0.75rem; color: var(--muted); }
  .source-dot { width: 8px; height: 8px; border-radius: 50%;
                background: var(--border); flex-shrink: 0; }
  .source-dot.fetching { background: var(--yellow); }
  .source-dot.ok       { background: var(--green); }
  .source-dot.error    { background: var(--red); }

  /* ── Module progress (left panel) ── */
  .progress-bar-wrap { background: #21262d; border-radius: 4px; height: 4px;
                       overflow: hidden; margin-bottom: 0.75rem; }
  .progress-bar { height: 100%; background: var(--purple); transition: width 0.4s; width: 0%; }
  .step-list { display: flex; flex-direction: column; gap: 0.35rem; }
  .step-item { display: flex; align-items: center; gap: 0.5rem;
               font-size: 0.75rem; color: var(--muted); }
  .step-item.active { color: var(--text); }
  .step-item.done   { color: var(--green); }
  .step-icon { width: 16px; text-align: center; flex-shrink: 0; }

  /* ── Right panel ── */
  .right-panel { flex: 1; overflow-y: auto; padding: 2rem; }
  .empty-state { height: 100%; display: flex; flex-direction: column;
                 align-items: center; justify-content: center;
                 color: var(--muted); gap: 0.75rem; }
  .empty-icon { font-size: 2.5rem; }
  .empty-state h2 { font-size: 1.1rem; color: var(--text); }
  .empty-state p  { font-size: 0.875rem; text-align: center; max-width: 320px; }

  /* ── Result cards ── */
  .result-card { background: var(--surface); border: 1px solid var(--border);
                 border-radius: 10px; padding: 1.25rem 1.5rem; margin-bottom: 1rem; }
  .result-card h3 { font-size: 0.85rem; font-weight: 700; text-transform: uppercase;
                    letter-spacing: 0.06em; color: var(--purple); margin-bottom: 0.75rem; }
  .confidence-tag { display: inline-block; font-size: 0.7rem; font-weight: 600;
                    padding: 2px 8px; border-radius: 10px; margin-left: 0.5rem; }
  .conf-High   { background: #0d2b1a; color: var(--green); }
  .conf-Medium { background: #1a1500; color: var(--yellow); }
  .conf-Low    { background: #2b0a0a; color: var(--red); }

  /* ── Executive summary ── */
  .exec-summary { background: #0d1f38; border: 1px solid #1f3a5f;
                  border-radius: 10px; padding: 1.25rem 1.5rem; margin-bottom: 1.25rem; }
  .exec-summary h2 { font-size: 1rem; font-weight: 700; color: #79c0ff; margin-bottom: 0.6rem; }
  .exec-summary p { font-size: 0.875rem; line-height: 1.7; color: #c9d1d9; }

  /* ── rNPV table ── */
  .rnpv-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; margin-top: 0.75rem; }
  .rnpv-table th { text-align: left; padding: 0.4rem 0.6rem; color: var(--muted);
                   font-size: 0.72rem; letter-spacing: 0.05em; text-transform: uppercase;
                   border-bottom: 1px solid var(--border); }
  .rnpv-table td { padding: 0.5rem 0.6rem; border-bottom: 1px solid #21262d; }
  .rnpv-table tr:last-child td { border-bottom: none; }
  .val-green { color: var(--green); font-weight: 600; }
  .val-yellow { color: var(--yellow); }
  .val-red    { color: var(--red); }

  /* ── Investment stance ── */
  .stance { display: inline-flex; align-items: center; gap: 0.4rem;
            padding: 0.4rem 1rem; border-radius: 8px; font-weight: 700;
            font-size: 0.9rem; margin-bottom: 1rem; }
  .stance-Constructive { background: #0d2b1a; color: #3fb950; }
  .stance-Neutral      { background: #1a1a1a; color: #c9d1d9; }
  .stance-Cautious     { background: #1a1500; color: #d29922; }
  .stance-Avoid        { background: #2b0a0a; color: #f85149; }

  /* ── Data fields ── */
  .field-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem 1.25rem;
                margin-top: 0.5rem; }
  .field { font-size: 0.82rem; }
  .field-label { color: var(--muted); font-size: 0.72rem; margin-bottom: 1px; }
  .field-val   { color: var(--text); font-weight: 500; }
  .field-null  { color: var(--muted); font-style: italic; }
  .flag-item { font-size: 0.8rem; color: var(--yellow);
               padding: 0.25rem 0; border-bottom: 1px solid #21262d; }
  .flag-item:last-child { border-bottom: none; }
  .sources-note { font-size: 0.72rem; color: var(--muted); margin-top: 0.75rem;
                  padding-top: 0.6rem; border-top: 1px solid var(--border); }

  /* ── Error / rate-limit ── */
  .error-box { background: #2b0a0a; border: 1px solid #5c1a1a; border-radius: 8px;
               padding: 1rem 1.25rem; color: var(--red); font-size: 0.875rem; }
</style>
</head>
<body>
<header>
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#8b5cf6" stroke-width="2">
    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
  </svg>
  <h1>Biotech Diligence Agent</h1>
  <span class="badge-header">Public Markets</span>
  <div class="header-right">
    <a href="/" class="header-link">&larr; VC Agent</a>
    <span class="nav-sep">|</span>
    <a href="/news" class="header-link">News</a>
    <span class="nav-sep">|</span>
    <a href="/history" class="header-link">History</a>
    <span class="nav-sep">|</span>
    <a href="/methodology" class="header-link">How It Works</a>
  </div>
</header>

<div class="layout">

  <!-- Left panel -->
  <div class="left-panel">
    <div>
      <label>Ticker Symbol</label>
      <input id="ticker" type="text" placeholder="e.g. MRNA, RXRX, REGN&hellip;"
             oninput="this.value = this.value.toUpperCase()"
             onkeydown="if(event.key==='Enter') runPublic()" />
    </div>
    <div>
      <label>Additional context <span style="text-transform:none;font-weight:400">(optional)</span></label>
      <textarea id="pubInputs" placeholder="Paste analyst reports, PDUFA dates, conference call notes, competitive intelligence, or any context not in public filings&hellip;"></textarea>
    </div>
    <button class="btn btn-primary" id="runBtn" onclick="runPublic()">
      Run Public Markets Diligence
    </button>

    <hr class="divider">

    <!-- Data fetch status -->
    <div>
      <label>Data Sources</label>
      <div class="sources-grid" id="sourcesGrid">
        <div class="source-row"><div class="source-dot" id="src-edgar"></div>SEC EDGAR</div>
        <div class="source-row"><div class="source-dot" id="src-ct"></div>ClinicalTrials.gov</div>
        <div class="source-row"><div class="source-dot" id="src-fda"></div>openFDA</div>
        <div class="source-row"><div class="source-dot" id="src-pubmed"></div>PubMed</div>
        <div class="source-row"><div class="source-dot" id="src-uspto"></div>USPTO Patents</div>
      </div>
    </div>

    <hr class="divider">

    <!-- Module progress -->
    <div id="progressPanel" style="display:none">
      <label>Analysis Progress</label>
      <div class="progress-bar-wrap"><div class="progress-bar" id="progressBar"></div></div>
      <div class="step-list" id="stepList"></div>
    </div>
  </div>

  <!-- Right panel -->
  <div class="right-panel" id="rightPanel">
    <div class="empty-state">
      <div class="empty-icon">&#x1F4C8;</div>
      <h2>No analysis yet</h2>
      <p>Enter a ticker symbol and click <strong>Run Public Markets Diligence</strong>
         to fetch live data from EDGAR, ClinicalTrials.gov, FDA, USPTO, and PubMed,
         then run 10 analysis modules.</p>
      <p style="margin-top:0.5rem; font-size:0.8rem;">
        Data sources are free APIs &mdash; no paid subscriptions required.
        Each run uses ~20,000 Groq tokens.
      </p>
    </div>
  </div>
</div>

<script>
const MODULES = [
  "M1: Company Identity",
  "M2: Pipeline Mapping",
  "M3: Catalyst Calendar",
  "M4: Clinical PoS",
  "M5: Commercial Opportunity",
  "M6: IP & Exclusivity",
  "M7: Financial Health",
  "M8: Ownership & Sentiment",
  "M9: Competitive Landscape",
  "M10: rNPV Synthesis",
];

const SOURCE_MAP = {
  "SEC EDGAR": "src-edgar",
  "ClinicalTrials.gov": "src-ct",
  "openFDA": "src-fda",
  "PubMed": "src-pubmed",
  "USPTO": "src-uspto",
};

let currentResults = {};

function srcId(name) {
  for (const [k, v] of Object.entries(SOURCE_MAP)) {
    if (name.toLowerCase().includes(k.toLowerCase())) return v;
  }
  return null;
}

function setSrcStatus(name, status) {
  const id = srcId(name);
  if (!id) return;
  const el = document.getElementById(id);
  if (el) { el.className = "source-dot " + status; }
}

function resetSources() {
  Object.values(SOURCE_MAP).forEach(id => {
    const el = document.getElementById(id);
    if (el) el.className = "source-dot";
  });
}

function initStepList() {
  const list = document.getElementById("stepList");
  list.innerHTML = MODULES.map((m, i) =>
    `<div class="step-item" id="step-${i}"><span class="step-icon">&#x25CB;</span>${m}</div>`
  ).join("");
  document.getElementById("progressPanel").style.display = "block";
  document.getElementById("progressBar").style.width = "0%";
}

function setStep(idx, state) {
  const el = document.getElementById(`step-${idx}`);
  if (!el) return;
  el.className = "step-item " + state;
  const icons = { active: "&#x25CF;", done: "&#x2713;" };
  el.querySelector(".step-icon").innerHTML = icons[state] || "&#x25CB;";
}

function fmtUsd(val) {
  if (val == null) return '<span class="field-null">null</span>';
  const n = parseFloat(val);
  if (n >= 1e3) return `$${(n/1e3).toFixed(1)}B`;
  return `$${n.toFixed(0)}M`;
}

function confTag(c) {
  if (!c) return "";
  return `<span class="confidence-tag conf-${c}">${c}</span>`;
}

function esc(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

function nullOrVal(v, suffix) {
  if (v == null || v === "null") return '<span class="field-null">null</span>';
  return esc(String(v)) + (suffix || "");
}

// ── Renderers for each module ─────────────────────────────────────────────────

function renderM1(r) {
  return `
    <p style="font-size:0.875rem;line-height:1.7;color:#c9d1d9;margin-bottom:1rem">
      ${esc(r.snapshot_paragraph || "")}
    </p>
    <div class="field-grid">
      <div class="field"><div class="field-label">Therapeutic Area</div>
        <div class="field-val">${nullOrVal(r.therapeutic_area)}</div></div>
      <div class="field"><div class="field-label">Stage</div>
        <div class="field-val">${nullOrVal(r.stage)}</div></div>
      <div class="field"><div class="field-label">Cash</div>
        <div class="field-val">${r.cash_usd_m != null ? fmtUsd(r.cash_usd_m) : '<span class="field-null">null</span>'}</div></div>
      <div class="field"><div class="field-label">Runway</div>
        <div class="field-val">${nullOrVal(r.runway_months, " months")}</div></div>
      <div class="field"><div class="field-label">Modality</div>
        <div class="field-val">${nullOrVal(r.modality_flag)}</div></div>
      <div class="field"><div class="field-label">Commercial Status</div>
        <div class="field-val">${nullOrVal(r.commercial_status)}</div></div>
    </div>
    ${(r.immediate_flags||[]).length ? `<div style="margin-top:0.75rem"><div class="field-label" style="margin-bottom:0.35rem">Flags</div>${r.immediate_flags.map(f=>`<div class="flag-item">⚠️ ${esc(f)}</div>`).join("")}</div>` : ""}
    ${r.china_exposure && r.china_exposure !== "null" ? `<div class="flag-item" style="margin-top:0.5rem;color:#f85149">🇨🇳 China exposure: ${esc(r.china_exposure)}</div>` : ""}`;
}

function renderM2(r) {
  const progs = r.programmes || [];
  if (!progs.length) return '<p style="color:var(--muted);font-size:0.85rem">No programmes extracted.</p>';
  const rows = progs.map(p => `
    <tr>
      <td><strong>${esc(p.name||"?")}</strong></td>
      <td>${esc(p.indication||"")}</td>
      <td>${esc(p.phase||"")}</td>
      <td>${esc(p.estimated_completion||"—")}</td>
      <td>${esc(p.next_catalyst||"—")}</td>
    </tr>`).join("");
  const discr = (r.discrepancies||[]).filter(Boolean);
  return `
    <table class="rnpv-table">
      <thead><tr><th>Asset</th><th>Indication</th><th>Phase</th><th>Est. Completion</th><th>Next Catalyst</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    ${discr.length ? `<div style="margin-top:0.75rem"><div class="field-label" style="margin-bottom:0.35rem">⚠️ Discrepancies (CT.gov vs filings)</div>${discr.map(d=>`<div class="flag-item">${esc(d)}</div>`).join("")}</div>` : ""}`;
}

function renderM3(r) {
  const cats = r.catalysts || [];
  if (!cats.length) return '<p style="color:var(--muted);font-size:0.85rem">No catalysts identified.</p>';
  const rows = cats.map(c => `
    <tr>
      <td>${esc(c.expected_date||"—")}</td>
      <td><strong>${esc(c.event||"")}</strong> — ${esc(c.programme||"")}</td>
      <td>${esc(c.indication||"")}</td>
      <td><span style="color:${c.price_impact==="High"?"var(--green)":c.price_impact==="Medium"?"var(--yellow)":"var(--muted)"}">${esc(c.price_impact||"—")}</span></td>
    </tr>`).join("");
  const pdufa = r.pdufa_source_note ? `<div class="sources-note">⚠️ ${esc(r.pdufa_source_note)}</div>` : "";
  return `
    <table class="rnpv-table">
      <thead><tr><th>Date</th><th>Event</th><th>Indication</th><th>Impact</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>${pdufa}`;
}

function renderM4(r) {
  const ests = r.pos_estimates || [];
  if (!ests.length) return '<p style="color:var(--muted);font-size:0.85rem">No PoS estimates.</p>';
  const rows = ests.map(e => `
    <tr>
      <td><strong>${esc(e.name||"?")}</strong></td>
      <td>${esc(e.phase||"")}</td>
      <td>${esc(e.modality||"")}</td>
      <td class="val-green">${esc(e.calculated_pos||"—")}</td>
      <td>${esc(e.analyst_pos||"—")}</td>
      <td style="font-size:0.78rem;color:var(--muted)">${esc(e.key_swing_factor||"—")}</td>
    </tr>`).join("");
  return `
    <table class="rnpv-table">
      <thead><tr><th>Asset</th><th>Phase</th><th>Modality</th><th>Calc PoS</th><th>Analyst PoS</th><th>Key Swing Factor</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function renderM5(r) {
  const items = r.commercial_assessments || [];
  if (!items.length) return '<p style="color:var(--muted);font-size:0.85rem">No commercial data.</p>';
  return items.map(c => `
    <div style="margin-bottom:1rem;padding-bottom:1rem;border-bottom:1px solid var(--border)">
      <div style="font-weight:600;margin-bottom:0.4rem">${esc(c.asset||"?")} <span style="font-weight:400;color:var(--muted);font-size:0.8rem">${esc(c.indication||"")}</span></div>
      <div class="field-grid" style="grid-template-columns:1fr 1fr 1fr">
        <div class="field"><div class="field-label">Bear</div><div class="field-val val-red">${c.peak_sales_bear_usdm!=null?fmtUsd(c.peak_sales_bear_usdm):"—"}</div></div>
        <div class="field"><div class="field-label">Base</div><div class="field-val val-yellow">${c.peak_sales_base_usdm!=null?fmtUsd(c.peak_sales_base_usdm):"—"}</div></div>
        <div class="field"><div class="field-label">Bull</div><div class="field-val val-green">${c.peak_sales_bull_usdm!=null?fmtUsd(c.peak_sales_bull_usdm):"—"}</div></div>
      </div>
      ${c.patent_cliff_flag&&c.patent_cliff_flag!=="null"?`<div class="flag-item" style="margin-top:0.4rem">⏳ ${esc(c.patent_cliff_flag)}</div>`:""}
    </div>`).join("");
}

function renderM7(r) {
  const runway = r.runway_months;
  const rColor = runway == null ? "var(--muted)" :
    runway < 6 ? "var(--red)" : runway < 12 ? "var(--yellow)" : "var(--green)";
  return `
    <div class="field-grid">
      <div class="field"><div class="field-label">Cash</div>
        <div class="field-val">${r.cash_usd_m!=null?fmtUsd(r.cash_usd_m):'<span class="field-null">null</span>'}</div></div>
      <div class="field"><div class="field-label">Monthly Burn</div>
        <div class="field-val">${r.monthly_burn_usd_m!=null?fmtUsd(r.monthly_burn_usd_m):'<span class="field-null">null</span>'}</div></div>
      <div class="field"><div class="field-label">Runway</div>
        <div class="field-val" style="color:${rColor}">${runway!=null?runway+" months":'<span class="field-null">null</span>'}</div></div>
      <div class="field"><div class="field-label">Runway Assessment</div>
        <div class="field-val">${nullOrVal(r.runway_assessment)}</div></div>
      <div class="field"><div class="field-label">Dilution Risk</div>
        <div class="field-val">${nullOrVal(r.dilution_risk)}</div></div>
      <div class="field"><div class="field-label">Negotiating Position</div>
        <div class="field-val">${nullOrVal(r.negotiating_position)}</div></div>
    </div>
    ${(r.key_observations||[]).length?`<div style="margin-top:0.75rem">${r.key_observations.map(o=>`<div class="flag-item" style="color:var(--muted)">${esc(o)}</div>`).join("")}</div>`:""}`;
}

function renderM10(r) {
  const rnpv = r.rnpv || {};
  const sens = rnpv.wacc_sensitivity || {};
  const stance = r.investment_stance || "Neutral";
  return `
    <div class="stance stance-${stance}">${stance}</div>
    <p style="font-size:0.875rem;line-height:1.7;color:#c9d1d9;margin-bottom:1rem">
      ${esc(r.investment_rationale||"")}
    </p>
    <div class="field-label" style="margin-bottom:0.4rem">rNPV — Equity Value (USD)</div>
    <table class="rnpv-table">
      <thead><tr><th>Scenario</th><th>Bear</th><th>Base</th><th>Bull</th></tr></thead>
      <tbody>
        <tr><td>WACC 10%</td><td class="val-red">—</td>
          <td class="val-yellow">${sens["10_pct_base_usd_m"]!=null?fmtUsd(sens["10_pct_base_usd_m"]):"—"}</td><td class="val-green">—</td></tr>
        <tr><td>WACC 12% (base)</td>
          <td class="val-red">${rnpv.equity_value_bear_usd_m!=null?fmtUsd(rnpv.equity_value_bear_usd_m):"—"}</td>
          <td class="val-yellow">${rnpv.equity_value_base_usd_m!=null?fmtUsd(rnpv.equity_value_base_usd_m):"—"}</td>
          <td class="val-green">${rnpv.equity_value_bull_usd_m!=null?fmtUsd(rnpv.equity_value_bull_usd_m):"—"}</td></tr>
        <tr><td>WACC 15%</td><td class="val-red">—</td>
          <td class="val-yellow">${sens["15_pct_base_usd_m"]!=null?fmtUsd(sens["15_pct_base_usd_m"]):"—"}</td><td class="val-green">—</td></tr>
      </tbody>
    </table>
    ${rnpv.per_share_base_usd!=null?`<div class="field" style="margin-top:0.75rem"><div class="field-label">Implied per-share value (base case)</div><div class="field-val val-yellow">$${rnpv.per_share_base_usd}</div></div>`:""}
    ${(r.top_swing_factors||[]).length?`
    <div style="margin-top:0.75rem">
      <div class="field-label" style="margin-bottom:0.35rem">Top Swing Factors</div>
      ${r.top_swing_factors.map(f=>`<div class="flag-item" style="color:${f.direction==="upside"?"var(--green)":"var(--red)"}">
        ${f.direction==="upside"?"&#x2191;":"&#x2193;"} ${esc(f.factor||"")}
      </div>`).join("")}
    </div>`:""}
    ${(r.key_risks_not_in_model||[]).length?`
    <div style="margin-top:0.75rem">
      <div class="field-label" style="margin-bottom:0.35rem">Unmodelled Risks</div>
      ${r.key_risks_not_in_model.map(k=>`<div class="flag-item">⚠️ ${esc(k)}</div>`).join("")}
    </div>`:""}`;
}

function renderM6(r) {
  const loe = r.approved_asset_loe || [];
  const gaps = r.data_gaps || [];
  const fto  = r.freedom_to_operate_risks || [];
  return `
    <div class="field-grid" style="grid-template-columns:1fr 1fr;margin-bottom:0.75rem">
      <div class="field"><div class="field-label">Patent Data</div>
        <div class="field-val">${r.patent_data_available===false||r.patent_data_available==="false"
          ? '<span style="color:var(--red)">Unavailable</span>'
          : (r.patent_count!=null ? r.patent_count+" patents" : "—")
        }</div></div>
      <div class="field"><div class="field-label">IP Type</div>
        <div class="field-val">${nullOrVal(r.ip_type_assessment)}</div></div>
    </div>
    ${r.portfolio_assessment ? `<p style="font-size:0.85rem;color:var(--muted);margin-bottom:0.75rem">${esc(r.portfolio_assessment)}</p>` : ""}
    ${loe.length ? `
    <div style="margin-bottom:0.75rem">
      <div class="field-label" style="margin-bottom:0.35rem">Regulatory Exclusivity (LOE)</div>
      <table class="rnpv-table">
        <thead><tr><th>Asset</th><th>Type</th><th>Approval</th><th>Est. LOE</th><th>Flag</th></tr></thead>
        <tbody>${loe.map(a => `<tr>
          <td><strong>${esc(a.asset||"?")}</strong></td>
          <td style="font-size:0.78rem">${esc(a.exclusivity_type||"—")}</td>
          <td>${esc(a.approval_date||"—")}</td>
          <td>${esc(a.estimated_loe||"—")}</td>
          <td style="color:${a.flag&&a.flag!=="null"?"var(--red)":"var(--muted)"}">${esc(a.flag&&a.flag!=="null"?a.flag:"✓")}</td>
        </tr>`).join("")}</tbody>
      </table>
    </div>` : ""}
    ${fto.length ? `
    <div style="margin-bottom:0.6rem">
      <div class="field-label" style="margin-bottom:0.3rem">Freedom-to-Operate Risks</div>
      ${fto.map(f => `<div class="flag-item">⚠️ ${esc(f)}</div>`).join("")}
    </div>` : ""}
    ${gaps.length ? `
    <div>
      <div class="field-label" style="margin-bottom:0.3rem;color:var(--muted)">Data Gaps</div>
      ${gaps.map(g => `<div class="flag-item" style="color:var(--muted);font-size:0.8rem">⬜ ${esc(g)}</div>`).join("")}
    </div>` : ""}`;
}

function renderM8(r) {
  const ins  = r.insider_summary || {};
  const inst = r.institutional_summary || {};
  const flag = r.sentiment_flag || "Insufficient data";
  const flagColor = flag === "Positive" ? "var(--green)"
                  : flag === "Negative" ? "var(--red)"
                  : flag === "Neutral"  ? "var(--yellow)"
                  : "var(--muted)";
  const note = r._note || inst.data_limitation || "";
  return `
    <div class="field-grid" style="grid-template-columns:repeat(3,1fr);margin-bottom:0.75rem">
      <div class="field"><div class="field-label">Insider Buys (90d)</div>
        <div class="field-val" style="color:var(--green)">${ins.buys_90d!=null?ins.buys_90d:"—"}</div></div>
      <div class="field"><div class="field-label">Insider Sells (90d)</div>
        <div class="field-val" style="color:var(--red)">${ins.sells_90d!=null?ins.sells_90d:"—"}</div></div>
      <div class="field"><div class="field-label">Net Signal</div>
        <div class="field-val" style="color:${ins.net_signal==="Bullish"?"var(--green)":ins.net_signal==="Bearish"?"var(--red)":"var(--muted)"}">${esc(ins.net_signal||"—")}</div></div>
    </div>
    ${(ins.notable_events||[]).length ? `
    <div style="margin-bottom:0.75rem">
      <div class="field-label" style="margin-bottom:0.3rem">Notable Events</div>
      ${ins.notable_events.map(e => `<div class="flag-item">${esc(e)}</div>`).join("")}
    </div>` : ""}
    <div style="margin-bottom:0.75rem">
      <div class="field-label" style="margin-bottom:0.3rem">Institutional Holders (SC 13G/D)</div>
      ${(inst.known_holders||[]).length
        ? inst.known_holders.map(h => `<div class="flag-item" style="color:var(--muted)">${esc(h)}</div>`).join("")
        : `<div style="font-size:0.82rem;color:var(--muted)">${esc(inst.recent_changes||"Insufficient data from free sources")}</div>`
      }
    </div>
    <div class="field" style="margin-bottom:0.6rem">
      <div class="field-label">Sentiment Signal</div>
      <div class="field-val" style="color:${flagColor}">${esc(flag)}</div>
    </div>
    ${note ? `<div style="font-size:0.78rem;color:var(--muted);margin-top:0.5rem;font-style:italic">${esc(note)}</div>` : ""}`;
}

function renderM9(r) {
  const items = r.competitive_assessments || [];
  if (!items.length) return '<p style="color:var(--muted);font-size:0.85rem">No competitive data available.</p>';
  return items.map(c => {
    const comps = c.competitors_identified || [];
    const crowdColor = c.crowding_risk && c.crowding_risk.startsWith("High") ? "var(--red)"
                     : c.crowding_risk && c.crowding_risk.startsWith("Medium") ? "var(--yellow)"
                     : "var(--green)";
    return `
    <div style="margin-bottom:1rem;padding-bottom:1rem;border-bottom:1px solid var(--border)">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem">
        <div><strong>${esc(c.our_asset||"?")}</strong>
          <span style="color:var(--muted);font-size:0.82rem;margin-left:0.5rem">${esc(c.indication||"")} · ${esc(c.our_phase||"")}</span>
        </div>
        <div style="display:flex;gap:0.5rem;align-items:center">
          <span style="font-size:0.78rem;padding:2px 8px;border-radius:4px;background:rgba(255,255,255,0.07)">${esc(c.positioning||"Unclear")}</span>
          <span style="font-size:0.78rem;color:${crowdColor}">${esc(c.crowding_risk||"—")}</span>
        </div>
      </div>
      ${c.key_differentiator&&c.key_differentiator!=="null"
        ? `<div style="font-size:0.82rem;color:var(--muted);margin-bottom:0.5rem">💡 ${esc(c.key_differentiator)}</div>` : ""}
      ${comps.length ? `
      <table class="rnpv-table" style="font-size:0.78rem">
        <thead><tr><th>Company</th><th>Mechanism</th><th>Phase</th><th>Timeline</th></tr></thead>
        <tbody>${comps.map(comp => `<tr>
          <td>${esc(comp.company||"Unknown")}</td>
          <td>${esc(comp.mechanism||"—")}</td>
          <td>${esc(comp.phase||"—")}</td>
          <td style="color:var(--muted)">${esc(comp.timeline||"unknown")}</td>
        </tr>`).join("")}</tbody>
      </table>` : ""}
    </div>`;
  }).join("");
}

function renderGeneric(r) {
  // Show a clear error banner if the module failed
  if (r._error) {
    const isRateLimit = r._rate_limited || r._error.includes("429") || r._error.toLowerCase().includes("rate limit");
    if (isRateLimit) {
      return `<p style="color:var(--yellow);font-size:0.85rem">⏳ Rate limit reached — this module will retry automatically on the next run. Try again in 1 minute.</p>`;
    }
    return `<p style="color:var(--red);font-size:0.85rem">⚠️ ${esc(r._error.slice(0, 200))}</p>`;
  }
  // Fallback: render key-value pairs, skipping internal fields
  const skip = new Set(["_module","_module_label","_confidence","_sources","_data_ts",
                         "_nulls","nulls_flagged","_pos_calculations","_rnpv_detail"]);
  const entries = Object.entries(r).filter(([k]) => !skip.has(k) && !k.startsWith("_"));
  if (!entries.length) return '<p style="color:var(--muted);font-size:0.85rem">No structured data available.</p>';
  return entries.slice(0, 6).map(([k, v]) => {
    const disp = typeof v === "object" ? JSON.stringify(v).slice(0, 120) : String(v).slice(0, 200);
    return `<div class="field" style="margin-bottom:0.5rem">
      <div class="field-label">${esc(k.replace(/_/g," "))}</div>
      <div class="field-val">${esc(disp)}</div>
    </div>`;
  }).join("");
}

const MODULE_RENDERERS = {
  m01_identity:   renderM1,
  m02_pipeline:   renderM2,
  m03_catalysts:  renderM3,
  m04_pos:        renderM4,
  m05_commercial: renderM5,
  m06_ip:         renderM6,
  m07_financial:  renderM7,
  m08_ownership:  renderM8,
  m09_competitive:renderM9,
  m10_rnpv:       renderM10,
};

function renderModuleCard(result) {
  const key   = result._module;
  const label = result._module_label || key;
  const conf  = result._confidence;
  const sources = (result._sources || []).join(", ");
  const ts    = result._data_ts || "";
  const nulls = result.nulls_flagged || result._nulls || [];
  const fn    = MODULE_RENDERERS[key] || renderGeneric;
  const body  = fn(result);

  return `
  <div class="result-card" id="card-${key}">
    <h3>${esc(label)}${confTag(conf)}</h3>
    ${body}
    <div class="sources-note">
      Sources: ${esc(sources) || "—"} &nbsp;&middot;&nbsp; ${esc(ts)}
      ${nulls.length ? `&nbsp;&middot;&nbsp; <span style="color:var(--red)">Nulls: ${nulls.map(esc).join(", ")}</span>` : ""}
    </div>
  </div>`;
}

function renderExecSummary(r) {
  const m10 = r.m10_rnpv;
  if (!m10 || !m10.executive_summary) return "";
  return `<div class="exec-summary">
    <h2>Executive Summary &mdash; ${esc(currentResults._ticker||"")}</h2>
    <p>${esc(m10.executive_summary)}</p>
  </div>`;
}

// ── Main run function ─────────────────────────────────────────────────────────

async function runPublic() {
  const ticker = document.getElementById("ticker").value.trim().toUpperCase();
  if (!ticker) { alert("Please enter a ticker symbol."); return; }

  const inputs = document.getElementById("pubInputs").value.trim();
  const runBtn = document.getElementById("runBtn");
  const panel  = document.getElementById("rightPanel");

  runBtn.disabled = true;
  currentResults  = { _ticker: ticker };
  resetSources();
  initStepList();

  panel.innerHTML = `<div style="color:var(--muted);padding:2rem;font-size:0.875rem">
    Fetching data from EDGAR, ClinicalTrials.gov, FDA, USPTO, PubMed&hellip;</div>`;

  const evtSource = new EventSource(`/public/run?ticker=${encodeURIComponent(ticker)}&inputs=${encodeURIComponent(inputs)}`);

  // Reopen as POST via fetch + ReadableStream (SSE over POST not supported natively)
  // Instead use query params (inputs may be long — use body via fetch streams)
  evtSource.close();

  // POST-based SSE via fetch
  let buffer = "";
  try {
    const resp = await fetch("/public/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker, inputs }),
    });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\\n");
      buffer = lines.pop();
      let eventType = null, dataLine = null;
      for (const line of lines) {
        if (line.startsWith("event:")) eventType = line.slice(6).trim();
        if (line.startsWith("data:"))  dataLine  = line.slice(5).trim();
        if (line === "" && dataLine) {
          try {
            handleEvent(JSON.parse(dataLine));
          } catch(e) {}
          eventType = null; dataLine = null;
        }
      }
    }
  } catch(e) {
    panel.innerHTML = `<div class="error-box"><strong>Connection error:</strong> ${esc(String(e))}</div>`;
  }

  runBtn.disabled = false;
}

function handleEvent(evt) {
  const panel = document.getElementById("rightPanel");

  if (evt.type === "fetch") {
    setSrcStatus(evt.source, evt.status === "ok" ? "ok" :
                              evt.status === "error" ? "error" : "fetching");
    return;
  }

  if (evt.type === "module_start") {
    const idx = evt.step - 1;
    setStep(idx, "active");
    document.getElementById("progressBar").style.width = `${(evt.step-1)/evt.total*100}%`;
    if (panel.querySelector(".empty-state") ||
        panel.innerHTML.includes("Fetching data")) {
      panel.innerHTML = "";
    }
    return;
  }

  if (evt.type === "module_done") {
    const idx = evt.step - 1;
    setStep(idx, "done");
    document.getElementById("progressBar").style.width = `${evt.step/evt.total*100}%`;
    const result = evt.result;
    currentResults[result._module] = result;
    panel.innerHTML += renderModuleCard(result);
    panel.scrollTop = panel.scrollHeight;
    return;
  }

  if (evt.type === "complete") {
    document.getElementById("progressBar").style.width = "100%";
    // Prepend exec summary at the top
    const summary = renderExecSummary(currentResults);
    if (summary) panel.insertAdjacentHTML("afterbegin", summary);
    return;
  }

  if (evt.type === "rate_limit") {
    // Show a persistent banner — don't stop the run, it will auto-resume
    const existing = document.getElementById("rate-limit-banner");
    if (!existing) {
      const banner = document.createElement("div");
      banner.id = "rate-limit-banner";
      banner.style.cssText = "background:rgba(210,153,34,0.15);border:1px solid var(--yellow);border-radius:8px;padding:0.75rem 1rem;margin-bottom:1rem;font-size:0.85rem;color:var(--yellow);display:flex;align-items:center;gap:0.5rem";
      banner.innerHTML = `⏳ <span>${esc(evt.message||"Rate limit reached — auto-resuming…")}</span>`;
      panel.insertBefore(banner, panel.firstChild);
    }
    return;
  }

  if (evt.type === "error") {
    panel.innerHTML += `<div class="error-box"><strong>Error:</strong> ${esc(evt.message||"Unknown error")}</div>`;
    document.getElementById("runBtn").disabled = false;
    return;
  }
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    try:
        import uvicorn
    except ImportError:
        raise SystemExit("Run: pip3 install uvicorn")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
