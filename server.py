#!/usr/bin/env python3
"""
Biotech Diligence Agent — FastAPI Web Server

Provides a REST API to run diligence analysis and retrieve memos.

Endpoints:
  POST /diligence          — Run full diligence on a company
  POST /diligence/module   — Run a single module
  POST /diligence/update   — Update with new data
  GET  /history/{company}  — Get run history
  GET  /runs/{company}/{run_id} — Get specific run
  GET  /health             — Health check
"""

import os
import json
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, PlainTextResponse
    from pydantic import BaseModel
except ImportError:
    raise SystemExit("FastAPI not installed. Run: pip3 install fastapi uvicorn")

from biotech_diligence.orchestrator import DiligenceOrchestrator, MODULE_MAP

app = FastAPI(
    title="Biotech Diligence Agent",
    description="VC-grade biotech investment analysis powered by Claude AI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_orchestrator() -> DiligenceOrchestrator:
    return DiligenceOrchestrator(verbose=False)


# ------------------------------------------------------------------
# Request / Response models
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
    return {"status": "ok", "version": "1.0.0"}


@app.get("/modules")
def list_modules():
    return {"modules": list(MODULE_MAP.keys())}


@app.post("/diligence")
def run_full_diligence(req: DiligenceRequest):
    """Run full 11-module diligence and return memo."""
    try:
        orch = get_orchestrator()
        memo = orch.run_full_diligence(
            company=req.company,
            inputs=req.inputs,
            output_format=req.format,
        )
        return {"company": req.company, "memo": memo, "format": req.format}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/diligence/module")
def run_single_module(req: ModuleRequest):
    """Run a single diligence module."""
    if req.module not in MODULE_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown module: {req.module}. Valid: {list(MODULE_MAP.keys())}",
        )
    try:
        orch = get_orchestrator()
        result = orch.run_module(
            module_name=req.module,
            company=req.company,
            inputs=req.inputs,
            prior_results=req.prior_results,
        )
        return {"company": req.company, "module": req.module, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/diligence/update")
def update_diligence(req: UpdateRequest):
    """Update existing diligence with new data."""
    try:
        orch = get_orchestrator()
        memo = orch.update_with_new_data(
            company=req.company,
            new_inputs=req.new_inputs,
            modules_to_rerun=req.modules_to_rerun,
            output_format=req.format,
        )
        return {"company": req.company, "memo": memo}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{company}")
def get_history(company: str):
    """Get all diligence runs for a company."""
    from biotech_diligence.state_manager import StateManager
    mgr = StateManager()
    runs = mgr.list_runs(company)
    return {"company": company, "runs": runs}


@app.get("/runs/{company}/{run_id}")
def get_run(company: str, run_id: str):
    """Get a specific diligence run."""
    from biotech_diligence.state_manager import StateManager
    mgr = StateManager()
    run = mgr.load_run(company, run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return run


@app.get("/companies")
def list_companies():
    """List all companies with saved diligence."""
    from biotech_diligence.state_manager import StateManager
    mgr = StateManager()
    return {"companies": mgr.list_companies()}


@app.get("/", response_class=HTMLResponse)
def index():
    """Simple HTML dashboard."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Biotech Diligence Agent</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #0a0f1e; color: #e2e8f0; min-height: 100vh; }
    header { background: #111827; border-bottom: 1px solid #1e3a5f;
             padding: 1.5rem 2rem; display: flex; align-items: center; gap: 1rem; }
    header h1 { font-size: 1.4rem; font-weight: 700; color: #38bdf8; }
    header span { font-size: 0.8rem; background: #1e3a5f; color: #7dd3fc;
                  padding: 2px 10px; border-radius: 12px; }
    main { max-width: 900px; margin: 2rem auto; padding: 0 1.5rem; }
    .card { background: #111827; border: 1px solid #1e293b; border-radius: 12px;
            padding: 1.5rem; margin-bottom: 1.5rem; }
    .card h2 { font-size: 1rem; color: #94a3b8; margin-bottom: 1rem;
               text-transform: uppercase; letter-spacing: 0.05em; }
    input, textarea, select { width: 100%; background: #0a0f1e; border: 1px solid #334155;
              color: #e2e8f0; padding: 0.7rem 1rem; border-radius: 8px; font-size: 0.9rem;
              margin-bottom: 0.8rem; font-family: inherit; }
    textarea { min-height: 100px; resize: vertical; }
    button { background: #0284c7; color: white; border: none; padding: 0.7rem 1.5rem;
             border-radius: 8px; cursor: pointer; font-size: 0.9rem; font-weight: 600;
             transition: background 0.2s; }
    button:hover { background: #0369a1; }
    button:disabled { background: #334155; cursor: not-allowed; }
    #output { background: #0a0f1e; border: 1px solid #334155; border-radius: 8px;
              padding: 1rem; font-family: 'Courier New', monospace; font-size: 0.8rem;
              white-space: pre-wrap; max-height: 600px; overflow-y: auto; color: #94a3b8; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
    .badge-invest { background: #064e3b; color: #6ee7b7; }
    .badge-watch  { background: #451a03; color: #fcd34d; }
    .badge-pass   { background: #450a0a; color: #fca5a5; }
    .loading { color: #38bdf8; animation: pulse 1s infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.5} }
    .endpoint { background: #0a0f1e; border-radius: 6px; padding: 0.5rem 0.8rem;
                margin: 0.3rem 0; font-family: monospace; font-size: 0.8rem; color: #7dd3fc; }
    .method { color: #34d399; font-weight: bold; margin-right: 0.5rem; }
  </style>
</head>
<body>
  <header>
    <h1>Biotech Diligence Agent</h1>
    <span>VC-Grade Analysis</span>
  </header>
  <main>
    <div class="card">
      <h2>Run Full Diligence</h2>
      <input id="company" type="text" placeholder="Company name (e.g. Relay Therapeutics, Agenus...)" />
      <textarea id="inputs" placeholder="Optional: paste raw inputs (trial data, press releases, deck text)..."></textarea>
      <button onclick="runDiligence()" id="runBtn">Run Diligence</button>
    </div>
    <div class="card">
      <h2>Output</h2>
      <div id="output">Enter a company name and click Run Diligence...</div>
    </div>
    <div class="card">
      <h2>API Endpoints</h2>
      <div class="endpoint"><span class="method">POST</span>/diligence — Full diligence</div>
      <div class="endpoint"><span class="method">POST</span>/diligence/module — Single module</div>
      <div class="endpoint"><span class="method">POST</span>/diligence/update — Update with new data</div>
      <div class="endpoint"><span class="method">GET</span>&nbsp;/history/{company} — Run history</div>
      <div class="endpoint"><span class="method">GET</span>&nbsp;/companies — All analyzed companies</div>
      <div class="endpoint"><span class="method">GET</span>&nbsp;/docs — Interactive API docs (Swagger)</div>
    </div>
  </main>
  <script>
    async function runDiligence() {
      const company = document.getElementById('company').value.trim();
      const inputs = document.getElementById('inputs').value.trim();
      const btn = document.getElementById('runBtn');
      const out = document.getElementById('output');
      if (!company) { out.textContent = 'Please enter a company name.'; return; }
      btn.disabled = true;
      btn.textContent = 'Analyzing...';
      out.innerHTML = '<span class="loading">Running 11-module diligence analysis... this takes 60-120 seconds...</span>';
      try {
        const res = await fetch('/diligence', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({company, inputs: inputs || null, format: 'markdown'})
        });
        const data = await res.json();
        if (res.ok) {
          out.textContent = data.memo;
        } else {
          out.textContent = 'Error: ' + (data.detail || JSON.stringify(data));
        }
      } catch(e) {
        out.textContent = 'Error: ' + e.message;
      } finally {
        btn.disabled = false;
        btn.textContent = 'Run Diligence';
      }
    }
    document.getElementById('company').addEventListener('keydown', e => {
      if (e.key === 'Enter') runDiligence();
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    try:
        import uvicorn
    except ImportError:
        raise SystemExit("uvicorn not installed. Run: pip3 install uvicorn")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
