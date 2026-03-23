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
            q.put(("memo", memo))
        except Exception as e:
            q.put(("error", str(e)))
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
  .layout { display: grid; grid-template-columns: 340px 1fr; min-height: calc(100vh - 57px); }

  /* ── Left Panel ── */
  .left-panel { background: var(--surface); border-right: 1px solid var(--border);
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
    <a href="/docs" class="header-link">API Docs</a>
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
