"""
State Manager — persists diligence runs and tracks thesis changes over time.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


STATE_DIR = Path(".diligence_state")


class StateManager:
    """Saves and loads diligence runs. Enables thesis tracking over time."""

    def __init__(self, state_dir: Optional[Path] = None):
        self.state_dir = state_dir or STATE_DIR
        self.state_dir.mkdir(exist_ok=True)

    def _company_slug(self, company: str) -> str:
        return company.lower().replace(" ", "_").replace("/", "_")[:40]

    def _company_dir(self, company: str) -> Path:
        d = self.state_dir / self._company_slug(company)
        d.mkdir(exist_ok=True)
        return d

    def save_run(self, company: str, results: dict, run_id: Optional[str] = None) -> str:
        """Save a full diligence run. Returns the run_id."""
        if run_id is None:
            run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        company_dir = self._company_dir(company)
        run_path = company_dir / f"{run_id}.json"

        payload = {
            "run_id": run_id,
            "company": company,
            "timestamp": datetime.utcnow().isoformat(),
            "results": results,
        }

        with open(run_path, "w") as f:
            json.dump(payload, f, indent=2)

        # Update index
        self._update_index(company, run_id, results)
        return run_id

    def _update_index(self, company: str, run_id: str, results: dict):
        """Maintain an index file with summary of all runs."""
        company_dir = self._company_dir(company)
        index_path = company_dir / "index.json"

        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
        else:
            index = {"company": company, "runs": []}

        # Extract key summary fields
        decision = results.get("decision_engine", {})
        screen = results.get("rapid_screen", {})

        index["runs"].append({
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "verdict": decision.get("verdict", screen.get("verdict", "N/A")),
            "confidence": decision.get("confidence", "N/A"),
            "ic_one_liner": decision.get("ic_one_liner", ""),
        })

        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

    def load_run(self, company: str, run_id: str) -> Optional[dict]:
        """Load a specific run."""
        run_path = self._company_dir(company) / f"{run_id}.json"
        if not run_path.exists():
            return None
        with open(run_path) as f:
            return json.load(f)

    def load_latest_run(self, company: str) -> Optional[dict]:
        """Load the most recent run for a company."""
        company_dir = self._company_dir(company)
        runs = sorted(company_dir.glob("*.json"))
        runs = [r for r in runs if r.name != "index.json"]
        if not runs:
            return None
        with open(runs[-1]) as f:
            return json.load(f)

    def list_runs(self, company: str) -> list[dict]:
        """List all runs for a company with summary."""
        index_path = self._company_dir(company) / "index.json"
        if not index_path.exists():
            return []
        with open(index_path) as f:
            index = json.load(f)
        return index.get("runs", [])

    def list_companies(self) -> list[str]:
        """List all companies with saved diligence."""
        return [d.name for d in self.state_dir.iterdir() if d.is_dir()]

    def compare_runs(self, company: str, run_id_a: str, run_id_b: str) -> dict:
        """
        Compare two runs and surface thesis changes.
        Returns a dict of changed fields between runs.
        """
        run_a = self.load_run(company, run_id_a)
        run_b = self.load_run(company, run_id_b)

        if not run_a or not run_b:
            return {"error": "One or both runs not found"}

        changes = {}
        modules = set(run_a["results"].keys()) | set(run_b["results"].keys())

        for mod in modules:
            a_data = run_a["results"].get(mod, {})
            b_data = run_b["results"].get(mod, {})
            mod_changes = _diff_dicts(a_data, b_data)
            if mod_changes:
                changes[mod] = mod_changes

        return {
            "company": company,
            "run_a": {"id": run_id_a, "timestamp": run_a["timestamp"]},
            "run_b": {"id": run_id_b, "timestamp": run_b["timestamp"]},
            "changes": changes,
        }


def _diff_dicts(a: dict, b: dict, path: str = "") -> list[dict]:
    """Recursively find differences between two dicts."""
    diffs = []
    all_keys = set(a.keys()) | set(b.keys())
    for key in all_keys:
        full_path = f"{path}.{key}" if path else key
        a_val = a.get(key)
        b_val = b.get(key)
        if isinstance(a_val, dict) and isinstance(b_val, dict):
            diffs.extend(_diff_dicts(a_val, b_val, full_path))
        elif a_val != b_val:
            # Skip internal metadata
            if not key.startswith("_"):
                diffs.append({
                    "field": full_path,
                    "before": a_val,
                    "after": b_val,
                })
    return diffs
