"""
Diligence Orchestrator — coordinates all modules, manages state, and produces memos.
"""

import os
from typing import Optional

from .providers import auto_client

from .modules import (
    RapidScreenModule,
    ScientificValidationModule,
    DataEvidenceModule,
    DevelopmentPathwayModule,
    CompetitiveLandscapeModule,
    MarketCommercialModule,
    TeamExecutionModule,
    IPRegulatoryModule,
    RiskDecompositionModule,
    InvestmentFramingModule,
    DecisionEngineModule,
)
from .memo_generator import MemoGenerator
from .state_manager import StateManager


# Ordered pipeline: upstream outputs feed downstream modules
MODULE_PIPELINE = [
    ("rapid_screen", RapidScreenModule),
    ("scientific_validation", ScientificValidationModule),
    ("data_evidence", DataEvidenceModule),
    ("development_pathway", DevelopmentPathwayModule),
    ("competitive_landscape", CompetitiveLandscapeModule),
    ("market_commercial", MarketCommercialModule),
    ("team_execution", TeamExecutionModule),
    ("ip_regulatory", IPRegulatoryModule),
    ("risk_decomposition", RiskDecompositionModule),
    ("investment_framing", InvestmentFramingModule),
    ("decision_engine", DecisionEngineModule),
]

MODULE_MAP = {name: cls for name, cls in MODULE_PIPELINE}


class DiligenceOrchestrator:
    """
    Main agent orchestrator.

    Usage:
        orch = DiligenceOrchestrator()
        memo = orch.run_full_diligence("Relay Therapeutics", inputs="...")
        print(memo)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        save_state: bool = True,
        verbose: bool = True,
    ):
        self.client, detected_model = auto_client()
        self.model = model or detected_model
        self.save_state = save_state
        self.verbose = verbose
        self.memo_gen = MemoGenerator()
        self.state_mgr = StateManager()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_full_diligence(
        self,
        company: str,
        inputs: Optional[str] = None,
        output_format: str = "markdown",
        progress_callback=None,
    ) -> str:
        """
        Run all 11 modules in sequence and return a formatted memo.

        Args:
            company: Company or asset name.
            inputs: Optional raw text (trial data, press releases, deck text, etc.)
            output_format: "markdown" or "plain"

        Returns:
            Formatted investment memo as a string.
        """
        self._log(f"\n{'='*60}")
        self._log(f"  BIOTECH DILIGENCE AGENT")
        self._log(f"  Company: {company}")
        self._log(f"{'='*60}\n")

        context = {"company": company, "inputs": inputs or ""}
        results = {}

        total = len(MODULE_PIPELINE)
        for i, (module_name, module_cls) in enumerate(MODULE_PIPELINE, 1):
            self._log(f"Running module: {module_cls.MODULE_LABEL}...")
            if progress_callback:
                progress_callback({"step": i, "total": total, "module": module_cls.MODULE_LABEL, "status": "running"})
            module = module_cls(self.client, self.model)
            result = module.run(context)
            results[module_name] = result
            context[module_name] = result
            self._log(f"  → Done")
            if progress_callback:
                progress_callback({"step": i, "total": total, "module": module_cls.MODULE_LABEL, "status": "done"})

        if self.save_state:
            run_id = self.state_mgr.save_run(company, results)
            self._log(f"\nState saved. Run ID: {run_id}")
        else:
            run_id = None

        self._log(f"\nGenerating memo...\n")
        memo = self.memo_gen.generate(
            company=company,
            results=results,
            run_id=run_id,
            format=output_format,
        )

        return memo

    def run_module(
        self,
        module_name: str,
        company: str,
        inputs: Optional[str] = None,
        prior_results: Optional[dict] = None,
    ) -> dict:
        """
        Run a single module independently.

        Args:
            module_name: One of the module keys (e.g., "rapid_screen", "decision_engine")
            company: Company name
            inputs: Optional raw text inputs
            prior_results: Optional dict of prior module outputs to include as context

        Returns:
            Module result dict
        """
        if module_name not in MODULE_MAP:
            raise ValueError(
                f"Unknown module: {module_name}. Valid modules: {list(MODULE_MAP.keys())}"
            )

        context = {"company": company, "inputs": inputs or ""}
        if prior_results:
            context.update(prior_results)

        module_cls = MODULE_MAP[module_name]
        module = module_cls(self.client, self.model)
        self._log(f"Running module: {module_cls.MODULE_LABEL}...")
        result = module.run(context)
        self._log(f"  → Done")
        return result

    def update_with_new_data(
        self,
        company: str,
        new_inputs: str,
        modules_to_rerun: Optional[list[str]] = None,
        output_format: str = "markdown",
    ) -> str:
        """
        Update an existing diligence with new data (e.g., new trial results).
        Loads the latest run, merges new inputs, and re-runs specified modules.

        Args:
            company: Company name
            new_inputs: New data to incorporate
            modules_to_rerun: List of module names to rerun. If None, reruns all.
            output_format: "markdown" or "plain"

        Returns:
            Updated memo
        """
        latest = self.state_mgr.load_latest_run(company)
        if not latest:
            self._log(f"No prior run found for {company}. Running fresh diligence.")
            return self.run_full_diligence(company, new_inputs, output_format)

        prior_results = latest["results"]
        combined_inputs = (latest.get("inputs", "") + "\n\nUPDATE:\n" + new_inputs).strip()

        self._log(f"Loaded prior run: {latest['run_id']}")
        self._log(f"Updating with new data...")

        target_modules = modules_to_rerun or list(MODULE_MAP.keys())
        context = {"company": company, "inputs": combined_inputs}
        # Seed context with all prior results; overwrite as we re-run modules
        context.update(prior_results)

        results = dict(prior_results)  # start with prior state

        for module_name, module_cls in MODULE_PIPELINE:
            if module_name in target_modules:
                self._log(f"Re-running: {module_cls.MODULE_LABEL}...")
                module = module_cls(self.client, self.model)
                result = module.run(context)
                results[module_name] = result
                context[module_name] = result
                self._log(f"  → Done")

        if self.save_state:
            run_id = self.state_mgr.save_run(company, results)
            self._log(f"Updated state saved. Run ID: {run_id}")
        else:
            run_id = None

        return self.memo_gen.generate(company=company, results=results, run_id=run_id, format=output_format)

    def compare_theses(self, company: str, run_id_a: str, run_id_b: str) -> str:
        """
        Compare two diligence runs and surface thesis changes.
        Returns a formatted report.
        """
        comparison = self.state_mgr.compare_runs(company, run_id_a, run_id_b)
        if "error" in comparison:
            return f"Error: {comparison['error']}"

        lines = [
            f"# THESIS EVOLUTION: {company}",
            f"Run A: {comparison['run_a']['id']} ({comparison['run_a']['timestamp'][:10]})",
            f"Run B: {comparison['run_b']['id']} ({comparison['run_b']['timestamp'][:10]})",
            "",
            "## Changes Detected:",
            "",
        ]
        changes = comparison.get("changes", {})
        if not changes:
            lines.append("No changes detected between runs.")
        else:
            for module, diffs in changes.items():
                lines.append(f"### {module.replace('_', ' ').title()}")
                for d in diffs:
                    lines.append(f"- **{d['field']}**: `{d['before']}` → `{d['after']}`")
                lines.append("")
        return "\n".join(lines)

    def list_history(self, company: str) -> str:
        """Return formatted history of diligence runs for a company."""
        runs = self.state_mgr.list_runs(company)
        if not runs:
            return f"No diligence history found for: {company}"
        lines = [f"# Diligence History: {company}", ""]
        for r in runs:
            lines.append(
                f"- `{r['run_id']}` | {r.get('timestamp', '')[:10]} | "
                f"**{r.get('verdict', 'N/A')}** ({r.get('confidence', 'N/A')}) | "
                f"{r.get('ic_one_liner', '')}"
            )
        return "\n".join(lines)

    def _log(self, msg: str):
        if self.verbose:
            print(msg)
