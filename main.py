#!/usr/bin/env python3
"""
Biotech Diligence Agent — CLI Entry Point

Usage:
  python main.py "Relay Therapeutics"
  python main.py "Agenus" --inputs inputs/agenus_data.txt
  python main.py "Karuna Therapeutics" --module rapid_screen
  python main.py "Relay Therapeutics" --update "New Phase 2 data shows..."
  python main.py "Relay Therapeutics" --history
  python main.py "Relay Therapeutics" --compare RUN_A RUN_B
"""

import sys
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Biotech VC Diligence Agent — Investment-grade analysis for biotech companies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("company", help="Company or asset name to analyze")
    parser.add_argument(
        "--inputs", "-i",
        type=str,
        default=None,
        help="Path to a text file with raw inputs (trial data, press releases, deck text)",
    )
    parser.add_argument(
        "--module", "-m",
        type=str,
        default=None,
        help="Run a single module only (e.g., rapid_screen, scientific_validation, decision_engine)",
    )
    parser.add_argument(
        "--update", "-u",
        type=str,
        default=None,
        help="New data to update an existing diligence with (triggers re-run of affected modules)",
    )
    parser.add_argument(
        "--rerun-modules",
        type=str,
        default=None,
        help="Comma-separated list of modules to rerun when using --update",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Show diligence history for a company",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("RUN_A", "RUN_B"),
        help="Compare two diligence runs: --compare RUN_A_ID RUN_B_ID",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: print to stdout). Use .md for markdown.",
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["markdown", "plain"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-6",
        help="Claude model to use (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save state after this run",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Lazy import to avoid slow startup when --help is used
    from biotech_diligence.orchestrator import DiligenceOrchestrator

    orch = DiligenceOrchestrator(
        model=args.model if args.model != "claude-sonnet-4-6" else None,
        save_state=not args.no_save,
        verbose=not args.quiet,
    )

    # Load inputs file if provided
    raw_inputs = None
    if args.inputs:
        inputs_path = Path(args.inputs)
        if not inputs_path.exists():
            print(f"Error: Inputs file not found: {args.inputs}", file=sys.stderr)
            sys.exit(1)
        raw_inputs = inputs_path.read_text()

    # Determine action
    output = None

    if args.history:
        output = orch.list_history(args.company)

    elif args.compare:
        output = orch.compare_theses(args.company, args.compare[0], args.compare[1])

    elif args.update:
        rerun = None
        if args.rerun_modules:
            rerun = [m.strip() for m in args.rerun_modules.split(",")]
        output = orch.update_with_new_data(
            company=args.company,
            new_inputs=args.update,
            modules_to_rerun=rerun,
            output_format=args.format,
        )

    elif args.module:
        result = orch.run_module(
            module_name=args.module,
            company=args.company,
            inputs=raw_inputs,
        )
        import json
        output = json.dumps(result, indent=2)

    else:
        # Full diligence
        output = orch.run_full_diligence(
            company=args.company,
            inputs=raw_inputs,
            output_format=args.format,
        )

    # Output
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output)
        print(f"\nMemo saved to: {out_path}")
    else:
        print("\n" + output)


if __name__ == "__main__":
    main()
