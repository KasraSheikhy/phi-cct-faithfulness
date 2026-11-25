"""Recompute the recorded study's analysis and verify it against raw responses."""
import json
from pathlib import Path

from analyze_results import analyze
from review_actual_examples import run_mention_sensitivity
from verify_real_outputs import verify_real_outputs

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT / "config/experiment.json").read_text())
    analyze(ROOT, config, "full")
    run_mention_sensitivity(ROOT, config)
    verify_real_outputs(ROOT)
    print("Analysis complete. Results: outputs/results/; plots: outputs/figures/")


if __name__ == "__main__":
    main()
