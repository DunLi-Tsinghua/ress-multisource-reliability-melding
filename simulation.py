"""Top-level reproducibility runner.

Run:
    python simulation.py

This delegates to scripts/simulate_reliability.py and regenerates the
controlled synthetic data tables, result summaries, and manuscript figures.
"""

from scripts.simulate_reliability import main


if __name__ == "__main__":
    main()
