# Reproducibility Package: Multi-Source Reliability Melding

This repository contains the reproducibility package for the controlled
numerical experiments in the manuscript:

**Reliability Assessment of Complex Industrial Systems under Multi-Source
Uncertain Information: An Operational-Profile-Aware Imprecise Bayesian Melding
Framework**

The repository regenerates the controlled synthetic evidence tables, finite
scenario-wise reliability summaries, sampling-stability table, and all PDF/PNG
figures used in the manuscript.

## Data Scope

All data in this repository are controlled synthetic numerical data generated
for methodological illustration.

No proprietary industrial field data are included. No private maintenance
records are included. No real industrial asset logs, field failure databases,
API keys, passwords, or confidential files are required or used.

## Repository Contents

```text
ress-multisource-reliability-melding/
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- simulation.py
|-- scripts/
|   `-- simulate_reliability.py
|-- config/
|   `-- experiment_config.yaml
|-- data/
|   |-- controlled_parameters.csv
|   |-- load_multipliers.csv
|   |-- source_weight_scenarios.csv
|   `-- README.md
|-- results/
|   |-- reliability_envelope_summary.csv
|   |-- profile_shift_summary.csv
|   |-- prior_imprecision_summary.csv
|   |-- source_inclusion_summary.csv
|   |-- validation_planning_summary.csv
|   `-- sampling_stability_summary.csv
|-- figures/
|   |-- fig1_framework.pdf
|   |-- fig1_framework.png
|   |-- fig2_rbd.pdf
|   |-- fig2_rbd.png
|   |-- fig3_reliability_envelope.pdf
|   |-- fig3_reliability_envelope.png
|   |-- fig4_profile_shift.pdf
|   |-- fig4_profile_shift.png
|   |-- fig5_prior_imprecision.pdf
|   |-- fig5_prior_imprecision.png
|   |-- fig6_source_inclusion.pdf
|   |-- fig6_source_inclusion.png
|   |-- fig7_decision_regions.pdf
|   |-- fig7_decision_regions.png
|   |-- fig8_validation_planning.pdf
|   `-- fig8_validation_planning.png
`-- docs/
    `-- reproduction_notes.md
```

## Python Environment

The package was checked with:

- Python 3.10.9
- numpy 1.23.5
- scipy 1.10.0
- pandas 1.5.3
- matplotlib 3.7.0
- pyyaml 6.0

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Reproduce Figures and Tables

From the repository root, run:

```bash
python simulation.py
```

This regenerates:

- all manuscript figures in both PDF and PNG formats under `figures/`;
- controlled parameter and synthetic evidence tables under `data/`;
- source-weight scenario table under `data/`;
- finite scenario-wise result summaries under `results/`;
- the 10,000 / 20,000 / 50,000 sample stability table under `results/`.

The random seed is fixed at `20260611`.

## Expected Numerical Checks

The regenerated outputs should reproduce the manuscript values:

- main 400 h finite envelope: approximately `[0.7724, 0.8096]`;
- stressed-profile 400 h finite envelope: approximately `[0.723, 0.765]`;
- reliability requirement: `R_req = 0.78`;
- validation-planning width reductions over 12 budget units:
  - uniform: approximately `0.049` to `0.031`;
  - profile-proportional: approximately `0.049` to `0.024`;
  - risk-targeted: approximately `0.049` to `0.016`;
- sampling-stability table for 10,000, 20,000, and 50,000 importance samples.
