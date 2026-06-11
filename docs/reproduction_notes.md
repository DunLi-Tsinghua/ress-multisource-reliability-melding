# Reproduction Notes

This repository reproduces the controlled numerical experiments for:

Reliability Assessment of Complex Industrial Systems under Multi-Source
Uncertain Information: An Operational-Profile-Aware Imprecise Bayesian Melding
Framework.

The study uses controlled synthetic numerical data only. It does not use real
industrial field data, proprietary maintenance records, or private asset logs.

## Reproduction Steps

1. Create and activate a Python environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run `python simulation.py` from the repository root.
4. Check regenerated CSV files under `data/` and `results/`.
5. Check regenerated PDF and PNG figures under `figures/`.

The random seed is fixed at `20260611`. The main finite scenario-wise envelope
uses 20,000 prior samples. The sampling-stability check uses 10,000, 20,000,
and 50,000 samples.

## Expected Numerical Checks

- Main 400 h envelope: approximately `[0.7724, 0.8096]`.
- Stressed-profile 400 h envelope: approximately `[0.723, 0.765]`.
- Reliability requirement: `R_req = 0.78`.
- Validation-planning widths over 12 budget units:
  - uniform: approximately `0.049` to `0.031`;
  - profile-proportional: approximately `0.049` to `0.024`;
  - risk-targeted: approximately `0.049` to `0.016`.

The finite scenario-wise envelopes are exact only over the declared finite
prior, source-weight, and profile scenarios encoded in the script and config.
