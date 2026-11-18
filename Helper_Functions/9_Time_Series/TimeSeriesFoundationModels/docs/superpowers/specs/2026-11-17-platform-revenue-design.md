# Synthetic platform revenue intelligence

Implement the methods proposed in the conversation against the platform-revenue objective in timestral_kpi_intelligence_proposal.md. User requested generated sample data and implementation, using Superpowers and conda llm.

## Approach
A new executed notebook plus a reusable helper builds on the existing Python notebook workflow. Keeping all logic in notebook cells makes reuse and leakage checks difficult; a dashboard adds deployment work before there is evidence to present. Use a helper and notebook.

## Design
Generate five years (1825 days) of seeded daily audience, discovery, engagement/social, supply and monetization data. Document the synthetic generator's delayed dependencies, seasonality, campaigns and shock labels; labels never enter the model. Daily platform revenue is DAU × payer rate × revenue per payer. Export sample CSV and result CSVs.

Use Timestral-5M on CPU in llm with a 365-day context. Use historical KPIs as past-only covariates; use a predetermined calendar as known-future covariates for every Timestral configuration. Forecast 90 days per origin and evaluate cumulative point predictions at 7/30/90 days. Compare seasonal naive, revenue-only, audience baseline, group additions, full model and group removals. Repeat the audience+discovery+engagement model at delays 0/1/7/14/30, shifting features and leaving revenue current.

Use origins 90 days apart: first three development, next nine calibration, final four evaluation. Freeze specifications before evaluation. Report paired error deltas and wins without significance claims from four test origins. Use finite-sample residual quantiles from calibration origins for cumulative intervals; daily multivariate KPI forecasts get historical-scale normalized residual thresholds. Report held-out coverage/false alarm diagnostics, label flagged synthetic events, and do not assume exchangeability or guaranteed time-series coverage.

Explain observed revenue change with an exact symmetric Shapley accounting decomposition over DAU, payer rate and revenue per payer, including all interactions. Use a single date pair for exact comparisons. KPI-level anomaly forecasts and group predictive value support investigation priorities; no causal attribution or intervention revenue estimates.

## Constraints and verification
Preserve existing notebooks and unrelated user changes. No dependency on removed timestral_example.py. Use installed upstream timestral package. Keep a compact set of analytical regression tests and execute the notebook with the llm kernel and real pretrained weights. Synthetic findings are demonstrations, not platform findings. No SHAP benchmark or production dashboard in this implementation; these were complementary options, not required for the proposed four methods.
