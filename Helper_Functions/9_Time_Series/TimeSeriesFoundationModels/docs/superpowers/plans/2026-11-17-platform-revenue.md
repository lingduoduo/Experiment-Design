# Platform Revenue Intelligence Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** An executed synthetic-data notebook implementing the four proposed analytical methods.
**Architecture:** platform_revenue.py contains generation, forecasting and analysis; 04_platform_revenue_intelligence.ipynb presents results and exports artifacts.
**Tech Stack:** conda llm, NumPy, pandas, matplotlib, Timestral, nbclient.
**Spec:** docs/superpowers/specs/2026-11-17-platform-revenue-design.md

## Global Constraints
Preserve user files; synthetic data only; true platform revenue target; no realized future KPI covariates; use llm; no causal claims; compact tests.

## Review Focus
Future data must not affect a forecast at a fixed cutoff. Delays must shift only selected historical features. Accounting components must sum to observed revenue change. Calibration must exclude final evaluation outcomes. Synthetic seed must reproduce the accounting-consistent sample.

## Task 1 — Analytical helper
- [x] Write four focused failing tests in test_platform_revenue.py for generation, cutoff/delay isolation, exact decomposition, and finite-sample residual calibration.
- [x] Run and observe failures before implementing.
- [x] Implement make_sample, forecast_at, decomposition, calibration_radius, run_study in platform_revenue.py; use seeded data and explicit chronological partitions.
- [x] Run compact suite and resolve failures.

## Task 2 — Executed notebook and artifacts
- [x] Create 04_platform_revenue_intelligence.ipynb with generator narrative, data audit, experiments, paired summaries, delay study, calibrated anomaly plots, decomposition and evidence-based investigation summary.
- [x] Execute with llm kernel and Timestral-5M; save sample and forecast/evaluation CSVs under artifacts/platform_revenue.
- [x] Verify every code cell ran without errors, inspect plots and measured results, and request one independent review.
- [x] Address material findings and record final evidence. Leave changes uncommitted for user review.

## Execution evidence
- All four analytical guards observed failing before implementation, then passing.
- First real notebook run exposed a pandas Series `.dt` accessor collision; timestamp label access corrected and complete notebook rerun passed.
- Twelve code cells executed successfully with Python (llm), real Timestral-5M weights, seven saved plots and fourteen CSV/JSON artifacts.
- Independent reviewer found no critical/important issues. Its tie-count naming suggestion was already addressed by naming the field removal_nonimproving_origins, which includes ties explicitly.
- Reviewed saved heatmap and anomaly figures visually. Four focused pytest checks pass.
- Existing notebooks and unrelated user deletions were preserved; changes remain uncommitted.
