# Synthetic platform revenue intelligence

Open `04_platform_revenue_intelligence.ipynb` and select **Python (llm)**. Run all cells to generate data, run real Timestral-5M forecasts, and save charts, tables and CSV evidence. The checkpoint downloads on first use.

```sh
conda activate llm
python -m pip install -r requirements-platform-revenue.txt
python -m ipykernel install --user --name llm --display-name 'Python (llm)'
```

The reusable analytical code is `platform_revenue.py`. Four focused tests protect accounting, reproducibility, cutoff/delay isolation and calibration rank; the executed notebook provides the model integration check.

```sh
conda run -n llm python -m pytest -q
```

The sample contains 1825 fictional daily records and injected discovery, supply and payment incidents. It models platform revenue, not individual CLTV. All behavior metrics are historical covariates; only a predetermined calendar is known in the future. The synthetic paying-user count is an expected continuous count.

A fixed chronological schedule uses three development, nine calibration and four evaluation origins, with disjoint 90-day target windows. Results include group additions/removals, delayed-signal comparisons, calibrated total-revenue intervals, multivariate KPI anomaly flags and an exact three-factor accounting decomposition. Four test origins are illustrative, not enough for production evidence. Residual intervals have no guaranteed coverage under temporal dependence or regime change.

Generated evidence lives in `artifacts/platform_revenue/`; `manifest.json` records the checkpoint and environment. The notebook's findings are measured synthetic results, never causal impact estimates. The design and implementation plan live under `docs/superpowers/`.
