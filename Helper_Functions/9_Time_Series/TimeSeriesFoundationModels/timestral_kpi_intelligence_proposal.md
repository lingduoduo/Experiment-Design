# Timestral-Based KPI Intelligence: Understanding the Drivers of Long-Term Platform Revenue

## 1. Research Objective

Develop a **Timestral-powered KPI intelligence dashboard** to understand how user acquisition, content discovery, engagement, social interactions, and monetization relate to long-term cumulative platform revenue.

The primary objective is **not traditional revenue forecasting**. Instead, the system should help explain why revenue performance improves or deteriorates and identify which engagement signals are most informative about sustainable platform growth.

## 2. Core Business Hypothesis

**Better recommendations → Better content discovery → More meaningful engagement → Stronger platform participation and retention → Higher long-term revenue.**

For a large live-streaming platform, viewers do not necessarily establish persistent relationships with individual broadcasters:

- Viewers may browse many rooms without caring who the broadcaster is.
- Broadcasters may serve hundreds of thousands or millions of viewers without recognizing individuals.
- Broadcasters appear and disappear dynamically.
- Following a broadcaster is only one possible engagement signal.
- Users may return for the overall content experience rather than for specific broadcasters.

Therefore, persistent viewer–broadcaster pair counts should **not** be the primary measure of community strength. Instead, emphasize content-discovery quality, meaningful engagement, repeat platform participation, and monetization.

## 3. KPI Framework

Use daily platform-level observations with historical rolling windows (L1, L7, L30, L365).

| Dimension | Candidate KPIs |
|---|---|
| Audience | DAU, new viewers, returning viewers, retention |
| Content supply | Active broadcasters, available rooms, content diversity |
| Discovery | Room impressions, views, clicks, room joins, CTR |
| Engagement | Watch time, sessions, rooms per viewer, meaningful engagements |
| Social signals | Follows, unfollows, follow rate, follower growth |
| Monetization | Tipping users, tip amounts, revenue, revenue per active user |
| Long-term value | Cumulative platform revenue, rolling revenue growth |

Additional derived KPIs include views per DAU, clicks per view, meaningful room engagements per impression, watch time per room join, tipping conversion rate, revenue per active user, returning-viewer rate, and period-over-period revenue growth.

All ratios should use compatible measurement windows and clearly defined denominators.

## 4. Data Structure

One row per platform date:

```text
dt, dau, views_l1, views_l7, views_l30, views_l365,
clicks_l7, tips_l7, follows_l7, active_rooms_l7,
revenue_l1, revenue_l7, revenue_l30, revenue_l365, ...
```

Additional features:

- Day-over-day and week-over-week changes
- Historical lag features
- Non-overlapping historical intervals
- Engagement conversion rates
- Content availability and seasonality indicators

Use only data available as of each observation date to avoid temporal leakage.

## 5. Defining the Revenue Target

The objective is **platform-level cumulative revenue**, not individual-user CLTV.

Let daily platform revenue be $r_t$. Cumulative platform revenue is:

$$
C_t = \sum_{s \le t} r_s
$$

For diagnostic modeling, predict subsequent incremental revenue:

$$
Y_{t,H} = \sum_{h=1}^{H} r_{t+h}
$$

Then:

$$
\widehat C_{t+H} = C_t + \widehat Y_{t,H}
$$

This avoids relying on the naturally increasing cumulative total to explain business performance. Initial horizons: **7, 30, and 90 days**, with 365-day analysis if sufficient history is available.

## 6. Timestral Modeling Strategy

Timestral serves as the temporal modeling backbone. Use historical revenue and daily engagement covariates to investigate which signals provide information about subsequent revenue.

Organize covariates into five groups:

1. **Audience:** DAU, new and returning viewers, retention.
2. **Content supply:** Active broadcasters, rooms, content diversity.
3. **Discovery:** Impressions, views, clicks, CTR, room joins.
4. **Engagement and social:** Watch time, sessions, follows, unfollows.
5. **Monetization:** Historical tips, paying users, revenue.

| Experiment | Inputs |
|---|---|
| Baseline | Historical revenue only |
| Model A | Revenue + audience |
| Model B | Revenue + discovery |
| Model C | Revenue + engagement and social |
| Model D | Revenue + content supply |
| Full model | All available covariate groups |

Use **rolling-origin backtesting** and compare out-of-sample forecast errors. Feature groups that consistently improve performance provide evidence of incremental *predictive information*, not necessarily causal effects. Because features can be correlated, use both group addition and removal experiments.

## 7. Explaining Revenue Changes

The dashboard should answer three questions.

### What changed?

Identify changes in revenue, DAU, discovery, engagement, content supply, and monetization.

### Which factors are associated with the change?

Use complementary methods:

- **KPI decomposition** for observed revenue changes.
- **Timestral feature-group ablation** for incremental predictive value.
- **Lag analysis** for delayed relationships.
- **LightGBM and SHAP** as an interpretable benchmark.
- **Anomaly detection** for unusual engagement and revenue patterns.

### What should the business investigate?

Translate findings into opportunities in recommendation relevance, content availability, engagement, and monetization. Predictive attribution must be distinguished from causal impact; A/B testing is needed to establish whether interventions improve long-term value.

## 8. KPI Dashboard Design

### A. Executive Overview

- Cumulative platform revenue
- Rolling 7/30/365-day revenue
- DAU and returning viewers
- Revenue per active user
- Revenue growth versus previous periods

### B. Engagement and Content Discovery

- Views, clicks, CTR
- Room joins
- Watch time
- Meaningful engagement rate
- Follow/unfollow trends

### C. Content Supply

- Active broadcasters
- Active rooms
- Content diversity
- Supply changes associated with engagement changes

### D. Timestral Driver Intelligence

- Predictive importance by KPI group
- Incremental forecast improvement
- Historical lag relationships
- Unusual deviations from historical patterns
- Differences across short- and long-term horizons

### E. Business Interpretation

Produce concise, evidence-backed summaries, such as:

> Revenue declined alongside lower DAU and reduced discovery engagement. Historical audience and content-discovery metrics contributed the largest incremental predictive information, while tipping conversion remained stable.

This is an **illustrative** summary, not an observed result. Actual statements must be grounded in measured KPIs and model evaluation.

## 9. Important Research Considerations

- **Correlation versus causation:** Timestral can identify predictive temporal relationships but cannot independently establish causal effects.
- **Revenue leakage:** Historical tipping and revenue features must not overlap with the future revenue target.
- **Feature redundancy:** L1/L7/L30/L365 windows overlap. Non-overlapping intervals and normalized rates can improve interpretation.
- **Aggregation limitations:** Platform-level data can reveal aggregate trends but cannot establish individual viewer behavior or individual CLTV.
- **Recommendation quality:** CTR alone is insufficient; meaningful engagement, repeat participation, and monetization should also be evaluated.
- **Content supply:** Changes in broadcaster availability may explain discovery and engagement changes independently of recommendation quality.
- **Known-future covariates:** Do not supply future views, clicks, tips, or follows unless those future values are genuinely known; otherwise use historical covariates or independently generated scenarios.
- **Revenue accounting:** Cumulative revenue and revenue growth should be presented separately; a rising cumulative total can mask weakening daily performance.

## 10. Proposed Proof of Concept

**Phase 1 — Data preparation:** Build a daily Snowflake KPI dataset with rolling engagement and revenue features.

**Phase 2 — Timestral evaluation:** Establish revenue-only and multivariate benchmarks using chronological backtesting.

**Phase 3 — Driver attribution:** Evaluate covariate groups, temporal lags, and predictive sensitivity.

**Phase 4 — Dashboard:** Build an interactive dashboard showing KPI trends, Timestral diagnostics, and interpretable business explanations.

**Phase 5 — Validation:** Identify promising recommendation interventions and validate their effects through controlled experiments.

## 11. Expected Outcomes

The project should help answer:

- Why is platform revenue growing or declining?
- Which engagement metrics are the strongest leading indicators of revenue?
- Does content discovery contribute information beyond DAU?
- Do follows and other social signals provide additional predictive value?
- How does content availability affect engagement and monetization?
- Which signals matter most over 7, 30, 90, or 365 days?
- What opportunities should the recommendation team investigate?

## Conclusion

The central hypothesis is that **sustainable platform value depends on continuously connecting viewers with engaging content**, rather than necessarily establishing persistent one-to-one relationships with broadcasters.

Timestral provides a temporal modeling foundation for investigating these relationships, while KPI decomposition, feature ablation, and interpretable benchmarks make the results useful in a business dashboard.

**The ultimate goal is to move beyond monitoring what happened and toward understanding which engagement and content-discovery signals are most informative about long-term platform revenue.**
