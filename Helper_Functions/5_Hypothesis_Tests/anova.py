import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

def anova_one_way(values, groups, alpha=0.05):
    """
    One-way ANOVA: compare means across multiple groups.

    Parameters
    ----------
    values : array-like
        Numeric observations
    groups : array-like
        Group labels (same length as values)
    alpha : float
        Significance level (default 0.05)

    Returns
    -------
    dict with ANOVA table and decision
    """
    df = pd.DataFrame({
        "value": values,
        "group": groups
    })

    # Fit ANOVA model
    model = ols("value ~ C(group)", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    f_stat = anova_table.loc["C(group)", "F"]
    p_value = anova_table.loc["C(group)", "PR(>F)"]

    decision = "Reject H0 (group means differ)" if p_value < alpha else "Fail to reject H0"

    return {
        "f_statistic": f_stat,
        "p_value": p_value,
        "alpha": alpha,
        "decision": decision,
        "anova_table": anova_table,
        "model": model
    }

# scores = [78,85,88,92,75, 88,90,94,91,89, 70,72,68,75,73]
# methods = ["A"]*5 + ["B"]*5 + ["C"]*5

# res = anova_one_way(scores, methods)

# print("F:", res["f_statistic"])
# print("p-value:", res["p_value"])
# print(res["decision"])
# print(res["anova_table"])


def anova_two_way(values, factor_a, factor_b, alpha=0.05):
    """
    Two-way ANOVA with interaction.

    Parameters
    ----------
    values : array-like
        Numeric observations
    factor_a : array-like
        First categorical factor
    factor_b : array-like
        Second categorical factor
    alpha : float
        Significance level

    Returns
    -------
    dict with ANOVA table and decisions
    """
    df = pd.DataFrame({
        "value": values,
        "A": factor_a,
        "B": factor_b
    })

    model = ols("value ~ C(A) * C(B)", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    results = {}
    for term in anova_table.index.drop("Residual"):
        p = anova_table.loc[term, "PR(>F)"]
        results[term] = {
            "F": anova_table.loc[term, "F"],
            "p_value": p,
            "decision": "Reject H0" if p < alpha else "Fail to reject H0"
        }

    return {
        "alpha": alpha,
        "anova_table": anova_table,
        "effects": results,
        "model": model
    }
# values = [78,85,88,92,75, 88,90,94,91,89]
# method = ["A","A","A","B","B","A","A","B","B","B"]
# gender = ["M","F","M","F","M","F","M","F","M","F"]

# res = anova_two_way(values, method, gender)

# print(res["anova_table"])
# print(res["effects"])
