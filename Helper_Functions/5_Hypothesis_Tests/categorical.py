import numpy as np
from scipy.stats import chisquare, chi2_contingency, fisher_exact, norm
import pandas as pd


# ----------------------------------------------------------
# 1) One-way Chi-Squared Test:Chi-Square Test for Goodness of Fit
# ----------------------------------------------------------
# Use when: comparing observed counts in categories vs. an expected distribution. 
# Example: did 3 versions get clicks according to an expected 40/30/30 split?
def chi_square_gof(observed, expected_probs):
    """
    One-way Chi-Squared Goodness-of-Fit test.

    Parameters
    ----------
    observed : array-like
        Observed counts per category.
    expected_probs : array-like
        Expected proportions per category (must sum to 1).

    Returns
    -------
    chi2 : float
        Chi-squared statistic.
    p_value : float
        p-value of the test.
    expected : np.ndarray
        Expected counts.
    """
    observed = np.asarray(observed, dtype=float)
    expected_probs = np.asarray(expected_probs, dtype=float)

    if observed.ndim != 1:
        raise ValueError("observed must be a 1D array")

    if len(observed) != len(expected_probs):
        raise ValueError("observed and expected_probs must have the same length")

    if not np.isclose(expected_probs.sum(), 1.0):
        raise ValueError("expected_probs must sum to 1")

    expected = expected_probs * observed.sum()

    chi2, p_value = chisquare(f_obs=observed, f_exp=expected)
    return chi2, p_value, expected

# observed = [420, 310, 270]
# expected_probs = [0.4, 0.3, 0.3]

# chi2, p, expected = chi_square_gof(observed, expected_probs)

# print("chi2 =", chi2)
# print("p    =", p)
# print("expected counts =", expected)


# ----------------------------------------------------------------
# 2) Chi-Square Test for Association (Contingency Table)
# ----------------------------------------------------------------
# Use when: testing independence/association between 2 categorical variables.
# Example: conversion (yes/no) differs by country (US/CA/UK)?

def chi_square_contingency(table, correction=False):
    """
    Chi-Squared test of independence for a contingency table.

    Parameters
    ----------
    table : array-like (RxC)
        Contingency table of observed counts.
    correction : bool
        Whether to apply Yates' correction (default False).

    Returns
    -------
    chi2 : float
        Chi-squared statistic.
    p_value : float
        p-value of the test.
    dof : int
        Degrees of freedom.
    expected : np.ndarray
        Expected counts under independence.
    """
    table = np.asarray(table, dtype=float)

    if table.ndim != 2:
        raise ValueError("table must be a 2D array (contingency table)")

    if np.any(table < 0):
        raise ValueError("table counts must be non-negative")

    chi2, p_value, dof, expected = chi2_contingency(
        table, correction=correction
    )
    return chi2, p_value, dof, expected
# table = [
#     [120, 880],   # US
#     [ 90, 910],   # CA
#     [ 60, 940],   # UK
# ]

# chi2, p, dof, expected = chi_square_contingency(table)

# print("chi2 =", chi2)
# print("dof  =", dof)
# print("p    =", p)
# print("expected counts:\n", expected)

# ------------------------
# 3) Fisher’s Exact Test
# ------------------------
# Use when: 2x2 contingency table, especially with small counts.
# Example: conversion differs between Control vs Treatment?
def fisher_exact_test(table, alternative="two-sided"):
    """
    Fisher's Exact Test for a 2x2 contingency table.

    Parameters
    ----------
    table : array-like (2x2)
        Contingency table of observed counts.
    alternative : str
        'two-sided', 'less', or 'greater'.

    Returns
    -------
    odds_ratio : float
        Estimated odds ratio.
    p_value : float
        p-value of the test.
    """
    table = np.asarray(table, dtype=float)
    if not np.all(np.isfinite(table)) or np.any(table != np.floor(table)):
        raise ValueError("table counts must be finite integers")

    if table.shape != (2, 2):
        raise ValueError("Fisher's Exact Test requires a 2x2 table")

    if np.any(table < 0):
        raise ValueError("table counts must be non-negative")

    odds_ratio, p_value = fisher_exact(
        table, alternative=alternative
    )
    return odds_ratio, p_value

# Rows: group (control, treatment)
# Cols: outcome (converted, not_converted)
# table_2x2 = np.array([
#     [8,  92],   # control
#     [18, 82],   # treatment
# ])

# odds_ratio, p = fisher_exact_test(table_2x2, alternative="two-sided")
# print("\nFisher’s Exact Test (2x2)")
# print("  odds_ratio =", odds_ratio)
# print("  p          =", p)
# Interpretation: small p => evidence of different conversion rates between groups.


def contingency_table(x, y):
    """
    Create a contingency table from two categorical variables.

    Parameters
    ----------
    x, y : array-like
        Categorical variables.

    Returns
    -------
    pd.DataFrame
        Contingency table.
    """
    return pd.crosstab(x, y)

def two_proportion_ci_test(x1, n1, x2, n2, conf=0.95):
    """
    Compare two proportions: p1 - p2

    Inputs:
      x1, n1 : successes and trials in group 1
      x2, n2 : successes and trials in group 2

    Returns:
      difference, z-stat, p-value, confidence interval
    """
    if not 0 < conf < 1:
        raise ValueError("conf must be between 0 and 1")
    for successes, trials in ((x1, n1), (x2, n2)):
        if (not np.isfinite([successes, trials]).all()
                or trials <= 0 or not 0 <= successes <= trials
                or successes != int(successes) or trials != int(trials)):
            raise ValueError("Require integer counts with n>0 and 0<=x<=n")
    p1 = x1 / n1
    p2 = x2 / n2
    diff = p1 - p2

    # CI (unpooled SE)
    zcrit = norm.ppf(0.5 + conf / 2)
    se_ci = np.sqrt(
        p1 * (1 - p1) / n1 +
        p2 * (1 - p2) / n2
    )
    ci = (diff - zcrit * se_ci, diff + zcrit * se_ci)

    # Hypothesis test (pooled SE)
    p_pool = (x1 + x2) / (n1 + n2)
    se_test = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = diff / se_test if se_test > 0 else 0.0
    p_value = 2 * norm.sf(abs(z))

    return {
        "p1_hat": p1,
        "p2_hat": p2,
        "difference": diff,
        "z": z,
        "p_value": p_value,
        "ci": ci
    }
