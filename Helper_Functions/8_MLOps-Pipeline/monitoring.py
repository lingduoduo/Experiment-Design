import seaborn as sns
from scipy.stats import gaussian_kde, truncnorm
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import distance
from pyspark.sql import functions as F

def plot_single_distribution(distibution):
    """
    Plots the given distributions 
    
    :param distribution: rv_continuous 

    """
    sns.kdeplot(distibution, shade=True, color="g", label=1)
    plt.legend(loc="upper right", borderaxespad=0)

def get_truncated_right_skewed_values(dataframe, colname, scale=3):
    """
    Generates truncated distribution based on given right-skewed distribution
    
    :param dataframe: spark dataframe
    :param colname: column in spark dataframe 

    :return df: dataframe with upper outliers flagged  
    """
    Q1, Q3 = dataframe.approxQuantile(colname, [0.25, 0.75], 0.05)
    IQR = Q3 - Q1
    upper_bound = Q3 + scale * IQR
    print(f"{colname}'s upper_bound cutoff: {upper_bound}")
    upper_outliers = dataframe.filter(F.col(colname) > upper_bound)
    df_filtered_outliers = dataframe.withColumn(colname, F.when(F.col(colname) > upper_bound, F.lit(upper_bound)).otherwise(F.col(colname)))
    return df_filtered_outliers

def plot_distribution(distibution_1, distibution_2):
    """
    Plots the two given distributions 

    :param distribution_1: rv_continuous 
    :param distribution_2: rv_continuous 

    """
    sns.kdeplot(distibution_1, shade=True, color="g", label=1)
    sns.kdeplot(distibution_2, shade=True, color="b", label=2)
    plt.legend(loc="upper right", borderaxespad=0)

def get_truncated_normal(mean=0, sd=1, low=0.2, upp=0.8, n_size=1000, seed=999):
    """
    Generates truncated normal distribution based on given mean, standard deviation, lower bound, upper bound and sample size 

    :param mean: float, mean used to create the distribution 
    :param sd: float, standard deviation used to create distribution
    :param low: float, lower bound used to create the distribution 
    :param upp: float, upper bound used to create the distribution 
    :param n_size: integer, desired sample size 

    :return distb: rv_continuous 
    """
    np.random.seed(seed=seed)

    a = (low-mean) / sd
    b = (upp-mean) / sd
    distb = truncnorm(a, b, loc=mean, scale=sd).rvs(n_size, random_state=seed)
    return distb

def calculate_ks(distibution_1, distibution_2):
    """
    Helper function that calculated the KS stat and plots the two distributions used in the calculation 

    :param distribution_1: rv_continuous
    :param distribution_2: rv_continuous 

    :return p_value: float, resulting p-value from KS calculation
    :return ks_drift: bool, detection of significant difference across the distributions 
    """
    base, comp = distibution_1, distibution_2
    p_value = np.round(stats.ks_2samp(base, comp)[1],3)
    ks_drift = p_value < 0.05

    # Generate plots
    plot_distribution(base, comp)
    label = f"KS Stat suggests model drift: {ks_drift} \n P-value = {p_value}"
    plt.title(label, loc="center")
    return p_value, ks_drift

def calculate_probability_vector(distibution_1, distibution_2):
    """
    Helper function that turns raw values into a probability vector 

    :param distribution_1: rv_continuous
    :param distribution_2: rv_continuous 

    :return p: array, probability vector of distribution_1
    :return q: array, probability vector of distribution_2
    """
    global_min = min(min(distibution_1), min(distibution_2))
    global_max = max(max(distibution_1), max(distibution_2))
    
    p = np.histogram(distibution_1, bins=20, range=(global_min, global_max))
    q = np.histogram(distibution_2, bins=20, range=(global_min, global_max))
    
    return p[0], q[0]
    
def calculate_js_distance(p, q, raw_distribution_1, raw_distribution_2, threshold=0.2):
    """
    Helper function that calculated the JS distance and plots the two distributions used in the calculation 

    :param p: array, probability vector for the first distribution
    :param q: array, probability vector for the second distribution 
    :param raw_distribution_1: array, raw values used in plotting
    :param raw_distribution_2: array, raw values used in plotting
    :param threshold: float, cutoff threshold for the JS statistic

    :return js_stat: float, resulting distance measure from JS calculation
    :return js_drift: bool, detection of significant difference across the distributions 
    """
    js_stat = distance.jensenshannon(p, q, base=2)
    js_stat_rounded = np.round(js_stat, 3)
    js_drift = js_stat > threshold

    # Generate plot
    plot_distribution(raw_distribution_1, raw_distribution_2)
    label = f"Jensen Shannon suggests model drift: {js_drift} \n JS Distance = {js_stat_rounded}"
    plt.title(label, loc="center")

    return js_stat, js_drift

def handle_numeric_ks(pdf1, pdf2, continuous_columns, alpha=.05):
        """
        Handle the numeric features with the Two-Sample Kolmogorov-Smirnov (KS) Test with Bonferroni Correction 
        """
        corrected_alpha = alpha / len(continuous_columns)

        for num in continuous_columns:
            ks_stat, ks_pval = stats.ks_2samp(pdf1[num], pdf2[num], mode="asymp")
            if ks_pval <= corrected_alpha:
                print(f"Drift found in {num}!")
            else:
                print(f"No signs of drift in {num}.")

def handle_numeric_js(pdf1, pdf2, continuous_columns, js_stat_threshold=0.2):
    """
    Handles the numeric features with the Jensen Shannon (JS) test using the threshold attribute
    """
    for num in continuous_columns:
        # Run test comparing old and new for that attribute
        range_min = min(pdf1[num].min(), pdf2[num].min())
        range_max = max(pdf1[num].max(), pdf2[num].max())
        base = np.histogram(pdf1[num], bins=20, range=(range_min, range_max))
        comp = np.histogram(pdf2[num], bins=20, range=(range_min, range_max))
        js_stat = distance.jensenshannon(base[0], comp[0], base=2)
        if js_stat >= js_stat_threshold:
            print(f"Drift found in {num}!")
        else:
            print(f"No signs of drift in {num}.")

def generate_null_counts(pdf1, pdf2, palette="#2ecc71"):
    """
    Generate the visualization of percent null counts of all features
    Optionally provide a color palette for the visual
    """
    cm = sns.light_palette(palette, as_cmap=True)
    return pd.concat([100 * pdf1.isnull().sum() / len(pdf1), 
                        100 * pdf2.isnull().sum() / len(pdf2)], axis=1, 
                        keys=["pdf1", "pdf2"]).style.background_gradient(cmap=cm, text_color_threshold=0.5, axis=1)

def handle_categorical(pdf1, pdf2, categorical_columns, alpha=.05):
    """
    Handle the Categorical features with Two-Way Chi-Squared Test with Bonferroni Correction
    Note: null counts can skew the results of the Chi-Squared Test so they're currently dropped
        by `.value_counts()`
    """
    corrected_alpha = alpha / len(categorical_columns)

    for feature in categorical_columns:
        pdf_count1 = pd.DataFrame(pdf1[feature].value_counts()).sort_index().rename(columns={feature:"pdf1"})
        pdf_count2 = pd.DataFrame(pdf2[feature].value_counts()).sort_index().rename(columns={feature:"pdf2"})
        pdf_counts = pdf_count1.join(pdf_count2, how="outer")#.fillna(0)
        obs = np.array([pdf_counts["pdf1"], pdf_counts["pdf2"]])
        _, p, _, _ = stats.chi2_contingency(obs)
        if p < corrected_alpha:
            print(f"Drift found in {feature}!")
        else:
            print(f"No signs of drift in {feature}.")

def generate_percent_change(pdf1, pdf2, continuous_columns, palette="#2ecc71"):
    """
    Generate visualization of percent change in summary statistics of numeric features
    Optionally provide a color palette for the visual
    """
    cm = sns.light_palette(palette, as_cmap=True)
    summary1_pdf = pdf1.describe()[continuous_columns]
    summary2_pdf = pdf2.describe()[continuous_columns]
    percent_change = 100 * abs((summary1_pdf - summary2_pdf) / (summary1_pdf + 1e-100))
    return percent_change.style.background_gradient(cmap=cm, text_color_threshold=0.5, axis=1)
