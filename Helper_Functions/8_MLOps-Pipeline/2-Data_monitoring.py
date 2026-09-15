# Databricks notebook source
# MAGIC %md
# MAGIC ## Read-in Data
# MAGIC
# MAGIC Install SDV along with supporting libraries for visualization. (`pandas-profiling` needs a small update to pick up a bug fix.)

# COMMAND ----------

import dlt
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, truncnorm
from scipy import stats
from scipy.spatial import distance

# COMMAND ----------

new_tx = spark.sql("""
SELECT * 
FROM data_science_prod.default.transactions_users_silver
WHERE t_transaction_time > current_date() - INTERVAL 30 DAYS
""")

# COMMAND ----------

display(new_tx.agg(
    F.min("t_transaction_time").alias("min_transaction_time"),
    F.max("t_transaction_time").alias("max_transaction_time")
))

# COMMAND ----------

import sys
import os
sys.path.append(os.path.abspath('..'))

from datetime import datetime, timedelta, date
import dates, monitoring, encoder_decoder

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Generate sampled data

# COMMAND ----------

# Convert 't_transaction_time' to timestamp type if it's not already
df = new_tx.withColumn("t_transaction_time", F.col("t_transaction_time").cast("timestamp"))

# Extract month, week, and year from 't_transaction_time'
df = df.withColumn('month', F.month(F.col('t_transaction_time')))
df = df.withColumn('week', F.weekofyear(F.col('t_transaction_time')))
df = df.withColumn('year', F.year(F.col('t_transaction_time')))

def correct_week_year(week, month, year):
    if (week > 50) and month == 1:
        year -= 1
    return year

custom_udf = udf(correct_week_year, IntegerType())
df = df.withColumn("year", custom_udf(df["week"], df["month"],  df["year"]))
df = df.withColumn('year_week', F.col('year') * 100 + F.col('week')) 

# Get the most recent date
most_recent_date = df.select(F.max(F.to_date("t_transaction_time"))).collect()[0][0]
sample_date1 = dates.shift_date(dates.dt_str(most_recent_date), False, 7)
print(sample_date1)
sample_date2 = dates.shift_date(dates.dt_str(most_recent_date), False, 6)
print(sample_date2)

# Filter the dataset for the most recent date
group1 = df.filter(F.to_date("t_transaction_time") == sample_date1)
# Show the result
print(group1.count())
# Filter the dataset for the most recent date
group2 = df.filter(F.to_date("t_transaction_time") == sample_date2)
# Show the result
print(group2.count())

# COMMAND ----------

group1 = group1.toPandas()
group2 = group2.toPandas()

# COMMAND ----------

monitoring.generate_null_counts(group1, group2, palette="#2ecc71")

# COMMAND ----------

# MAGIC %md <i18n value="232b2c47-e056-4adf-8f74-9515e3fc164e"/>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Numeric Features

# COMMAND ----------

monitoring.plot_distribution(group1.t_price, group2.t_price)

# COMMAND ----------

mean1 = np.mean(group1.t_price)
std_dev1 = np.std(group1.t_price, ddof=1) 
group1_truncated = monitoring.get_truncated_normal(mean1, std_dev1)
mean2 = np.mean(group2.t_price)
std_dev2 = np.std(group2.t_price, ddof=1) 
group2_truncated = monitoring.get_truncated_normal(mean2, std_dev2)
monitoring.plot_distribution(group1_truncated , group2_truncated)

# COMMAND ----------

monitoring.calculate_ks(group1_truncated , group2_truncated)

# COMMAND ----------

p, q = monitoring.calculate_probability_vector(group1_truncated , group2_truncated)

# COMMAND ----------

monitoring.calculate_js_distance(p, q, group1_truncated, group2_truncated, threshold=0.2) 

# COMMAND ----------

final_numeric_cols = ['t_price', 'u_income']
monitoring.generate_percent_change(group1, group2, final_numeric_cols)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC The metrics seem to have many of their stats changed significantly, so we would want to look into those. Now run the KS test on the two subsets of the data. However, we cannot use the default alpha level of 0.05 in this situation because we are running a group of tests. This is because the probability of at least one false positive (concluding the feature's distribution changed when it did not) in a group of tests increases with the number of tests in the group. 
# MAGIC
# MAGIC To solve this problem we will employ the **Bonferroni Correction**. This changes the alpha level to 0.05 / number of tests in group. It is common practice and reduces the probability of false positives. 

# COMMAND ----------

# Set the Bonferroni Corrected alpha level
monitoring.handle_numeric_ks(group1, group2, final_numeric_cols, alpha=.05)

# COMMAND ----------

monitoring.handle_numeric_js(group1, group2, final_numeric_cols, js_stat_threshold=0.2)

# COMMAND ----------


