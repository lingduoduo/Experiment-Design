# Databricks notebook source
# MAGIC %md
# MAGIC ## Read-in Data
# MAGIC

# COMMAND ----------

import dlt
from pyspark.sql import functions as F
from scipy.stats import gaussian_kde, truncnorm
from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math

# COMMAND ----------

new_tx = spark.sql("""select * from data_science_prod.default.transactions_users_silver""")

# COMMAND ----------

new_tx.count()

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
# MAGIC ### Check Schema

# COMMAND ----------

new_tx.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Check Data Profiling

# COMMAND ----------

dbutils.data.summarize(new_tx)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Identify Missing Values

# COMMAND ----------

## dump the data with missing values
condition = F.lit(False)
for column in new_tx.columns:
    condition = condition | F.col(column).isNull()

# Filter the DataFrame based on the condition
new_tx_missing = new_tx.filter(condition)

# COMMAND ----------

display(new_tx_missing)

# COMMAND ----------

all_missing_cols = []
for column in new_tx.columns:
    non_missing_count = new_tx.filter(new_tx[column].isNotNull()).count()
    is_all_missing = (non_missing_count == 0)
    if is_all_missing:
        all_missing_cols.append(column)
        print(f"All values in '{column}' are missing: {is_all_missing}")

# COMMAND ----------

# Calculate the total number of rows in the DataFrame for percentage calculation
total_rows = new_tx.count()
d = {}
for column in new_tx.columns:
    # Count the number of non-missing (nonnull) entries in the column
    non_missing_count = new_tx.filter(F.col(column).isNotNull()).count()
    missing_percentage = ((total_rows - non_missing_count) / total_rows) * 100
    if missing_percentage > 0 and missing_percentage < 100:  
        d[column] = missing_percentage  
        print(f"Column '{column}' is {missing_percentage:.2f}% missing.")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Split the Features into Different Categories

# COMMAND ----------

from pyspark.sql.types import StringType, BooleanType, LongType, TimestampType, NumericType

timestamp_cols = [field.name for field in new_tx.schema.fields if isinstance(field.dataType, (TimestampType)) or 'timestamp' in field.name.lower()]
print(f"Timestamp cols: {timestamp_cols}")

date_cols = [field.name for field in new_tx.schema if 'date' in field.name.lower() or 'year_month' in field.name.lower()]
print(f"DateTime_cols: {date_cols }")

identifier_cols = [field.name for field in new_tx.schema if 'id' in field.name.lower() or 'alias' in field.name.lower()]
print(f"Identifier cols: {identifier_cols}")

location_cols = [field.name for field in new_tx.schema if 'latitude' in field.name.lower() or 'longitude' in field.name.lower()]
print(f"location_cols: {location_cols}")

# COMMAND ----------

# Identify categorical columns (StringType and BooleanType)
categorical_cols = [field.name for field in new_tx.schema.fields if isinstance(field.dataType, (StringType, BooleanType))]

# Define columns to exclude
exclude_cols = all_missing_cols +  timestamp_cols + date_cols + location_cols + identifier_cols 
include_cols = ['t_merchant_category_code', 'u_num_children']

# Exclude specified columns from the list of categorical columns
final_categorical_cols = [col for col in set(categorical_cols + include_cols) - set(exclude_cols) if col in d and d[col] < 20] 
for field in final_categorical_cols:
  print(f"Final Categorical Columns: {field}")

final_categorical_cols = ['u_city', 
                          'u_relationship_status', 
                          'u_sexuality', 
                          'u_home_location', 
                          'u_num_children', 
                          'u_yearly_household_income_group', 
                          'u_gender_group', 
                          'u_ethnicity', 
                          'u_state', 
                          'u_age_group', 
                          'u_zipcode', 
                          't_merchant_name', 
                          't_merchant_url']

# COMMAND ----------

numeric_cols = [field.name for field in new_tx.schema.fields if isinstance(field.dataType, (LongType,  NumericType))]

numeric_cols = [col for col in numeric_cols if col not in exclude_cols + timestamp_cols + location_cols] 
print("Numeric Columns:", numeric_cols)

final_numeric_cols = ['t_price', 'u_income']
print("Final Numeric Columns:", final_numeric_cols)

# COMMAND ----------

binary_cols = list(filter(lambda f: f.startswith('t_is'), new_tx.columns))
print("Binary Columns:", binary_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Examine binary distribution

# COMMAND ----------

# Determine required rows and columns
plt_df_binary = new_tx[binary_cols].toPandas()
b_feature_count = len(binary_cols)
b_column_count = 3
b_row_count = math.ceil(b_feature_count / b_column_count)

# Create a figure to hold the subplots
plt.figure(figsize=(b_column_count * 3, b_row_count * 3))

for k in range(b_feature_count):
    b_col = k % b_column_count
    b_row = k // b_column_count
    
    f = binary_cols[k]
    print(f"{k}: {f}")

    value_counts = plt_df_binary[f].value_counts()
    plt.subplot(b_row_count, b_column_count, k + 1)
    plt.pie(
        x=value_counts.values,
        labels=value_counts.index,
        explode=None,  # Modify as needed
        autopct='%1.1f%%',
        labeldistance=None,  # Modify as needed
        # pctdistance=0.4,  # Uncomment and modify as needed
        frame=True,  # This might not have an effect in plt.pie
        radius=0.48,  # Adjust size as needed
        center=(0.5, 0.5)  # This might not be applicable in plt.pie
    )
    plt.title(f)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Data distribution anomalies

# COMMAND ----------

dbutils.data.summarize(new_tx[final_numeric_cols])

# COMMAND ----------

df_filtered_outliers = new_tx
for column in final_numeric_cols:
    df_filtered_outliers = monitoring.get_truncated_right_skewed_values(df_filtered_outliers, column)

# COMMAND ----------

plt_df_before = new_tx[final_numeric_cols].toPandas()
plt_df_after = df_filtered_outliers[final_numeric_cols].toPandas()

# COMMAND ----------

import matplotlib.pyplot as plt

for k in range(len(final_numeric_cols)):
    print(f"Plot Before Truncated Outliers: {final_numeric_cols[k]}")
    monitoring.plot_single_distribution(plt_df_before[final_numeric_cols[k]])
    plt.show()
    
    print(f"Plot After Truncated Outliers: {final_numeric_cols[k]}")
    monitoring.plot_single_distribution(plt_df_after[final_numeric_cols[k]])
    plt.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC ##  Default Values using Categorical Data

# COMMAND ----------

display(new_tx.groupBy('u_gender').mean('u_income'))

# COMMAND ----------

display(new_tx.groupBy('u_age_group').mean('u_income'))

# COMMAND ----------

display(new_tx.groupBy('u_state').mean('u_income'))

# COMMAND ----------


