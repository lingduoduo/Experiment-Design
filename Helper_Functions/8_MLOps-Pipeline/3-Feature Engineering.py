# Databricks notebook source
# MAGIC %pip install dython==0.7.1
# MAGIC %pip install databricks-feature-engineering
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import math
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from pyspark.sql import functions as F
from pyspark.sql import window as w 
from pyspark.sql.types import FloatType, IntegerType

# COMMAND ----------

new_tx = spark.sql("""select * from data_science_prod.default.transactions_users_silver
                     where t_instance_timestamp/1000 > (UNIX_TIMESTAMP(current_timestamp()) - 30 * 24 * 60 * 60); """)

# COMMAND ----------

display(new_tx)

# COMMAND ----------

# DBTITLE 1,Enhanced Features
win_asc = w.Window.partitionBy('t_alias').orderBy(F.col('t_transaction_time'))
win_desc = w.Window.partitionBy('t_alias').orderBy(F.col('t_transaction_time').desc())
new_tx_enhanced = new_tx.withColumn(
    'days_since_prior_transaction',
    F.coalesce(  # Replace nulls with a default value, e.g., 0
        F.datediff(
            F.col('t_transaction_time'),
            F.lag('t_transaction_time', 1).over(win_asc)
        ),
        F.lit(0)
    )
)

new_tx_enhanced = (
    new_tx_enhanced
      .withColumn(
        'days_prior_to_last_transaction', 
        F.sum('days_since_prior_transaction').over(win_asc) - F.coalesce(F.col('days_since_prior_transaction'), F.lit(0))
        )
)

# COMMAND ----------

new_tx_users = (new_tx_enhanced.withColumn('row_num', F.row_number().over(win_desc))
                .filter(F.col('row_num') == 1)
                .select(
                    'u_alias',
                    'u_age_group',
                    'u_gender_group',
                    'u_yearly_household_income',
                    'u_ethnicity',
                    'u_zipcode',
                    'u_state',
                    'days_since_prior_transaction',
                    'days_prior_to_last_transaction',
                    't_transaction_time'
                ))

enhanced_trans_l7 = (
    new_tx.groupBy(
        "t_alias", F.window("t_transaction_time", "7 days")
    )  
    .agg(
        F.mean("t_price").alias("mean_transaction_window_7_day"),  
        F.count("*").alias("count_transaction_window_7_day"),
    )
    .select(
        "t_alias",
        F.unix_timestamp(F.col("window.end")).cast("timestamp").alias("window_end_ts"),
        F.col("mean_transaction_window_7_day").cast(FloatType()),  
        F.col("count_transaction_window_7_day").cast(IntegerType()),
    ))

new_tx_df = new_tx_users.join(enhanced_trans_l7, 
    (new_tx_users.u_alias == enhanced_trans_l7.t_alias) &
    (new_tx_users.t_transaction_time == enhanced_trans_l7.window_end_ts)
    ).select(
        'u_alias',
        'u_age_group',
        'u_gender_group',
        'u_yearly_household_income',
        'u_ethnicity',
        'u_zipcode',
        'u_state',
        'days_since_prior_transaction',
        'days_prior_to_last_transaction',
        'mean_transaction_window_7_day',
        'count_transaction_window_7_day'
    )

# COMMAND ----------

display(new_tx_df)

# COMMAND ----------

# dbutils.fs.rm("dbfs:/tmp/new_tx_enhanced", recurse=True)
new_tx_df.write.format('delta').mode('overwrite').save('dbfs:/tmp/new_tx_enhanced')

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Store Missing Gender Data

# COMMAND ----------

new_tx_users = spark.read.format('delta').load('dbfs:/tmp/new_tx_enhanced')

# COMMAND ----------

display(new_tx_users.groupBy('u_gender_group').count())

# COMMAND ----------

missing_gender = new_tx_users.filter((F.col('u_gender_group') == 'none') | (F.col('u_gender_group').isNull()))

# COMMAND ----------

# dbutils.fs.rm('dbfs:/tmp/missing_gender', recurse=True)
missing_gender.write.format('delta').mode('overwrite').save('dbfs:/tmp/missing_gender')

# COMMAND ----------

training_gender = new_tx_users.filter((F.col('u_gender_group') != 'none') & (F.col('u_gender_group').isNotNull()))

# COMMAND ----------

# dbutils.fs.rm('dbfs:/tmp/training_gender', recurse=True)
training_gender.write.format('delta').mode('overwrite').save('dbfs:/tmp/training_gender')

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Store Missing Income Data

# COMMAND ----------

new_tx_users = spark.read.format('delta').load('dbfs:/tmp/new_tx_enhanced')

# COMMAND ----------

display(new_tx_users.groupBy('u_yearly_household_income').count())

# COMMAND ----------

census_state_income = spark.read.format('csv').option("header", "true").load('dbfs:/tmp/Income By State.csv')
census_state_income = (census_state_income.withColumnRenamed('u_state', 'state').withColumnRenamed('Median Household Income (2021)', 'state_median_income'))

# COMMAND ----------

new_tx_users_income = (new_tx_users.join(census_state_income, new_tx_users.u_state == census_state_income.state).select(
        'u_alias',
        'u_age_group',
        'u_gender_group',
        'u_yearly_household_income',
        'u_ethnicity',
        'u_zipcode',
        'u_state',
        'days_since_prior_transaction',
        'days_prior_to_last_transaction',
        'mean_transaction_window_7_day',
        'count_transaction_window_7_day',
        'state_median_income'
    ))

# COMMAND ----------

# dbutils.fs.rm("dbfs:/tmp/new_tx_income", recurse=True)
new_tx_users_income.write.format('delta').mode('overwrite').save('dbfs:/tmp/new_tx_income')

# COMMAND ----------

display(new_tx_users_income.groupBy('u_yearly_household_income').count())

# COMMAND ----------

missing_income = new_tx_users_income.filter((F.col('u_yearly_household_income') == 'preferNotToSay') | (F.col('u_yearly_household_income').isNull()))

# COMMAND ----------

# dbutils.fs.rm('dbfs:/tmp/missing_income', recurse=True)
missing_income.write.format('delta').mode('overwrite').save('dbfs:/tmp/missing_income')

# COMMAND ----------

trainining_income = new_tx_users_income.filter((F.col('u_yearly_household_income') != 'preferNotToSay') & (F.col('u_yearly_household_income').isNotNull()))

# COMMAND ----------

# dbutils.fs.rm('dbfs:/tmp/trainining_income', recurse=True)
trainining_income.write.format('delta').mode('overwrite').save('dbfs:/tmp/trainining_income')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Impute Missing Age Data

# COMMAND ----------

new_tx_users = spark.read.format('delta').load('dbfs:/tmp/new_tx_enhanced')

# COMMAND ----------

display(new_tx_users)

# COMMAND ----------

display(new_tx_users.groupBy('u_age_group').count())

# COMMAND ----------

missing_age = new_tx_users.filter((F.col('u_age_group').isNull()))

# COMMAND ----------

# dbutils.fs.rm('dbfs:/tmp/missing_age', recurse=True)
missing_age.write.format('delta').mode('overwrite').save('dbfs:/tmp/missing_age')

# COMMAND ----------

training_age = new_tx_users.filter((F.col('u_age_group').isNotNull()))

# COMMAND ----------

# dbutils.fs.rm('dbfs:/tmp/trainining_age', recurse=True)
training_age.write.format('delta').mode('overwrite').save('dbfs:/tmp/trainining_age')

# COMMAND ----------


