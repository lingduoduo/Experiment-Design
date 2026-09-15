# Databricks notebook source
import math
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from pyspark.sql import functions as F
from pyspark.sql import window as w 
from pyspark.sql.types import FloatType, IntegerType

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Impute Missing Gender

# COMMAND ----------

training_gender = spark.read.format('delta').load('dbfs:/tmp/training_gender')

# COMMAND ----------

display(training_gender)

# COMMAND ----------

# from databricks import automl

# summary = automl.classify(training_gender, target_col='u_gender_group', primary_metric="f1", data_dir='dbfs:/automl/', timeout_minutes=30)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Impute Missing Income

# COMMAND ----------

trainining_income= spark.read.format('delta').load('dbfs:/tmp/trainining_income')

# COMMAND ----------

display(trainining_income)

# COMMAND ----------

# from databricks import automl

# summary = automl.classify(trainining_income, target_col='u_yearly_household_income', primary_metric="f1", data_dir='dbfs:/automl/', timeout_minutes=30)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Impute Missing Age

# COMMAND ----------

training_age = spark.read.format('delta').load('dbfs:/tmp/missing_age')

# COMMAND ----------

display(training_age)

# COMMAND ----------

# from databricks import automl

# summary = automl.classify(training_age, target_col='u_age_group', primary_metric="f1", data_dir='dbfs:/automl/', timeout_minutes=30)

# COMMAND ----------


