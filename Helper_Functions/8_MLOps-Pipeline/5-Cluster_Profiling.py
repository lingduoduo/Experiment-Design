# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/segmentation.git. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/customer-segmentation.

# COMMAND ----------

# MAGIC %md The purpose of this notebook is to better understand the clusters generated in the prior notebook leveraging some standard profiling techniques. 

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import pandas as pd
import numpy as np

import statsmodels.api as sm
from statsmodels.graphics.mosaicplot import mosaic

import math

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from pyspark.sql.functions import expr

# COMMAND ----------

# MAGIC %md ## Step 1: Assemble Segmented Dataset
# MAGIC
# MAGIC We now have clusters but we're not really clear on what exactly they represent.  The feature engineering work we performed to avoid problems with the data that might lead us to invalid or inappropriate solutions have made the data very hard to interpret.  
# MAGIC
# MAGIC To address this problem, we'll retrieve the cluster labels (assigned to each household) along with the original features associated with each:

# COMMAND ----------

# DBTITLE 1,Retrieve Features & Labels
labeled_pd = spark.read.format('delta').load('dbfs:/tmp/df_labeled')

# COMMAND ----------

display(labeled_pd)

# COMMAND ----------

# DBTITLE 1,Set Cluster Design to Analyze
cluster_column = 'cluster'
labeled_pdf = labeled_pd.toPandas()
cluster_count = len(np.unique(labeled_pdf[cluster_column]))
cluster_colors = [cm.nipy_spectral(float(i)/cluster_count) for i in range(cluster_count)]

# COMMAND ----------

# MAGIC %md ## Step 2: Profile Segments
# MAGIC
# MAGIC To get us started, let's revisit the 2-dimensional visualization of our clusters to get us oriented to the clusters.  The color-coding we use in this chart will be applied across our remaining visualizations to make it easier to determine the cluster being explored:

# COMMAND ----------

# MAGIC %md  The segment design we came up with does not produce equal sized groupings.  Instead, we have one group a bit larger than the others, though the smaller groups are still of a size where they are useful to our team:

# COMMAND ----------

# DBTITLE 1,Count Cluster Members
# count members per cluster
cluster_member_counts = labeled_pdf.groupby([cluster_column]).agg({cluster_column:['count']})
cluster_member_counts.columns = cluster_member_counts.columns.droplevel(0)

# plot counts
plt.bar(
  cluster_member_counts.index,
  cluster_member_counts['count'],
  color = cluster_colors,
  tick_label=cluster_member_counts.index
  )

# stretch y-axis
plt.ylim(0,labeled_pdf.shape[0])

# labels
for index, value in zip(cluster_member_counts.index, cluster_member_counts['count']):
    plt.text(index, value, str(value)+'\n', horizontalalignment='center', verticalalignment='baseline')

# COMMAND ----------

# MAGIC %md Let's now examine how each segment differs relative to our base features.  For our categorical features, we'll plot the proportion of cluster members identified as participating in a specific promotional activity relative to the overall number of cluster members. For our continuous features, we will visualize values using a whisker plot:

# COMMAND ----------

# DBTITLE 1,Define Function to Render Plots
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd  # Assuming pandas is used for data manipulation

def profile_segments_by_features(data, features_to_plot, label_to_plot, label_count, label_colors):
  fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharey=True)  # Adjust subplot grid as needed

  for ax, feat in zip(axs.flatten(), features_to_plot):
      # Calculate normalized count using crosstab, then plot
      normalized_ct = (pd.crosstab(data[label_to_plot], data[feat]) / pd.crosstab(data[label_to_plot], data[feat]).sum()).T
      normalized_ct.plot.bar(stacked=True, ax=ax, legend=False)
      ax.set_title(feat)

  # Adjust legend placement
  handles, labels = ax.get_legend_handles_labels()
  fig.legend(handles, labels, loc='upper right')  # You can adjust the location as needed

  plt.tight_layout()
  plt.show()


label_columns = ['cluster']

# get feature names
feature_names = labeled_pdf.drop(label_columns, axis=1).columns
print(feature_names, feature_names, cluster_column, cluster_count, cluster_colors)

feature_names = ['u_age_group', 'u_gender_group', 'u_yearly_household_income', 'u_ethnicity']

# generate plots
profile_segments_by_features(labeled_pdf, feature_names, cluster_column, cluster_count, cluster_colors)

# COMMAND ----------

fig,axs = plt.subplots(2,2,figsize=(10,10),sharex=True)
num_features = ['days_since_prior_transaction', 'days_prior_to_last_transaction','mean_transaction_window_7_day', 'count_transaction_window_7_day']
max_cols = 2
feature_count  = len(num_features)
column_count = min(feature_count, max_cols)  # Number of columns in the grid
row_count = feature_count // column_count
for ax,feat in zip(axs.flatten(),num_features):
    pd.plotting.boxplot(labeled_pdf, column=[feat],by='cluster', ax=ax)
    ax.set_xlabel('')  
plt.tight_layout()

# COMMAND ----------


