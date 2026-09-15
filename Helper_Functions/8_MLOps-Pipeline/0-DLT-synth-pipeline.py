# Databricks notebook source
import dlt
from pyspark.sql import functions as F

# COMMAND ----------

# DBTITLE 1,Fetch transactions data
# @dlt.create_table(comment="Fetch transaction data") 
# def trans():
#   return spark.sql(""" 
#   SELECT 
#           t_accountID,
#           t_alias,
#           t_transaction_category,
#           t_top_transaction_category,
#           t_instance_timestamp,
#           t_is_bill,
#           t_is_dd,
#           t_is_exp,
#           t_is_fee,
#           t_is_income,
#           t_is_international,
#           t_is_od_fee,
#           t_is_payroll_advance,
#           t_is_subscription,
#           t_transaction_latitude,
#           t_transaction_longitude,
#           t_transaction_id,
#           t_merchant_name,
#           t_merchant_url,
#           t_merchant_category,
#           t_merchant_category_code,
#           t_merchant_country,
#           t_merchant_locality,
#           t_merchant_region,
#           t_merchant_zipcode,
#           t_merchant_latitude,
#           t_merchant_longitude,
#           t_merchant_address,
#           t_merchant_telephone,
#           t_price,
#           t_currency,
#           t_status,
#           t_merchant_ID,
#           t_transaction_type,
#           t_action,
#           t_transaction_time,
#           DATE_FORMAT(t_transaction_time, 'yyyy-MM-dd') AS t_transaction_date,
#           CONCAT(year(t_transaction_time), LPAD(month(t_transaction_time), 2, '0')) AS t_transaction_year_month
#       FROM os_production.os_silver.transactions_silver
#       WHERE t_transaction_time > (current_timestamp() - INTERVAL 30 DAYS)
#   """).alias("trans")

# COMMAND ----------

# DBTITLE 1,Fetch user data
# @dlt.create_table(comment="Fetch user data")
# def users():
#   return spark.sql(""" 
#     WITH user_age_income AS (
#     SELECT 
#         u_alias,
#         u_sign_up_date,
#         u_num_children,
#         u_birth_date,
#         u_education_level,
#         u_employment_status,
#         u_gender,
#         u_hispanic_latino_origin,
#         u_voter_county,
#         u_voter_county_code,
#         u_home_location,
#         u_party_affiliation,
#         u_party_registration,
#         u_political_ideology,
#         u_is_registered_voter,
#         u_relationship_status,
#         u_sexuality,
#         u_work_location,
#         u_yearly_household_income,
#         u_yearly_salary,
#         u_instance_timestamp,
#         u_zipcode,
#         u_city,
#         u_state,
#         u_ethnicities,
#         FLOOR(DATEDIFF(CURRENT_TIMESTAMP, u_birth_date) / 365) AS age
#     FROM os_production.os_silver.users_silver
#     )
#     SELECT *,
#         CASE
#             WHEN lower(u_gender) not in ('nonbinary', 'female', 'male') THEN 'none'
#             ELSE u_gender
#         END AS u_gender_group,
#         CASE
#             WHEN age < 0 or age > 120 THEN 'none'
#             WHEN age < 18 THEN '18-'
#             WHEN age >= 18 AND age <= 24 THEN '18-24'
#             WHEN age >= 25 AND age <= 34 THEN '25-34'
#             WHEN age >= 35 AND age <= 44 THEN '35-44'
#             WHEN age >= 45 AND age <= 54 THEN '45-54'
#             WHEN age >= 55 AND age <= 64 THEN '55-64'
#             WHEN age >= 65 THEN '65+'
#         END AS u_age_group,
#         CASE
#             WHEN u_yearly_household_income == '$0-$19,999' THEN 10000
#             WHEN u_yearly_household_income == '$20,000-$49,999' THEN 35000
#             WHEN u_yearly_household_income == '$50,000-$89,999' THEN 70000
#             WHEN u_yearly_household_income == '$90,000-$149,999' THEN 120000
#             WHEN u_yearly_household_income == '$150,000+' THEN 200000
#         END AS u_income,
#         CASE
#             WHEN size(u_ethnicities) = 1 THEN u_ethnicities[0]
#             WHEN size(u_ethnicities) = 0 THEN 'none'
#             ELSE
#                 CASE
#                     WHEN u_hispanic_latino_origin AND size(u_ethnicities) = 0 THEN 'hispanic'
#                     WHEN u_hispanic_latino_origin AND u_ethnicities[0] != 'hispanic' THEN 'mixed'
#                     ELSE 'mixed'
#                 END
#             END AS u_ethnicity
#     FROM user_age_income
#   """).alias("users")

# COMMAND ----------

# DBTITLE 1,Merge Transactions and Users
# @dlt.create_table(comment="Enrich transactions with users")
# def transactions_users_silver():
#   trans = spark.read.format("delta").table("data_science_prod.default.trans")
#   users = spark.read.format("delta").table("data_science_prod.default.users")

@dlt.create_table(comment="Fetch transaction data") 
def transactions_users_silver():
  return spark.sql(""" 
    WITH user_age_income AS (
        SELECT 
            u_alias,
            u_sign_up_date,
            u_num_children,
            u_birth_date,
            u_education_level,
            u_employment_status,
            u_gender,
            u_hispanic_latino_origin,
            u_voter_county,
            u_voter_county_code,
            u_home_location,
            u_party_affiliation,
            u_party_registration,
            u_political_ideology,
            u_is_registered_voter,
            u_relationship_status,
            u_sexuality,
            u_work_location,
            u_yearly_household_income,
            u_yearly_salary,
            u_instance_timestamp,
            u_zipcode,
            u_city,
            u_state,
            u_ethnicities,
            FLOOR(DATEDIFF(CURRENT_TIMESTAMP, u_birth_date) / 365) AS age
        FROM os_production.os_silver.users_silver
    ),
    users AS (
        SELECT *,
              CASE
                  WHEN lower(u_gender) NOT IN ('nonbinary', 'female', 'male') THEN 'none'
                  ELSE u_gender
              END AS u_gender_group,
              CASE
                  WHEN age < 0 OR age > 120 THEN 'none'
                  WHEN age < 18 THEN '18-'
                  WHEN age >= 18 AND age <= 24 THEN '18-24'
                  WHEN age >= 25 AND age <= 34 THEN '25-34'
                  WHEN age >= 35 AND age <= 44 THEN '35-44'
                  WHEN age >= 45 AND age <= 54 THEN '45-54'
                  WHEN age >= 55 AND age <= 64 THEN '55-64'
                  WHEN age >= 65 THEN '65+'
              END AS u_age_group,
              CASE
                  WHEN u_yearly_household_income = '$0-$19,999' THEN 10000
                  WHEN u_yearly_household_income = '$20,000-$49,999' THEN 35000
                  WHEN u_yearly_household_income = '$50,000-$89,999' THEN 70000
                  WHEN u_yearly_household_income = '$90,000-$149,999' THEN 120000
                  WHEN u_yearly_household_income = '$150,000+' THEN 200000
              END AS u_income,
              CASE
                  WHEN size(u_ethnicities) = 1 THEN u_ethnicities[0]
                  WHEN size(u_ethnicities) = 0 THEN 'none'
                  ELSE
                      CASE
                          WHEN u_hispanic_latino_origin AND size(u_ethnicities) = 0 THEN 'hispanic'
                          WHEN u_hispanic_latino_origin AND u_ethnicities[0] != 'hispanic' THEN 'mixed'
                          ELSE 'mixed'
                      END
              END AS u_ethnicity
        FROM user_age_income
    ),
    trans AS (
        SELECT 
            t_accountID,
            t_alias,
            t_transaction_category,
            t_top_transaction_category,
            t_instance_timestamp,
            t_is_bill,
            t_is_dd,
            t_is_exp,
            t_is_fee,
            t_is_income,
            t_is_international,
            t_is_od_fee,
            t_is_payroll_advance,
            t_is_subscription,
            t_transaction_latitude,
            t_transaction_longitude,
            t_transaction_id,
            t_merchant_name,
            t_merchant_url,
            t_merchant_category,
            t_merchant_category_code,
            t_merchant_country,
            t_merchant_locality,
            t_merchant_region,
            t_merchant_zipcode,
            t_merchant_latitude,
            t_merchant_longitude,
            t_merchant_address,
            t_merchant_telephone,
            t_price,
            t_currency,
            t_status,
            t_merchant_ID,
            t_transaction_type,
            t_action,
            t_transaction_time,
            DATE_FORMAT(t_transaction_time, 'yyyy-MM-dd') AS t_transaction_date,
            CONCAT(year(t_transaction_time), LPAD(month(t_transaction_time), 2, '0')) AS t_transaction_year_month
        FROM os_production.os_silver.transactions_silver
    )
    SELECT 
        trans.*,
        users.*
    FROM trans
    JOIN users ON users.u_alias = trans.t_alias;
    """)

# COMMAND ----------


