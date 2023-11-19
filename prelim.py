

import pandas as pd
import time
import numpy as np


start_time = time.time()
raw = pd.read_csv("lc_loan_training.csv", low_memory=False)
end_time = time.time()
time2read = end_time - start_time
print(f'It takes {time2read} seconds to read the csv file')

print(raw.info())

# Data shape
print(f'The Length of the data is: {raw.shape}')
# The Length of the data is (887,379 x 74)

# Determine missing values amount
pd.set_option('display.max_rows', None)

missing_vals_percentage = (raw.isnull().sum() / len(raw)) * 100
cols_80_missing = missing_vals_percentage[missing_vals_percentage >= 80]

print("Columns with 80% or more missing values:")
print(cols_80_missing)

# Drop columns with 80% or more missing values, not practical to fill them in
cols_to_drop = ['id', 'member_id', ] + list(cols_80_missing.index)
raw.drop(columns=cols_to_drop, inplace=True)
print(raw.info())



# Additional value elimination


# print(raw.emp_title.nunique()) # Too many unique job titles (299,271), just remove column
raw.drop('emp_title', axis=1, inplace=True)

# emp_length showcases length of employment - 44,825 missing values
uq_emp_lengths = raw['emp_length'].unique()
print(f'Unique values in emp_length column is: {uq_emp_lengths}') # shows a set of range years
mode_emp_length = raw['emp_length'].mode()[0]
print(mode_emp_length) # Mode is 10+ years, so we will fill in missing values with this figure
raw['emp_length'].fillna(mode_emp_length, inplace=True)



# Irrelvant data for us - elimnate
add_cols_to_drop = ['last_pymnt_d', 'url' ,'next_pymnt_d', 'mths_since_last_delinq', 'mths_since_last_major_derog', 'title']
raw.drop(columns=add_cols_to_drop, inplace=True)


# Three columns are missing 70,276 values in each
# remove the (70,276) instances where there is no data in the tot_coll_amt, tot_cur_bal, and total_rev_hi_lim.
# These are good data points, and instead of deleting the whole columns, lets just remove the missing rows
raw.dropna(subset=['tot_cur_bal', 'total_rev_hi_lim', 'tot_coll_amt'], inplace=True)


# Revol_util is missing 381 values...fill it in with median
raw['revol_util'].fillna(raw['revol_util'].median(), inplace=True)

# last_credit_pull_d is missing 49 values, immaterial and will delete those rows
raw.dropna(subset=['last_credit_pull_d'], inplace=True)

missing_values = raw.isnull().sum()
print(missing_values) # no missing values anymore

print(raw.info())

# Converting object data into float64
object_cols = raw.select_dtypes(include=['object'])

# Example of each object
# for column in object_cols.columns:
#     unique_values = raw[column].unique()
#     sample_value = raw[column].iloc[0]
#     print(f"Column: {column}")
#     print(f"Unique Values: {unique_values}")
#     print(f"Sample Value: {sample_value}")
#     print("-" * 50)

# Changing term from object to int
term_mapping = {
    ' 36 months': 36,
    ' 60 months': 60}
raw['term'] = raw['term'].map(term_mapping)

# Dropping grade column because it is embedded in the subgrade, i.e. why have A when sub is A2
raw.drop(columns=['grade'], inplace=True)

# Changing pymnt_plan from (y, n) to (1, 0)
pymnt_plan_mapping = {
    'n': 0,
    'y': 1
}
raw['pymnt_plan'] = raw['pymnt_plan'].map(pymnt_plan_mapping)

# Removing the months from 'earliest_cr_line column
raw['earliest_cr_line'] = raw['earliest_cr_line'].str.extract(r'(\d{4})')

print(raw.info())

# Writing data to new csv
raw.to_csv("cleaned_data.csv", index=False)

"""
Exploratory Data Analysis
"""

# print(raw.info())



