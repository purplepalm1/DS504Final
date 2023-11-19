

import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

start_time = time.time()
working = pd.read_csv("cleaned_data.csv", low_memory=False)
end_time = time.time()
time2read = end_time - start_time
print(f'It takes {time2read} seconds to read the csv file')

print(working.info())

# Data shape
print(f'The Length of the data is: {working.shape}')

# Loan Status

def plot_value_counts(data, column_name):

    # Calculate value counts for unique values in the specified column
    value_counts = data[column_name].value_counts()

    plt.figure(figsize=(10, 6))
    plt.bar(value_counts.index, value_counts.values, color='skyblue')
    plt.xlabel(column_name)
    plt.ylabel('Count')
    plt.title(f'Number of Instances for Each Unique Value in {column_name}')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
    plt.tight_layout()
    plt.show()

    print("Value Counts:")
    print(value_counts)



# LOAN DEFINITIONS
# Default is when not current for more than 120 days
# Charged off is final straw, no mas payments

column_to_analyze = "loan_status"
plot_value_counts(working, column_to_analyze)




# Calculate default rates based on loan grade
loan_grade_default_rates = working.groupby('sub_grade')['loan_status'].value_counts(normalize=True).unstack().loc[:, 'Charged Off']
loan_grade_default_rates = loan_grade_default_rates.sort_index(ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x=loan_grade_default_rates.index, y=loan_grade_default_rates.values, palette='viridis')
plt.xlabel('Loan Grade')
plt.ylabel('Default Rate')
plt.title('Default Rates Based on Loan Grade')
plt.show()

# Calculate default rates based on employment length
emp_length_default_rates = working.groupby('emp_length')['loan_status'].value_counts(normalize=True).unstack().loc[:, 'Charged Off']

# Sort employment lengths for better visualization
emp_length_default_rates = emp_length_default_rates.reindex(['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'])


plt.figure(figsize=(10, 6))
sns.barplot(x=emp_length_default_rates.index, y=emp_length_default_rates.values, palette='viridis')
plt.xlabel('Employment Length')
plt.ylabel('Default Rate')
plt.title('Default Rates Based on Employment Length')
plt.xticks(rotation=45)
plt.show()

label_encoder = LabelEncoder()
working['loan_status_encoded'] = label_encoder.fit_transform(working['loan_status'])

# Select numeric columns for correlation analysis
numeric_columns = working.select_dtypes(include=[np.number]).columns.tolist()

# Correlation matrix
correlation_matrix = working[numeric_columns].corr()

# Calculate correlations with the encoded 'loan_status' column
correlations_with_default = correlation_matrix['loan_status_encoded'].drop('loan_status_encoded')

sorted_correlations = correlations_with_default.abs().sort_values(ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x=sorted_correlations.index, y=sorted_correlations.values, palette='viridis')
plt.xlabel('Numeric Features')
plt.ylabel('Correlation with Default Status (Encoded)')
plt.title('Correlation Analysis with Default Status')
plt.xticks(rotation=45)
plt.show()

# Interest rates vs. default status
plt.figure(figsize=(8, 6))
sns.boxplot(x='loan_status', y='int_rate', data=working, palette='viridis')
plt.xlabel('Loan Status')
plt.ylabel('Interest Rate')
plt.title('Interest Rates vs. Default Status')
plt.xticks(rotation=45)
plt.show()


# Calculate default rates based on loan purpose
loan_purpose_default_rates = working.groupby('purpose')['loan_status'].value_counts(normalize=True).unstack().loc[:, 'Charged Off']

# Sort by default rate to identify loan purposes with the most defaults
loan_purpose_default_rates = loan_purpose_default_rates.sort_values(ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x=loan_purpose_default_rates.index, y=loan_purpose_default_rates.values, palette='viridis')
plt.xlabel('Loan Purpose')
plt.ylabel('Default Rate')
plt.title('Default Rates Based on Loan Purpose')
plt.xticks(rotation=45)
plt.show()


# Select data for weddings, small businesses, and moving loan purposes
selected_loan_purposes = ['wedding', 'small_business', 'moving']
selected_data = working[working['purpose'].isin(selected_loan_purposes)]

# Strip plot of loan amount vs loan purpose
plt.figure(figsize=(12, 8))
sns.stripplot(x='purpose', y='loan_amnt', data=selected_data, jitter=True, palette='viridis')
plt.xlabel('Loan Purpose')
plt.ylabel('Loan Amount')
plt.title('Loan Amount Distribution for Weddings, Small Businesses, and Moving')
plt.xticks(rotation=45)
plt.show()

# Summary statistics for debt-to-income (DTI) ratios based on loan status
dti_summary = working.groupby('loan_status')['dti'].describe().reset_index()
dti_summary.columns = ['Loan Status', 'Count', 'Mean DTI', 'Std Dev DTI', 'Min DTI', '25th Percentile DTI', 'Median DTI', '75th Percentile DTI', 'Max DTI']
dti_summary.to_excel('dti_summary.xlsx', index=False)


# loan status vs mean total payment received
plt.figure(figsize=(12, 8))
sns.pointplot(x='loan_status', y='total_pymnt', data=working, ci="sd", capsize=0.2, palette='Set2')
plt.xlabel('Loan Status')
plt.ylabel('Mean Total Payment Received')
plt.title('Loan Status vs Mean Total Payment Received with Error Bars')
plt.show()

