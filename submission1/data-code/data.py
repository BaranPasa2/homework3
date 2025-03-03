import pandas as pd
import numpy as np
import os

# Install packages if needed - in Python we typically use pip or conda
# pip install pandas numpy matplotlib seaborn openpyxl

# Load the data
cig_data = pd.read_csv("submission1/data/input/taxTobaccoData.csv")
cpi_data = pd.read_excel("submission1/data/input/CPI_1913_2019.xlsx", skiprows=11)

# Clean tobacco data
cig_data['measure'] = cig_data['SubMeasureDesc'].map({
    "Average Cost per pack": "cost_per_pack",
    "Cigarette Consumption (Pack Sales Per Capita)": "sales_per_capita",
    "Federal and State tax as a Percentage of Retail Price": "tax_percent",
    "Federal and State Tax per pack": "tax_dollar",
    "Gross Cigarette Tax Revenue": "tax_revenue",
    "State Tax per pack": "tax_state"
})

cig_data = cig_data.rename(columns={
    'LocationAbbr': 'state_abb',
    'LocationDesc': 'state',
    'Data_Value': 'value'
})

cig_data = cig_data[['state_abb', 'state', 'Year', 'value', 'measure']]

# Pivot wider - equivalent to pivot_wider in R
final_data = cig_data.pivot_table(
    index=['state', 'Year'],
    columns='measure',
    values='value'
).reset_index()

# Sort the data
final_data = final_data.sort_values(['state', 'Year'])

# Clean CPI data
# Equivalent to pivot_longer in R
cpi_data_melted = pd.melt(
    cpi_data,
    id_vars=['Year'],
    value_vars=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    var_name='month',
    value_name='index'
)

# Group by year and calculate mean index
cpi_data = cpi_data_melted.groupby('Year')['index'].mean().reset_index()

# Form final dataset
# Merge dataframes - equivalent to left_join in R
final_data = pd.merge(final_data, cpi_data, on='Year', how='left')

# Adjust to 2010 dollars
final_data['price_cpi'] = final_data['cost_per_pack'] * (218 / final_data['index'])

# Write output files
final_data.to_csv("submission1/data/output/TaxBurden_Data.txt", sep='\t', index=False)

# For RDS equivalent, we can use pickle or joblib in Python
import pickle
with open("submission1/data/output/TaxBurden_Data.pkl", 'wb') as f:
    pickle.dump(final_data, f)