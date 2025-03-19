import pandas as pd
import numpy as np

# Load data
df = pd.read_pickle('submission2/data/output/TaxBurden_Data.pkl')

# Create log variables
df['log_sales'] = np.log(df['sales_per_capita'])
df['log_price'] = np.log(df['cost_per_pack'])

# Filter datasets by year
df_70_90 = df[(df['Year'] >= 1970) & (df['Year'] <= 1990)].copy()
df_91_15 = df[(df['Year'] >= 1991) & (df['Year'] <= 2015)].copy()

from pyfixest.estimation import feols

# OLS regression: log_sales ~ log_price
ols_70_90 = feols('log_sales ~ log_price', data=df_70_90)
print(ols_70_90.summary())

from pyfixest.estimation import feols

# IV regression: instrument log_price with tax_dollar
iv_70_90 = feols('log_sales ~ 1 | log_price ~ tax_dollar', data=df_70_90)
print(iv_70_90.summary())

#Q8

first_stage = feols('log_price ~ tax_dollar', data=df_70_90)
print(first_stage.summary())
reduced_form = feols('log_sales ~ tax_dollar', data=df_70_90)
print(reduced_form.summary())

# Q9

ols_91_15 = feols('log_sales ~ log_price', data=df_91_15)
print(ols_91_15.summary())
iv_91_15 = feols('log_sales ~ 1 | log_price ~ tax_dollar', data=df_91_15)
print(iv_91_15.summary())
